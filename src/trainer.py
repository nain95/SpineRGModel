import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau
)
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from pathlib import Path
from metrics import calculate_metrics
import json
import pandas as pd
from accelerate import Accelerator
import random
import numpy as np


class SpineXrayRGTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Any,
        device: str = "cuda",
        test_dataloader: Optional[DataLoader] = None
    ):
        # Set seed for reproducibility in trainer
        seed = getattr(config, 'seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        from accelerate.utils import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=getattr(config.training, 'gradient_accumulation_steps', 1),
            mixed_precision=getattr(config.training, 'mixed_precision', 'bf16'),
            project_dir=getattr(config.experiment, 'output_dir', './outputs'),
            kwargs_handlers=[ddp_kwargs]
        )

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        self.device = self.accelerator.device

        self.num_epochs = config.training.num_epochs
        self.learning_rate = config.training.learning_rate
        self.weight_decay = config.training.weight_decay
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps
        self.max_grad_norm = config.training.max_grad_norm
        self.contrastive_weight = config.training.contrastive_weight
        self.early_stopping_patience = config.training.early_stopping_patience
        self.min_epochs = getattr(config.training, 'min_epochs', 3)  # Minimum epochs before early stopping
        self.best_val_loss = float('inf')
        self.best_val_score = 0.0  # Combined score: (CIDEr + ROUGE-L + BLEU-2) / 3
        self.patience_counter = 0
        self.early_stopped = False
        self.max_generation_samples = 2000

        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.scheduler
        )

        try:
            if hasattr(self.model, 'module') and hasattr(self.model, '_set_static_graph'):
                self.model._set_static_graph()
                if self.accelerator.is_main_process:
                    print("✓ DDP static graph enabled (fixes gradient checkpointing + LoRA)")

            if hasattr(self.model, 'module'):
                if hasattr(self.model, 'find_unused_parameters'):
                    self.model.find_unused_parameters = True
                    if self.accelerator.is_main_process:
                        print("✓ DDP find_unused_parameters enabled")

            if hasattr(self.accelerator, 'state') and hasattr(self.accelerator.state, 'ddp_handler'):
                if hasattr(self.accelerator.state.ddp_handler, 'find_unused_parameters'):
                    self.accelerator.state.ddp_handler.find_unused_parameters = True
                    if self.accelerator.is_main_process:
                        print("✓ Accelerator DDP handler configured")

            import os
            os.environ['TORCH_DDP_FIND_UNUSED_PARAMETERS'] = '1'
            if self.accelerator.is_main_process:
                print("✓ TORCH_DDP_FIND_UNUSED_PARAMETERS=1")

        except Exception as e:
            if self.accelerator.is_main_process:
                print(f"Warning: Could not configure DDP settings: {e}")

        if self.test_dataloader is not None:
            if self.accelerator.is_main_process:
                print("Preparing test dataloader...")
            self.test_dataloader = self.accelerator.prepare(self.test_dataloader)
            if self.accelerator.is_main_process:
                print("Test dataloader prepared successfully")

        self.checkpoint_dir = Path(config.experiment.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.checkpoint_dir / "best_model.pt"
        self.last_model_path = self.checkpoint_dir / "last_model.pt"

        self.current_epoch = 0
        self.global_step = 0
        self.training_stats = {
            'train_losses': [],
            'val_losses': [],
            'val_combined_score': [],
            'val_bleu1': [],
            'val_bleu2': [],
            'val_bleu3': [],
            'val_bleu4': [],
            'val_cider': [],
            'val_rougeL': [],
            'learning_rates': []
        }

    def _setup_optimizer(self) -> AdamW:
        param_groups = []

        lora_params = []
        llm_backbone_params = []
        vision_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if 'lora_' in name or 'lora.' in name:
                lora_params.append(param)
                self.accelerator.print(f"LoRA param: {name}")
            elif hasattr(self.model, 'llm') and any(name.startswith(prefix) for prefix in ['llm.', 'model.llm.']):
                llm_backbone_params.append(param)
            elif 'visual_encoder' in name or 'vision_encoder' in name or 'swin' in name:
                vision_params.append(param)
            else:
                other_params.append(param)

        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': self.learning_rate * 2.0,  # LoRA (4.2M, 3%) → 2x LR
                'weight_decay': self.weight_decay * 0.5  # Lighter regularization
            })
            self.accelerator.print(f"LoRA params: {len(lora_params)} (LR: {self.learning_rate * 2.0:.2e})")

        if llm_backbone_params:
            param_groups.append({
                'params': llm_backbone_params,
                'lr': self.learning_rate * 0.01,
                'weight_decay': self.weight_decay
            })
            self.accelerator.print(f"LLM backbone params: {len(llm_backbone_params)} (LR: {self.learning_rate * 0.01:.2e})")

        if vision_params:
            param_groups.append({
                'params': vision_params,
                'lr': self.learning_rate * 1.0,  # Vision (124M, 92%) → base LR
                'weight_decay': self.weight_decay * 0.5  # Lighter for scratch training
            })
            self.accelerator.print(f"Vision params: {len(vision_params)} (LR: {self.learning_rate:.2e})")

        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.learning_rate * 1.5,  # Projector, embeddings → 1.5x LR
                'weight_decay': self.weight_decay * 0.5
            })
            self.accelerator.print(f"Other params: {len(other_params)} (LR: {self.learning_rate * 1.5:.2e})")

        opt_config = self.config.training.optimizer
        optimizer = AdamW(
            param_groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=opt_config.betas,
            eps=opt_config.eps
        )

        return optimizer

    def _setup_scheduler(self):
        sched_config = self.config.training.scheduler
        scheduler_name = sched_config.name.lower()

        if scheduler_name == "cosine_annealing_warm_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=sched_config.T_0,
                T_mult=getattr(sched_config, 'T_mult', 1),
                eta_min=getattr(sched_config, 'eta_min', 1e-6)
            )
            self.accelerator.print(f"✓ Scheduler: CosineAnnealingWarmRestarts (T_0={sched_config.T_0})")

        elif scheduler_name == "cosine_annealing":
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.T_max,
                eta_min=getattr(sched_config, 'eta_min', 1e-6)
            )
            self.accelerator.print(f"✓ Scheduler: CosineAnnealingLR (T_max={sched_config.T_max})")

        elif scheduler_name == "cosine_with_warmup":
            batches_per_epoch = len(self.train_dataloader)
            grad_accum_steps = self.config.training.gradient_accumulation_steps
            optimizer_steps_per_epoch = batches_per_epoch // grad_accum_steps
            num_training_steps = optimizer_steps_per_epoch * self.num_epochs

            if hasattr(sched_config, 'warmup_ratio'):
                num_warmup_steps = int(num_training_steps * sched_config.warmup_ratio)
            else:
                num_warmup_steps = getattr(sched_config, 'warmup_steps', int(0.1 * num_training_steps))

            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            self.accelerator.print(
                f"✓ Scheduler: CosineWithWarmup "
                f"(batches_per_epoch={batches_per_epoch}, grad_accum={grad_accum_steps}, "
                f"optimizer_steps_per_epoch={optimizer_steps_per_epoch}, "
                f"warmup_steps={num_warmup_steps}, total_steps={num_training_steps})"
            )

        elif scheduler_name == "reduce_lr_on_plateau":
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=getattr(sched_config, 'mode', 'min'),
                factor=getattr(sched_config, 'factor', 0.5),
                patience=getattr(sched_config, 'patience', 5),
                min_lr=getattr(sched_config, 'min_lr', 1e-6)
            )
            self.accelerator.print(f"✓ Scheduler: ReduceLROnPlateau (patience={getattr(sched_config, 'patience', 5)})")

        else:
            raise ValueError(
                f"Unknown scheduler: {scheduler_name}. "
                f"Supported: cosine_annealing_warm_restarts, cosine_annealing, cosine_with_warmup, reduce_lr_on_plateau"
            )

        return scheduler

    def get_current_lr(self) -> float:
        """Get current learning rate from scheduler or optimizer."""
        if isinstance(self.scheduler, ReduceLROnPlateau):
            return self.optimizer.param_groups[0]['lr']
        else:
            return self.scheduler.get_last_lr()[0]

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()

        if hasattr(self.train_dataloader, 'batch_sampler'):
            if hasattr(self.train_dataloader.batch_sampler, 'set_epoch'):
                self.train_dataloader.batch_sampler.set_epoch(self.current_epoch)
            elif hasattr(self.train_dataloader.batch_sampler, 'sampler') and hasattr(self.train_dataloader.batch_sampler.sampler, 'set_epoch'):
                self.train_dataloader.batch_sampler.sampler.set_epoch(self.current_epoch)

        epoch_losses = {
            'total_loss': 0.0,
            'generative_loss': 0.0,
            'contrastive_loss': 0.0,
        }

        running_losses = {
            'total_loss': 0.0,
            'generative_loss': 0.0,
            'contrastive_loss': 0.0,
        }

        num_batches = len(self.train_dataloader)

        if self.accelerator.is_main_process:
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch+1}/{self.num_epochs}")
        else:
            progress_bar = self.train_dataloader

        for batch_idx, batch_data in enumerate(progress_bar):
            try:
                if batch_data is None:
                    self.accelerator.print(f"Warning: None batch at index {batch_idx}, skipping...")
                    continue

                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    images, texts = batch_data
                else:
                    self.accelerator.print(f"Warning: Invalid batch format at index {batch_idx}, skipping...")
                    continue

                outputs = self.model(images, texts)

                total_loss = (outputs['generative_loss'] +
                              self.contrastive_weight * outputs['contrastive_loss'])

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    self.accelerator.print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping...")
                    continue

                with self.accelerator.accumulate(self.model):
                    self.accelerator.backward(total_loss)

                    if self.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.config.training.scheduler.name.lower() == "cosine_with_warmup":
                    if self.accelerator.sync_gradients:
                        self.scheduler.step()

                self.accelerator.wait_for_everyone()

                total_loss_gathered = self.accelerator.gather(total_loss.detach()).mean()
                gen_loss_gathered = self.accelerator.gather(outputs['generative_loss'].detach()).mean()
                con_loss_gathered = self.accelerator.gather(outputs['contrastive_loss'].detach()).mean()

                epoch_losses['total_loss'] += total_loss_gathered.item()
                epoch_losses['generative_loss'] += gen_loss_gathered.item()
                epoch_losses['contrastive_loss'] += con_loss_gathered.item()

                running_losses['total_loss'] += total_loss_gathered.item()
                running_losses['generative_loss'] += gen_loss_gathered.item()
                running_losses['contrastive_loss'] += con_loss_gathered.item()

                if self.accelerator.sync_gradients:
                    self.global_step += 1

                if self.accelerator.is_main_process:
                    current_step = batch_idx + 1
                    progress_bar.set_postfix({
                        'Loss': f"{running_losses['total_loss']/current_step:.4f}",
                        'Gen': f"{running_losses['generative_loss']/current_step:.4f}",
                        'Con': f"{running_losses['contrastive_loss']/current_step:.4f}",
                        'LR': f"{self.get_current_lr():.2e}"
                    })

            except RuntimeError as e:
                raise e

        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        val_losses = {
            'total_loss': 0.0,
            'generative_loss': 0.0,
            'contrastive_loss': 0.0,
        }

        all_ground_truth = []
        all_generated = []
        generation_count = 0

        num_batches = len(self.val_dataloader)

        if num_batches == 0:
            self.accelerator.print("Warning: Validation dataloader is empty!")
            return val_losses

        if self.accelerator.is_main_process:
            progress_bar = tqdm(self.val_dataloader, desc="Validating",
                                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        else:
            progress_bar = self.val_dataloader

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(progress_bar):
                try:
                    if batch_data is None:
                        self.accelerator.print(f"Warning: None batch at index {batch_idx}, skipping...")
                        continue

                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        images, texts = batch_data
                    else:
                        self.accelerator.print(f"Warning: Invalid batch format at index {batch_idx}, skipping...")
                        continue

                    outputs = self.model(images, texts, is_train=False)

                    total_loss = (outputs['generative_loss'] +
                                  self.contrastive_weight * outputs['contrastive_loss'])

                    self.accelerator.wait_for_everyone()

                    total_loss_gathered = self.accelerator.gather(total_loss.detach()).mean()
                    gen_loss_gathered = self.accelerator.gather(outputs['generative_loss'].detach()).mean()
                    con_loss_gathered = self.accelerator.gather(outputs['contrastive_loss'].detach()).mean()

                    val_losses['total_loss'] += total_loss_gathered.item()
                    val_losses['generative_loss'] += gen_loss_gathered.item()
                    val_losses['contrastive_loss'] += con_loss_gathered.item()

                    if generation_count < self.max_generation_samples:
                        unwrapped_model = self.accelerator.unwrap_model(self.model)
                        generated_reports = unwrapped_model.generate_report(images)

                        if self.accelerator.is_main_process:
                            all_ground_truth.extend(texts)
                            all_generated.extend(generated_reports)
                            generation_count += len(texts)

                    self.accelerator.wait_for_everyone()

                    if self.accelerator.is_main_process:
                        current_avg_loss = val_losses['total_loss'] / (batch_idx + 1)
                        postfix = {
                            'Loss': f"{current_avg_loss:.4f}",
                            'Gen': f"{val_losses['generative_loss'] / (batch_idx + 1):.4f}",
                            'Con': f"{val_losses['contrastive_loss'] / (batch_idx + 1):.4f}",
                        }
                        if generation_count > 0:
                            postfix['Samples'] = f"{generation_count}/{self.max_generation_samples}"
                        progress_bar.set_postfix(postfix)

                except RuntimeError as e:
                    raise e

        self.accelerator.wait_for_everyone()

        if num_batches > 0:
            for key in val_losses:
                val_losses[key] /= num_batches

        if self.accelerator.is_main_process:
            if len(all_generated) > 0:
                try:
                    gen_metrics = calculate_metrics(all_ground_truth, all_generated)
                    val_losses.update({
                        'bleu_1': gen_metrics.get('bleu_1', 0.0),
                        'bleu_2': gen_metrics.get('bleu_2', 0.0),
                        'bleu_3': gen_metrics.get('bleu_3', 0.0),
                        'bleu_4': gen_metrics.get('bleu_4', 0.0),
                        'cider': gen_metrics.get('cider', 0.0),
                        'rougeL': gen_metrics.get('rougeL', 0.0),
                    })

                    combined_score = 0.5 * val_losses['bleu_2'] + 0.5 * val_losses['rougeL']
                    val_losses['combined_score'] = combined_score

                    print(f"\nGeneration Metrics (on {len(all_generated)} samples):")
                    print(f"  BLEU-1: {val_losses['bleu_1']:.4f}")
                    print(f"  BLEU-2: {val_losses['bleu_2']:.4f}")
                    print(f"  BLEU-3: {val_losses['bleu_3']:.4f}")
                    print(f"  BLEU-4: {val_losses['bleu_4']:.4f}")
                    print(f"  CIDEr: {val_losses['cider']:.4f}")
                    print(f"  ROUGE-L: {val_losses['rougeL']:.4f}")
                    print(f"  Combined Score (0.5×BLEU-2 + 0.5×ROUGE-L): {combined_score:.4f}")
                except Exception as e:
                    print(f"Warning: Failed to calculate generation metrics: {e}")
                    # Set default values on failure
                    val_losses.update({
                        'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0,
                        'cider': 0.0, 'rougeL': 0.0, 'combined_score': 0.0
                    })
            else:
                # No samples generated, use zeros
                val_losses.update({
                    'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0,
                    'cider': 0.0, 'rougeL': 0.0, 'combined_score': 0.0
                })
        else:
            # Non-main processes: initialize with zeros (will be broadcasted from main)
            val_losses.update({
                'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0,
                'cider': 0.0, 'rougeL': 0.0, 'combined_score': 0.0
            })

        # Broadcast metrics from main process to all processes
        self.accelerator.wait_for_everyone()
        if self.accelerator.num_processes > 1:
            import torch.distributed as dist
            for key in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'cider', 'rougeL', 'combined_score']:
                metric_tensor = torch.tensor(val_losses[key], device=self.accelerator.device)
                dist.broadcast(metric_tensor, src=0)
                val_losses[key] = metric_tensor.item()

        self.accelerator.wait_for_everyone()

        return val_losses

    def save_checkpoint(self, is_best: bool = False, is_last: bool = False):
        if not self.accelerator.is_main_process:
            return

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_stats': self.training_stats,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config),
        }

        if is_best:
            self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, self.best_model_path)
            print(f"Best model saved to {self.best_model_path}")

        if is_last:
            self.last_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, self.last_model_path)
            print(f"Last model saved to {self.last_model_path}")

    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            if self.accelerator.is_main_process:
                print(f"Checkpoint not found: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device, weights_only=False)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        # strict=False tolerates architecture drift between an old checkpoint and current model code
        missing_keys, unexpected_keys = unwrapped_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing_keys:
            self.accelerator.print(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            self.accelerator.print(f"Unexpected keys in checkpoint (ignored): {unexpected_keys}")

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'accelerator_state' in checkpoint:
            self.accelerator.load_state_dict(checkpoint['accelerator_state'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_stats = checkpoint.get('training_stats', self.training_stats)

        self.accelerator.print(f"Checkpoint loaded from {checkpoint_path}")

    def check_early_stopping(self, val_metrics: Dict[str, float], is_best: bool = False) -> bool:
        # Skip early stopping check if we haven't reached minimum epochs
        if self.current_epoch < self.min_epochs:
            if self.accelerator.is_main_process:
                print(f"Skipping early stopping check (epoch {self.current_epoch + 1}/{self.min_epochs} minimum)")
            return False

        should_stop = False

        if self.accelerator.is_main_process:
            if is_best:
                self.patience_counter = 0
                if 'combined_score' in val_metrics:
                    print(f"New best Combined Score: {val_metrics['combined_score']:.4f}")
                else:
                    print(f"New best Val Loss: {val_metrics['total_loss']:.4f}")
                should_stop = False
            else:
                self.patience_counter += 1
                if 'combined_score' in val_metrics:
                    print(f"No improvement in Combined Score ({self.patience_counter}/{self.early_stopping_patience})")
                else:
                    print(f"No improvement in Val Loss ({self.patience_counter}/{self.early_stopping_patience})")

                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                    should_stop = True
                else:
                    should_stop = False

        self.accelerator.wait_for_everyone()
        if self.accelerator.num_processes > 1:
            import torch.distributed as dist
            stop_decision = torch.tensor(should_stop, dtype=torch.bool, device=self.accelerator.device)
            dist.broadcast(stop_decision, src=0)
            should_stop = stop_decision.item()

        return should_stop

    def train(self):
        if self.accelerator.is_main_process:
            print("Starting training...")
            print(f"Total epochs: {self.num_epochs}")
            print(f"Train batches: {len(self.train_dataloader)}")
            print(f"Val batches: {len(self.val_dataloader)}")
            print(f"Using {self.accelerator.num_processes} GPU(s)")
            print("Waiting for all processes to synchronize...")

        self.accelerator.wait_for_everyone()

        self.accelerator.print("All processes synchronized. Starting training loop...")

        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            train_losses = self.train_epoch()

            val_losses = self.validate()

            # Step scheduler with appropriate arguments based on type
            scheduler_name = self.config.training.scheduler.name.lower()
            if scheduler_name == "cosine_with_warmup":
                # Already stepped per-step in train_epoch(), skip here
                pass
            elif isinstance(self.scheduler, ReduceLROnPlateau):
                # ReduceLROnPlateau needs a metric (use combined_score or val_loss)
                metric_value = val_losses.get('combined_score', val_losses['total_loss'])
                self.scheduler.step(metric_value)
            else:
                # CosineAnnealingLR and CosineAnnealingWarmRestarts don't need arguments
                self.scheduler.step()

            if self.accelerator.is_main_process:
                self.training_stats['train_losses'].append(train_losses['total_loss'])
                self.training_stats['val_losses'].append(val_losses['total_loss'])
                self.training_stats['val_combined_score'].append(val_losses.get('combined_score', 0.0))
                self.training_stats['val_bleu1'].append(val_losses.get('bleu_1', 0.0))
                self.training_stats['val_bleu2'].append(val_losses.get('bleu_2', 0.0))
                self.training_stats['val_bleu3'].append(val_losses.get('bleu_3', 0.0))
                self.training_stats['val_bleu4'].append(val_losses.get('bleu_4', 0.0))
                self.training_stats['val_cider'].append(val_losses.get('cider', 0.0))
                self.training_stats['val_rougeL'].append(val_losses.get('rougeL', 0.0))
                self.training_stats['learning_rates'].append(self.get_current_lr())

                epoch_time = time.time() - start_time

                summary = (
                    f"Epoch {epoch+1}/{self.num_epochs} - "
                    f"Train Loss: {train_losses['total_loss']:.4f}, "
                    f"Val Loss: {val_losses['total_loss']:.4f}"
                )
                if 'bleu_1' in val_losses:
                    summary += f", BLEU-1: {val_losses['bleu_1']:.4f}"
                summary += f", Time: {epoch_time/3600:.2f}h"
                print(summary)

            if 'combined_score' in val_losses:
                is_best = val_losses['combined_score'] > self.best_val_score
            else:
                is_best = val_losses['total_loss'] < self.best_val_loss

            # Check early stopping BEFORE updating best scores
            if self.check_early_stopping(val_losses, is_best):
                self.early_stopped = True
                self.save_checkpoint(is_best=is_best, is_last=True)
                break

            # Update best scores AFTER early stopping check
            if is_best:
                if 'combined_score' in val_losses:
                    self.best_val_score = val_losses['combined_score']
                self.best_val_loss = val_losses['total_loss']

            self.save_checkpoint(is_best=is_best, is_last=True)

            start_time = time.time()

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            total_time = time.time() - start_time
            print(f"Training completed in {total_time/3600:.2f} hours")

            if self.early_stopped:
                print(f"Training stopped early at epoch {self.current_epoch+1}")

            stats_path = self.checkpoint_dir / "training_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(self.training_stats, f, indent=2)

        return self.training_stats

    def test(self):
        if self.test_dataloader is None:
            print("No test dataloader provided. Skipping test evaluation.")
            return

        print("Running test evaluation...")
        test_results = self.evaluate_test_set(test_batch_size=24, save_to_excel=True)
        print(f"Test evaluation completed. Results: {test_results}")
        return test_results

    def evaluate_test_set(self, test_batch_size: int = 2, save_to_excel: bool = True) -> Dict[str, float]:
        if self.test_dataloader is None:
            raise ValueError("Test dataloader is not provided")

        print("Starting test set evaluation...")
        self.model.eval()

        all_ground_truth_texts = []
        all_generated_reports = []
        all_file_paths = []  # Store file paths

        original_dataset = self.test_dataloader.dataset
        test_loader_small = DataLoader(
            original_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        total_batches = len(test_loader_small)
        print(f"Evaluating {len(original_dataset)} test samples in {total_batches} batches (batch_size={test_batch_size})")

        progress_bar = tqdm(test_loader_small, desc="Testing",
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        sample_idx = 0  # Track sample index for file path lookup

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(progress_bar):
                try:
                    if batch_data is None:
                        print(f"Warning: None batch at index {batch_idx}, skipping...")
                        continue

                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        images, texts = batch_data
                    else:
                        print(f"Warning: Invalid batch format at index {batch_idx}, skipping...")
                        continue

                    images = images.to(self.accelerator.device)
                    batch_size = len(texts)

                    progress_bar.set_postfix({
                        'Status': 'Generating...',
                        'Batch': f"{batch_idx+1}/{total_batches}",
                        'Samples': f"{len(all_generated_reports)}/{len(original_dataset)}"
                    })

                    # Use generate_report to preserve multi-view information
                    generated_reports = self.accelerator.unwrap_model(self.model).generate_report(images)

                    # Get file paths for this batch
                    for i in range(batch_size):
                        if sample_idx + i < len(original_dataset.df):
                            folder_path = original_dataset.df.iloc[sample_idx + i]['path']
                            all_file_paths.append(folder_path)
                        else:
                            all_file_paths.append("N/A")

                    all_ground_truth_texts.extend(texts)
                    all_generated_reports.extend(generated_reports)
                    sample_idx += batch_size

                    progress_bar.set_postfix({
                        'Status': 'Completed',
                        'Batch': f"{batch_idx+1}/{total_batches}",
                        'Samples': f"{len(all_generated_reports)}/{len(original_dataset)}"
                    })

                    if batch_idx % 5 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nGPU OOM during test batch {batch_idx}, trying to recover...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

        print(f"\nGeneration completed. Total samples: {len(all_generated_reports)}")

        test_metrics = calculate_metrics(all_ground_truth_texts, all_generated_reports)

        print("\n" + "="*60)
        print("TEST SET EVALUATION RESULTS")
        print("="*60)
        for metric, value in test_metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        print("="*60)

        if save_to_excel:
            self._save_results_to_excel(all_ground_truth_texts, all_generated_reports, all_file_paths, test_metrics)

        del all_ground_truth_texts, all_generated_reports, all_file_paths
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return test_metrics

    def _save_results_to_excel(self, ground_truth_texts: List[str], generated_reports: List[str],
                               file_paths: List[str], metrics: Dict[str, float]) -> None:
        # Create results dataframe with file paths
        results_df = pd.DataFrame({
            'File_Path': file_paths,
            'Ground_Truth': ground_truth_texts,
            'Generated_Report': generated_reports
        })

        # Save to both CSV and Excel
        csv_path = self.checkpoint_dir / 'test_results.csv'
        results_df.to_csv(csv_path, index=False)

        excel_path = self.checkpoint_dir / "test_results.xlsx"
        results_df.to_excel(excel_path, index=False, engine='openpyxl')

        print(f"\n✅ Results saved: {csv_path}")
        print(f"✅ Results saved: {excel_path}")

        # Save metrics
        if metrics:
            metrics_df = pd.DataFrame([metrics])
            metrics_path = self.checkpoint_dir / 'test_metrics.csv'
            metrics_df.to_csv(metrics_path, index=False)
            print(f"✅ Metrics saved: {metrics_path}")
