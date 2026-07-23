import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, List
import warnings
import os
from torch.utils.checkpoint import checkpoint

warnings.filterwarnings("ignore", message=".*bitsandbytes.*")
warnings.filterwarnings("ignore", message=".*GLIBC.*")
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

class TextFeatureEnhancer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, src_key_padding_mask=None):
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        return x

class FlamingoPerceiverResampler(nn.Module):
    def __init__(self,
                 num_latents: int = 6,
                 dim: int = 1536,
                 depth: int = 6,
                 num_heads: int = 8,
                 dim_head: int = 64):
        super().__init__()

        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)

        # Stack of Flamingo-style layers
        self.layers = nn.ModuleList([
            FlamingoResamplerLayer(dim, num_heads, dim_head)
            for _ in range(depth)
        ])

    def forward(self, visual_features):
        batch_size = visual_features.size(0)
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        for layer in self.layers:
            if self.training:
                latents = checkpoint(layer, latents, visual_features, use_reentrant=False)
            else:
                latents = layer(latents, visual_features)

        return latents


class FlamingoResamplerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_head: int = 64):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        # FFN with 4x expansion
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, latents, visual_features):
        cross_out, _ = self.cross_attn(
            query=latents,
            key=visual_features,
            value=visual_features
        )
        latents = self.norm1(latents + cross_out)

        self_out, _ = self.self_attn(latents, latents, latents)
        latents = self.norm2(latents + self_out)

        ffn_out = self.ffn(latents)
        latents = self.norm3(latents + ffn_out)

        return latents


class DynamicSpineXrayProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.swin_model = AutoModel.from_pretrained(config.model.visual_encoder.swin_model_name)
        self.swin_feature_dim = self.swin_model.config.hidden_size

        self.llama_embed_dim = None
        self.visual_projector = None

        self.max_views = config.model.visual_encoder.max_views
        self.target_size = tuple(config.model.visual_encoder.target_size)
        self.num_tokens_per_view = getattr(config.model.visual_encoder, 'num_tokens_per_view', 16)

        self.resampler = None

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, channels, height, width = images.shape

        if num_views > self.max_views:
            images = images[:, :self.max_views]
            num_views = self.max_views

        with torch.amp.autocast('cuda', enabled=False):
            all_views = images.view(batch_size * num_views, channels, height, width)

            # Convert to RGB
            if channels == 1:
                views_rgb = all_views.repeat(1, 3, 1, 1)
            else:
                views_rgb = all_views

            # Resize if needed
            if (height, width) != self.target_size:
                views_resized = F.interpolate(views_rgb, size=self.target_size,
                                             mode='bilinear', align_corners=False)
            else:
                views_resized = views_rgb

            target_h, target_w = self.target_size
            views_batched = views_resized.view(batch_size, num_views, 3, target_h, target_w)

            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=views_batched.device).view(1, 1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=views_batched.device).view(1, 1, 3, 1, 1)
            views_normalized = (views_batched - mean) / std

        return views_normalized

    def extract_view_features_with_checkpointing(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, channels, h, w = images.shape

        chunk_size = min(num_views, max(1, 64 // batch_size))
        all_features = []

        for i in range(0, num_views, chunk_size):
            end_idx = min(i + chunk_size, num_views)
            chunk_images = images[:, i:end_idx].contiguous()
            chunk_flat = chunk_images.view(-1, channels, h, w)

            def swin_forward(x):
                outputs = self.swin_model(pixel_values=x)
                return outputs.last_hidden_state

            if self.training:
                chunk_features = checkpoint(swin_forward, chunk_flat, use_reentrant=False)
            else:
                chunk_features = swin_forward(chunk_flat)

            # Perceiver compression
            chunk_features = self.resampler(chunk_features)

            chunk_size_actual = end_idx - i
            chunk_features = chunk_features.view(batch_size, chunk_size_actual, self.num_tokens_per_view, -1)
            all_features.append(chunk_features)

        return torch.cat(all_features, dim=1)

    def attention_based_aggregation(self, visual_sequence: torch.Tensor, return_attention: bool = False):
        global_embedding = visual_sequence.mean(dim=1)  # [B, D]

        if return_attention:
            batch_size, seq_len = visual_sequence.shape[0], visual_sequence.shape[1]
            uniform_attention = torch.full((batch_size, 1, seq_len), 1.0/seq_len, device=visual_sequence.device)
            return global_embedding, uniform_attention

        return global_embedding

    def forward(self, images: torch.Tensor, llama_model, return_attention: bool = False, use_aggregation: bool = False):
        batch_size = images.size(0)
        num_views = images.size(1)
        device = images.device

        processed_images = self.preprocess_images(images)
        view_features = self.extract_view_features_with_checkpointing(processed_images)

        # Project to LLM space
        target_dtype = self.visual_projector.weight.dtype
        view_features = view_features.to(dtype=target_dtype)

        b, nv, nt, d = view_features.shape
        view_features_flat = view_features.view(b * nv * nt, d)
        view_features_projected = self.visual_projector(view_features_flat)
        view_features = view_features_projected.view(b, nv, nt, -1)

        # Add positional embeddings
        view_position_ids = torch.arange(num_views, device=device)
        view_pos_emb = self.view_positional_embedding(view_position_ids).unsqueeze(0).unsqueeze(2)

        token_position_ids = torch.arange(self.num_tokens_per_view, device=device)
        token_pos_emb = self.token_positional_embedding(token_position_ids).unsqueeze(0).unsqueeze(0)

        positional_embeddings = view_pos_emb + token_pos_emb

        visual_tokens = view_features + positional_embeddings.to(dtype=target_dtype)
        visual_tokens_flat = visual_tokens.view(batch_size, num_views * self.num_tokens_per_view, -1)

        if not use_aggregation:
            # LLM path: multi-token
            return visual_tokens_flat
        else:
            # Contrastive path: global embedding
            if return_attention:
                visual_global_emb, attention_weights = self.attention_based_aggregation(visual_tokens_flat, return_attention=True)
                return visual_global_emb, attention_weights, num_views
            else:
                visual_global_emb = self.attention_based_aggregation(visual_tokens_flat)
                return visual_global_emb


class StandardContrastiveLearning(nn.Module):
    """Standard InfoNCE contrastive loss."""
    def __init__(self, config):
        super().__init__()
        self.temperature = config.model.contrastive.temperature

    def forward(self, visual_emb, text_emb, text_inputs=None):
        batch_size = visual_emb.size(0)

        target_dtype = visual_emb.dtype
        device = visual_emb.device
        text_emb = text_emb.to(dtype=target_dtype, device=device)

        # Normalize
        visual_emb = F.normalize(visual_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        # Similarity
        logits = torch.mm(visual_emb, text_emb.T) / self.temperature
        logits = logits.to(dtype=target_dtype)

        # InfoNCE
        targets = torch.arange(batch_size, device=device)
        i2t_loss = F.cross_entropy(logits, targets)
        t2i_loss = F.cross_entropy(logits.T, targets)

        return (i2t_loss + t2i_loss) / 2


class SpineRGModel(nn.Module):
    """Main model: Vision encoder + LLM + Contrastive learning."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.device = None
        target_dtype = self._get_torch_dtype(config)
        print(f"Initializing SpineRGModel with dtype: {target_dtype}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.llm.model_name, padding_side="right")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # LLM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.model.llm.model_name,
                torch_dtype=target_dtype,
                use_cache=config.model.llm.use_cache
            )

        # LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.model.lora.r,
            lora_alpha=config.model.lora.lora_alpha,
            lora_dropout=config.model.lora.lora_dropout,
            target_modules=config.model.lora.target_modules
        )
        self.llm = get_peft_model(self.llm, lora_config)

        # Vision encoder
        self.visual_encoder = DynamicSpineXrayProcessor(config)

        llama_embed_dim = self.llm.config.hidden_size
        self.text_enhancer = TextFeatureEnhancer(
            dim=llama_embed_dim,
            num_heads=8,    # increase (e.g. 16) if LLM hidden dim is large; rule of thumb: dim/64 or dim/128
            num_layers=2,
            dropout=0.1
        )
        self.text_enhancer.to(dtype=target_dtype)  # match LLM dtype
        
        self.visual_encoder.llama_embed_dim = llama_embed_dim
        self.visual_encoder.visual_projector = nn.Linear(self.visual_encoder.swin_feature_dim, llama_embed_dim)

        # Positional embeddings
        self.visual_encoder.view_positional_embedding = nn.Embedding(self.visual_encoder.max_views, llama_embed_dim)
        self.visual_encoder.token_positional_embedding = nn.Embedding(self.visual_encoder.num_tokens_per_view, llama_embed_dim)

        # Flamingo Perceiver Resampler: Spatial compression (patches → tokens)
        # Follows Flamingo architecture: 6 layers of [Cross-Attn → Self-Attn → FFN]
        # Compresses 144 Swin patches → 6 visual tokens per view
        self.visual_encoder.resampler = FlamingoPerceiverResampler(
            num_latents=self.visual_encoder.num_tokens_per_view,  # 6 tokens
            dim=self.visual_encoder.swin_feature_dim,              # 1536 for Swin-Large
            depth=6,                                                # Flamingo standard
            num_heads=8,                                            # Flamingo standard
            dim_head=64                                             # Flamingo standard
        )

        # Note: View aggregation now uses mean pooling (in attention_based_aggregation)
        # This ensures uniform gradient distribution to all visual tokens for contrastive learning

        # Contrastive learning
        self.contrastive_learning = StandardContrastiveLearning(config)
        self.generation_config = config.generation

    def _get_torch_dtype(self, config):
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64
        }
        dtype_str = getattr(config.model.llm, 'torch_dtype', 'bfloat16')
        return dtype_map.get(dtype_str, torch.bfloat16)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to multi-token representation."""
        visual_tokens = self.visual_encoder(images, self.llm, use_aggregation=False)
        return visual_tokens

    def encode_text(self, text: List[str]) -> torch.Tensor:
        """Encode text to embeddings."""
        encoded = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        return self.llm.get_input_embeddings()(encoded['input_ids'])

    def extract_text_global_embedding(self, text_input_ids, text_attention_mask):
        """Extract global text embedding for contrastive learning."""
        batch_size = text_input_ids.size(0)
        device = text_input_ids.device

        # EOS handling
        eos_token_id = self.tokenizer.eos_token_id
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if len(eos_token_id) > 0 else 128001
        elif eos_token_id is None:
            eos_token_id = 128001
        eos_token_id = int(eos_token_id)

        if text_input_ids[:, -1].ne(eos_token_id).any():
            eos_tokens = torch.full((batch_size, 1), eos_token_id, device=device, dtype=torch.long)
            text_input_ids = torch.cat([text_input_ids, eos_tokens], dim=1)
            text_attention_mask = torch.cat([
                text_attention_mask,
                torch.ones(batch_size, 1, device=device, dtype=text_attention_mask.dtype)
            ], dim=1)
            
        text_embeds = self.llm.get_input_embeddings()(text_input_ids)
        padding_mask = ~text_attention_mask.bool()
    
        text_embeds = self.text_enhancer(text_embeds, src_key_padding_mask=padding_mask)
        # Attention-weighted pooling
        try:
            attention_weights = text_attention_mask.float()
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
            text_global_emb = (text_embeds * attention_weights.unsqueeze(-1)).sum(dim=1)
        except Exception as e:
            print(f"Input shape: {text_input_ids.shape}")
            raise e

        return text_global_emb

    def forward(self, images: torch.Tensor, ground_truth_texts: List[str], is_train: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with dual losses:
        - Generative loss: LLM generation with multi-token visual input
        - Contrastive loss: Visual-text alignment with global embeddings
        """
        batch_size = images.size(0)
        num_views = images.size(1)
        num_tokens_per_view = self.visual_encoder.num_tokens_per_view
        device = images.device

        # Extract visual features
        processed_images = self.visual_encoder.preprocess_images(images)
        view_features = self.visual_encoder.extract_view_features_with_checkpointing(processed_images)

        # Project to LLM space
        target_dtype = self.visual_encoder.visual_projector.weight.dtype
        view_features = view_features.to(dtype=target_dtype)

        b, nv, nt, d = view_features.shape
        view_features_flat = view_features.view(b * nv * nt, d)
        view_features_projected = self.visual_encoder.visual_projector(view_features_flat)
        view_features = view_features_projected.view(b, nv, nt, -1)

        # Add positional embeddings
        view_position_ids = torch.arange(num_views, device=device)
        view_pos_emb = self.visual_encoder.view_positional_embedding(view_position_ids).unsqueeze(0).unsqueeze(2)

        token_position_ids = torch.arange(num_tokens_per_view, device=device)
        token_pos_emb = self.visual_encoder.token_positional_embedding(token_position_ids).unsqueeze(0).unsqueeze(0)

        positional_embeddings = view_pos_emb + token_pos_emb

        visual_tokens = view_features + positional_embeddings.to(dtype=target_dtype)
        visual_tokens_flat = visual_tokens.view(batch_size, num_views * num_tokens_per_view, -1)

        # Global embedding for contrastive
        visual_global_emb = self.visual_encoder.attention_based_aggregation(visual_tokens_flat)

        contrastive_loss = torch.tensor(0.0, device=device, requires_grad=is_train)
        lm_loss = torch.tensor(0.0, device=device, requires_grad=is_train)

        if ground_truth_texts:
            # Add EOS token to text for learning when to stop
            eos_token = self.tokenizer.eos_token or "<|eot_id|>"
            texts_with_eos = [text + eos_token for text in ground_truth_texts]

            # Tokenize (no BOS, EOS is in text string)
            text_tokens = self.tokenizer(texts_with_eos, return_tensors="pt",
                                         padding=True, truncation=True, max_length=512,
                                         add_special_tokens=False)
            input_ids = text_tokens['input_ids'].to(device)
            attention_mask = text_tokens['attention_mask'].to(device)

            eos_token_id = self.tokenizer.eos_token_id
            if isinstance(eos_token_id, (list, tuple)):
                eos_token_id = eos_token_id[0] if eos_token_id else 128009
            eos_token_id = int(eos_token_id or 128009)

            # Contrastive loss
            text_tokens_special = self.tokenizer(ground_truth_texts, return_tensors="pt",
                                                 padding=True, truncation=True, max_length=512)
            ids_special = text_tokens_special['input_ids'].to(device)
            mask_special = text_tokens_special['attention_mask'].to(device)
            true_text_emb = self.extract_text_global_embedding(ids_special, mask_special)
            contrastive_loss = self.contrastive_learning(visual_global_emb, true_text_emb, ground_truth_texts)

            # Generative loss: [visual][prompt][text+EOS]
            prompt_text = self.generation_config.prompt_text
            prompt_ids = self.tokenizer(prompt_text, return_tensors="pt",
                                        add_special_tokens=False)['input_ids'].to(device)
            prompt_ids = prompt_ids.expand(batch_size, -1)
            prompt_embeds = self.llm.get_input_embeddings()(prompt_ids).to(dtype=visual_tokens_flat.dtype)
            text_embeds = self.llm.get_input_embeddings()(input_ids).to(dtype=visual_tokens_flat.dtype)
            inputs_embeds = torch.cat([visual_tokens_flat, prompt_embeds, text_embeds], dim=1)

            # Labels: mask visual, prompt, and padding (keep EOS for learning to stop)
            labels = input_ids.clone()
            for i in range(batch_size):
                eos_pos = (input_ids[i] == eos_token_id).nonzero()
                if len(eos_pos) > 0 and eos_pos[0].item() + 1 < labels.size(1):
                    labels[i, eos_pos[0].item() + 1:] = -100

            num_visual = num_views * num_tokens_per_view
            num_prompt = prompt_ids.size(1)
            visual_labels = torch.full((batch_size, num_visual), -100, device=device, dtype=labels.dtype)
            prompt_labels = torch.full((batch_size, num_prompt), -100, device=device, dtype=labels.dtype)
            full_labels = torch.cat([visual_labels, prompt_labels, labels], dim=1)

            # Attention mask
            visual_attn = torch.ones(batch_size, num_visual, device=device, dtype=attention_mask.dtype)
            prompt_attn = torch.ones(batch_size, num_prompt, device=device, dtype=attention_mask.dtype)
            full_attention_mask = torch.cat([visual_attn, prompt_attn, attention_mask], dim=1)

            lm_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=full_attention_mask,
                                  labels=full_labels, return_dict=True)
            lm_loss = lm_outputs.loss if lm_outputs.loss is not None else lm_loss

        return {
            'generative_loss': lm_loss,
            'contrastive_loss': contrastive_loss,
            'visual_embeddings': visual_global_emb,
            'visual_tokens': visual_tokens_flat,
        }

    def generate_report(self, images: torch.Tensor) -> List[str]:
        """Generate reports from images."""
        self.eval()
        batch_size = images.size(0)
        num_views = images.size(1)
        num_tokens_per_view = self.visual_encoder.num_tokens_per_view
        device = images.device

        with torch.no_grad():
            # Extract visual features
            processed_images = self.visual_encoder.preprocess_images(images)
            view_features = self.visual_encoder.extract_view_features_with_checkpointing(processed_images)

            # Project to LLM space
            target_dtype = self.visual_encoder.visual_projector.weight.dtype
            view_features = view_features.to(dtype=target_dtype)

            b, nv, nt, d = view_features.shape
            view_features_flat = view_features.view(b * nv * nt, d)
            view_features_projected = self.visual_encoder.visual_projector(view_features_flat)
            view_features = view_features_projected.view(b, nv, nt, -1)

            # Add positional embeddings
            view_position_ids = torch.arange(num_views, device=device)
            view_pos_emb = self.visual_encoder.view_positional_embedding(view_position_ids).unsqueeze(0).unsqueeze(2)

            token_position_ids = torch.arange(num_tokens_per_view, device=device)
            token_pos_emb = self.visual_encoder.token_positional_embedding(token_position_ids).unsqueeze(0).unsqueeze(0)

            positional_embeddings = view_pos_emb + token_pos_emb

            visual_tokens = view_features + positional_embeddings.to(dtype=target_dtype)
            visual_tokens = visual_tokens.view(batch_size, num_views * num_tokens_per_view, -1)

            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                # Prompt - NO special tokens to match training
                prompt_text = self.generation_config.prompt_text
                prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt",
                                               add_special_tokens=False).input_ids.to(device)
                prompt_tokens = prompt_tokens.expand(batch_size, -1)
                prompt_embeds = self.llm.get_input_embeddings()(prompt_tokens)

                inputs_embeds = torch.cat([visual_tokens, prompt_embeds], dim=1)
                attention_mask = torch.ones(inputs_embeds.size()[:2], device=device)

                # Token IDs
                eos_token_id = self.tokenizer.eos_token_id
                pad_token_id = self.tokenizer.pad_token_id

                if isinstance(eos_token_id, (list, tuple)):
                    eos_token_id = eos_token_id[0] if len(eos_token_id) > 0 else 128001
                elif eos_token_id is None:
                    eos_token_id = 128001
                eos_token_id = int(eos_token_id)

                if isinstance(pad_token_id, (list, tuple)):
                    pad_token_id = pad_token_id[0] if len(pad_token_id) > 0 else eos_token_id
                elif pad_token_id is None:
                    pad_token_id = eos_token_id
                pad_token_id = int(pad_token_id)

                try:
                    outputs = self.llm.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        max_new_tokens=getattr(self.generation_config, 'max_new_tokens', 64),
                        min_length=getattr(self.generation_config, 'min_length', 20),
                        num_beams=getattr(self.generation_config, 'num_beams', 1) if getattr(
                            self.generation_config, 'num_beams', 1) is not None else 1,
                        do_sample=getattr(self.generation_config, 'do_sample', True),
                        temperature=getattr(self.generation_config, 'temperature', 0.9) if getattr(
                            self.generation_config, 'do_sample', True) else 1.0,
                        top_p=getattr(self.generation_config, 'top_p', 1.0) if getattr(self.generation_config, 'do_sample', True) else 1.0,
                        repetition_penalty=getattr(self.generation_config, 'repetition_penalty', 1.1),
                        length_penalty=getattr(self.generation_config, 'length_penalty', 1.0),
                        no_repeat_ngram_size=getattr(self.generation_config, 'no_repeat_ngram_size', 3)
                    )
                except Exception as e:
                    print(f"Error in generation: {e}")
                    outputs = torch.full((batch_size, 10), pad_token_id, device=device, dtype=torch.long)

            # Decode
            generated_texts = []
            for output in outputs:
                try:
                    text = self.tokenizer.decode(output, skip_special_tokens=True)
                    if prompt_text in text:
                        text = text.replace(prompt_text, "").strip()
                    generated_texts.append(text if text else "No report generated.")
                except Exception as e:
                    generated_texts.append("No report generated.")

            return generated_texts
