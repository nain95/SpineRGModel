import os
import pandas as pd
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import Tuple, List, Optional
import cv2

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    print("Warning: albumentations not available. Augmentation will be disabled.")
    ALBUMENTATIONS_AVAILABLE = False


class SpineXrayPreprocessing:
    """
    Essential preprocessing for all X-ray images (train/val/test).
    Applied before augmentation to ensure consistent image quality.
    """
    def __init__(self):
        if ALBUMENTATIONS_AVAILABLE:
            self.transform = A.Compose([
                # CLAHE: Essential for X-ray contrast enhancement
                # Improves visibility of subtle findings (fractures, lesions)
                A.CLAHE(
                    clip_limit=2.0,      # Standard for medical imaging
                    tile_grid_size=(8, 8),
                    always_apply=True    # Apply to ALL images
                ),
            ])
        else:
            self.transform = None

    def __call__(self, image):
        """Apply preprocessing to image"""
        if self.transform is not None:
            if len(image.shape) == 2:  # Grayscale
                transformed = self.transform(image=image)
                return transformed['image']
        return image


class SpineXrayAugmentation:
    """
    Data augmentation for training only.
    Designed specifically for spine X-ray Report Generation.

    Key principles for Report Generation:
    1. Preserve anatomical structure (NO elastic/optical distortion)
    2. Conservative intensity changes (avoid information loss)
    3. Focus on realistic variations (positioning, exposure)
    """
    def __init__(self, mode='train', augment_prob=0.5):
        self.mode = mode
        self.augment_prob = augment_prob
        self.transform = None

        if ALBUMENTATIONS_AVAILABLE and mode == 'train':
            self.transform = A.ReplayCompose([
                # 1. Geometric transforms - VERY CONSERVATIVE
                # Spine alignment is CRITICAL for diagnosis
                A.Rotate(
                    limit=3,  # ±3° - realistic patient positioning variation
                    p=0.3,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                ),

                # Shift and scale - simulate different positioning/magnification
                A.ShiftScaleRotate(
                    shift_limit=0.05,    # Reduced from 0.1 (less aggressive)
                    scale_limit=0.08,    # Reduced from 0.12 (preserve details)
                    rotate_limit=0,      # NO rotation (already in Rotate above)
                    p=0.4,               # Reduced from 0.5
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                ),

                # 2. Intensity augmentations - simulate exposure variations
                # Use OneOf to avoid cumulative intensity distortion
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.1, 0.1),  # Reduced from ±0.15
                        contrast_limit=(-0.15, 0.15),  # Reduced from ±0.2
                        p=1.0
                    ),
                    A.RandomGamma(
                        gamma_limit=(80, 120),
                        p=1.0
                    ),
                ], p=0.5),  # Only one intensity transform, 50% of the time

                # 3. Noise - very minimal for modern X-ray systems
                A.OneOf([
                    A.GaussNoise(var_limit=(1, 5), p=1.0),  # Reduced from (2,10)
                    A.MultiplicativeNoise(multiplier=(0.99, 1.01), p=1.0),
                ], p=0.15),  # Reduced from 0.25

                # 4. NO ElasticTransform - distorts anatomical structures
                # 5. NO OpticalDistortion - changes spatial relationships
                # 6. NO CoarseDropout - may hide important pathology

            ], p=self.augment_prob)
        else:
            # validation/test mode or albumentations not available
            self.transform = None

    def __call__(self, image):
        if self.transform is not None and self.mode == 'train':
            # Ensure image is in correct format for albumentations
            # Note: augment_prob is already applied via ReplayCompose(p=self.augment_prob)
            if len(image.shape) == 2:  # Grayscale
                transformed = self.transform(image=image)
                return transformed['image']
        return image


class SpineXrayDataset(Dataset):

    def __init__(self, csv_path: str, target_size: Tuple[int, int] = (384, 384),
                 image_indices: Optional[List[int]] = None,
                 mode: str = 'train', augment_prob: float = 0.8):
        self.df = pd.read_csv(csv_path)
        self.target_size = target_size
        self.image_indices = image_indices if image_indices is not None else [1, 2, 3, 4, 5, 6]
        self.mode = mode

        # Preprocessing applied to ALL images
        self.preprocessing = SpineXrayPreprocessing()

        # Augmentation only for training
        self.augmentation = SpineXrayAugmentation(mode=mode, augment_prob=augment_prob)

        print("SpineXrayDataset initialized:")
        print(f"  - Mode: {mode}")
        print(f"  - Target size: {target_size}")
        print(f"  - Image indices: {self.image_indices}")
        print(f"  - Preprocessing: CLAHE (always applied)")
        print(f"  - Augmentation prob: {augment_prob if mode == 'train' else 0}")

    def _load_xray_images(self, folder_path: str) -> np.ndarray:
        """
        Improved preprocessing pipeline:
        1. Load DICOM
        2. Apply windowing (DICOM metadata based)
        3. Resize to target size
        4. **CLAHE preprocessing (ALL images)**
        5. Augmentation (train only, SAME for all views)
        6. Normalize to [0, 1] range
        """
        images = []
        replay_params = None  # augmentation params from view 0, replayed identically on other views

        for idx, i in enumerate(self.image_indices):
            dcm_file = os.path.join(folder_path, f"{i:05d}.dcm")

            if not os.path.exists(dcm_file):
                raise FileNotFoundError(f"X-ray DICOM file not found: {dcm_file}")

            dicom_data = pydicom.dcmread(dcm_file)
            pixel_array = dicom_data.pixel_array.astype(np.float32)

            windowed_image = self._apply_windowing_from_xray_dicom(dicom_data, pixel_array)

            if windowed_image.shape != self.target_size:
                windowed_image = cv2.resize(windowed_image,
                                           (self.target_size[1], self.target_size[0]),
                                           interpolation=cv2.INTER_LINEAR)

            preprocessed_image = self.preprocessing(windowed_image)

            # Augmentation (train mode only, SAME for all views)
            if self.mode == 'train' and self.augmentation.transform is not None:
                if idx == 0:
                    transformed = self.augmentation.transform(image=preprocessed_image)
                    augmented_image = transformed['image']
                    replay_params = transformed['replay']
                else:
                    augmented_image = A.ReplayCompose.replay(
                        replay_params, 
                        image=preprocessed_image
                    )['image']
            else:
                augmented_image = preprocessed_image

            # Simple 0-1 normalization
            normalized_image = augmented_image.astype(np.float32) / 255.0

            images.append(normalized_image)

        return np.stack(images, axis=0)

    def _apply_windowing_from_xray_dicom(self, dicom_data, pixel_array: np.ndarray) -> np.ndarray:
        try:
            if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                window_center = dicom_data.WindowCenter
                window_width = dicom_data.WindowWidth

                if isinstance(window_center, (list, tuple)):
                    window_center = window_center[0]
                if isinstance(window_width, (list, tuple)):
                    window_width = window_width[0]

                window_center = float(window_center)
                window_width = float(window_width)
            else:
                window_center = (pixel_array.max() + pixel_array.min()) / 2
                window_width = (pixel_array.max() - pixel_array.min()) * 0.8
        except:
            window_center = (pixel_array.max() + pixel_array.min()) / 2
            window_width = (pixel_array.max() - pixel_array.min()) * 0.8

        lwin = window_center - window_width / 2
        rwin = window_center + window_width / 2

        windowed = np.clip(pixel_array, lwin, rwin)
        windowed = windowed - lwin

        if windowed.max() > windowed.min():
            windowed = windowed / (windowed.max() - windowed.min())

        windowed = (windowed * 255).astype(np.uint8)

        return windowed

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        row = self.df.iloc[idx]
        folder_path = row['path']
        caption = str(row['CR_reading'])

        try:
            images = self._load_xray_images(folder_path)
            images = np.expand_dims(images, axis=1)
            images_tensor = torch.from_numpy(images)
            return images_tensor, caption
        except (FileNotFoundError, Exception) as e:
            print(f"Warning: Error loading sample {idx} from {folder_path}: {e}")
            placeholder_images = np.zeros((len(self.image_indices), 1, self.target_size[0], self.target_size[1]), dtype=np.float32)
            placeholder_tensor = torch.from_numpy(placeholder_images)
            return placeholder_tensor, caption


def worker_init_fn(worker_id):
    """Initialize random seeds for each DataLoader worker for reproducibility"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(csv_path: str,
                      batch_size: int = 4,
                      shuffle: bool = True,
                      num_workers: int = 2,
                      target_size: Tuple[int, int] = (224, 224),
                      image_indices: Optional[List[int]] = None,
                      mode: str = 'train',
                      augment_prob: float = 0.8,
                      seed: int = 42) -> DataLoader:

    dataset = SpineXrayDataset(
        csv_path=csv_path,
        target_size=target_size,
        image_indices=image_indices,
        mode=mode,
        augment_prob=augment_prob,
    )

    # Create generator for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    print(f"   Directory: {os.path.abspath(OUTPUT_DIR)}")
