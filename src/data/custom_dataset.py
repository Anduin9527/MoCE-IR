import os
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, RandomCrop, ToPILImage
from utils.image_utils import random_augmentation, crop_img


class CustomAIODataset(Dataset):
    """
    Custom Dataset for All-In-One Image Restoration.
    Assumes structure:
    Root/
      ├── [DegradationType] (e.g., Blur, Rain, Haze...)
      │    ├── GT/ (Ground Truth)
      │    └── LQ/ (Low Quality)
    """

    def __init__(self, args):
        super(CustomAIODataset, self).__init__()
        self.args = args
        self.patch_size = args.patch_size
        self.root_dir = args.data_file_dir

        self.crop_transform = Compose(
            [
                ToPILImage(),
                RandomCrop(self.patch_size),
            ]
        )
        self.toTensor = ToTensor()

        # 1. Scan for degradation types
        self._scan_dataset()

    def _scan_dataset(self):
        """
        Scans the root directory for degradation folders.
        Each folder must contain 'GT' and 'LQ' subdirectories.
        """
        self.samples = []  # List of {"GT": path, "LQ": path, "de_type": int}
        self.degradation_types = []

        # Get all subdirectories in root
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Dataset root directory not found: {self.root_dir}")

        subdirs = sorted(
            [
                d
                for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            ]
        )

        print(f"Found potential degradation types: {subdirs}")

        valid_types = []
        for idx, de_name in enumerate(subdirs):
            de_path = os.path.join(self.root_dir, de_name)
            gt_path = os.path.join(de_path, "GT")
            lq_path = os.path.join(de_path, "LQ")

            # Check if GT and LQ exist
            if os.path.exists(gt_path) and os.path.exists(lq_path):
                valid_types.append(de_name)

                # Find images
                # Support common extensions
                exts = ["*.jpg", "*.png", "*.jpeg", "*.bmp"]
                lq_files = []
                for ext in exts:
                    lq_files.extend(glob.glob(os.path.join(lq_path, ext)))

                print(f"  -> Loading {de_name}: Found {len(lq_files)} LQ images.")

                for lq_f in lq_files:
                    filename = os.path.basename(lq_f)
                    gt_f = os.path.join(gt_path, filename)

                    if os.path.exists(gt_f):
                        self.samples.append(
                            {
                                "GT": gt_f,
                                "LQ": lq_f,
                                "de_type": idx,  # Map type to integer ID
                            }
                        )
                    else:
                        # Warning: Skipping unmatched pair
                        pass
            else:
                print(f"  -> Skipping {de_name}: Missing 'GT' or 'LQ' folder.")

        self.de_dict = {name: i for i, name in enumerate(valid_types)}
        self.de_dict_reverse = {i: name for i, name in enumerate(valid_types)}

        if not self.samples:
            raise ValueError("No valid image pairs found in the dataset directory!")

        print(
            f"Total training samples loaded: {len(self.samples)} across {len(valid_types)} tasks."
        )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        de_id = sample["de_type"]

        # Load images
        # Use convert('RGB') to ensure 3 channels
        lq_img = np.array(Image.open(sample["LQ"]).convert("RGB"))
        gt_img = np.array(Image.open(sample["GT"]).convert("RGB"))

        # Ensure sizes match (basic check) if needed, but crop handles it generally
        # unless one is smaller than patch size.

        # Random Crop & Augmentation
        # Use the utility from existing codebase if available, or implement standard one
        # existing dataset_utils uses random_augmentation(*self._crop_patch(lr, hr))

        lq_patch, gt_patch = self._crop_patch(lq_img, gt_img)
        lq_patch, gt_patch = random_augmentation(lq_patch, gt_patch)

        # To Tensor
        lq_tensor = self.toTensor(lq_patch)
        gt_tensor = self.toTensor(gt_patch)

        return [sample["LQ"], de_id], lq_tensor, gt_tensor

    def __len__(self):
        return len(self.samples)

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]

        # Pad if image is smaller than patch size
        if H < self.patch_size or W < self.patch_size:
            # Simple padding logic or error
            # For now let's assume images are larger, or pad
            pad_h = max(0, self.patch_size - H)
            pad_w = max(0, self.patch_size - W)
            if pad_h > 0 or pad_w > 0:
                img_1 = np.pad(img_1, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                img_2 = np.pad(img_2, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                H, W = img_1.shape[:2]

        ind_H = random.randint(0, H - self.patch_size)
        ind_W = random.randint(0, W - self.patch_size)

        patch_1 = img_1[
            ind_H : ind_H + self.patch_size, ind_W : ind_W + self.patch_size
        ]
        patch_2 = img_2[
            ind_H : ind_H + self.patch_size, ind_W : ind_W + self.patch_size
        ]

        return patch_1, patch_2


class CustomTestDataset(Dataset):
    """
    Custom Validation/Test Dataset.
    Returns full images (no cropping).
    """

    def __init__(self, args, split="test"):
        super(CustomTestDataset, self).__init__()
        self.args = args
        self.root_dir = args.data_file_dir
        self.toTensor = ToTensor()
        self._scan_dataset()

    def _scan_dataset(self):
        self.samples = []

        if not os.path.exists(self.root_dir):
            raise ValueError(f"Dataset root directory not found: {self.root_dir}")

        subdirs = sorted(
            [
                d
                for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            ]
        )

        for idx, de_name in enumerate(subdirs):
            # Check if user mentioned specific benchmarks, otherwise load all
            if (
                hasattr(self.args, "benchmarks")
                and self.args.benchmarks
                and de_name not in self.args.benchmarks
            ):
                continue

            de_path = os.path.join(self.root_dir, de_name)
            gt_path = os.path.join(de_path, "GT")
            lq_path = os.path.join(de_path, "LQ")

            if os.path.exists(lq_path):
                # If inference_only, we don't strictly need GT path to exist generally,
                # but let's assume structure is [Degradation]/LQ at least.
                # GT path might not exist.

                exts = ["*.jpg", "*.png", "*.jpeg", "*.bmp"]
                lq_files = []
                for ext in exts:
                    lq_files.extend(glob.glob(os.path.join(lq_path, ext)))

                for lq_f in lq_files:
                    filename = os.path.basename(lq_f)
                    gt_f = os.path.join(gt_path, filename)

                    if os.path.exists(gt_f):
                        self.samples.append({"GT": gt_f, "LQ": lq_f, "de_type": idx})
                    elif self.args.inference_only:
                        self.samples.append({"GT": None, "LQ": lq_f, "de_type": idx})
            elif self.args.inference_only:
                # Flattened support debug
                exts = ["*.jpg", "*.png", "*.jpeg", "*.bmp"]
                print(f"DEBUG: Checking flat structure in {de_path} with exts {exts}")
                lq_files = []
                for ext in exts:
                    glob_path = os.path.join(de_path, ext)
                    found = glob.glob(glob_path)
                    print(f"DEBUG: Globbing {glob_path} -> Found {len(found)}")
                    lq_files.extend(found)

                if lq_files:
                    print(
                        f"  -> Found {len(lq_files)} images in {de_name} (Flat structure). Using as LQ."
                    )
                    for lq_f in lq_files:
                        self.samples.append({"GT": None, "LQ": lq_f, "de_type": idx})
                else:
                    print(
                        f"  -> No images found in {de_path}. Checked extensions: {exts}"
                    )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        de_id = sample["de_type"]

        lq_img = np.array(Image.open(sample["LQ"]).convert("RGB"))
        lq_tensor = self.toTensor(lq_img)

        if sample["GT"]:
            gt_img = np.array(Image.open(sample["GT"]).convert("RGB"))
            gt_tensor = self.toTensor(gt_img)
        else:
            # For inference only, use lq as dummy gt to keep shape consistency
            gt_tensor = lq_tensor

        return [sample["LQ"], de_id], lq_tensor, gt_tensor

    def __len__(self):
        return len(self.samples)
