# datasets/face_forensic_sbi.py
#-*- coding: utf-8 -*-
import os
import sys
import random
import traceback

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as T
from box import Box

from .builder import DATASETS, PIPELINES, build_pipeline
from .master import MasterDataset
try:
    from .sbi.utils import get_transforms, reorder_landmark, sbi_hflip, gen_target
except ImportError:
    print("Warning: SBI utils not found. Using placeholders.")
    def get_transforms(): return T.Compose([T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    def reorder_landmark(lms): return lms
    def sbi_hflip(img, m1, lms, m2): return img, m1, lms, m2
    def gen_target(img, lms, margin): return None, None, img, np.zeros_like(img)

from package_utils.transform import get_affine_transform, get_center_scale, draw_landmarks
from package_utils.utils import vis_heatmap
from package_utils.image_utils import load_image, crop_by_margin


@DATASETS.register_module()
class SBIFaceForensic(MasterDataset):
    def __init__(self,
                 split, # Passed via default_args
                 **kwargs): # Config keys passed as kwargs
        """Initializes the SBIFaceForensic dataset."""
        self.split = split
        print(f"--- SBIFaceForensic.__init__ called for split: {split} ---")

        # --- 1. Reconstruct config object ---
        config_dict = kwargs
        config = Box(config_dict, default_box=True, default_box_attr=None)
        if 'DATA' not in config: config.DATA = {}
        if split.upper() not in config.DATA: config.DATA[split.upper()] = {}
        self._cfg = config
        print(f"--- SBIFaceForensic: Reconstructed _cfg ---")

        # --- 2. Set self.dataset AND self.compression BEFORE super() ---
        try:
            self.dataset = self._cfg.DATA[split.upper()].get('NAME')
            self.compression = self._cfg.get('COMPRESSION')
            if self.dataset is None: raise KeyError(f"NAME not found in config.DATA.{split.upper()}")
            if self.compression is None: raise KeyError("COMPRESSION not found in config")
            print(f"--- SBIFaceForensic: self.dataset = {self.dataset}, self.compression = {self.compression} ---") # DEBUG PRINT
        except (AttributeError, KeyError, TypeError) as e:
             raise ValueError(f"SBIFaceForensic __init__: Missing critical config (NAME or COMPRESSION). Error: {e}.")

        # --- 3. Call parent initializers ---
        print(f"--- SBIFaceForensic: Calling super().__init__ ---")
        try:
             # Pass reconstructed config positionally
             super(SBIFaceForensic, self).__init__(self._cfg, split=split, **kwargs)
        except Exception as e:
             print(f"ERROR during super().__init__ call: {e}")
             traceback.print_exc()
             raise e
        print(f"--- SBIFaceForensic: Returned from super().__init__ ---")


        # --- 4. Initialize SBIFaceForensic attributes ---
        self.rot = 0
        self.pixel_std = 200
        try: # Add try-except for safety accessing config
            self.target_w = self._cfg.IMAGE_SIZE[1]
            self.target_h = self._cfg.IMAGE_SIZE[0]
            self.aspect_ratio = self.target_w * 1.0 / self.target_h
            self.sigma = self._cfg.get('SIGMA', 2)
            self.heatmap_type = self._cfg.get('HEATMAP_TYPE', 'gaussian')
            self.debug = self._cfg.get('DEBUG', False)
            self.train = self.split == 'train'
            self.dynamic_fxray = self._cfg.get('DYNAMIC_FXRAY', False)
            self.sampler_active = self._cfg.get('SAMPLER_ACTIVE', False)
        except (AttributeError, KeyError, TypeError) as e:
            raise ValueError(f"Error accessing config attributes after super(): {e}")

        # --- 5. Load data ---
        print(f"--- SBIFaceForensic: Calling self._load_data for split: {split}... ---")
        try:
            self.image_paths_r, self.labels_r, self.mask_paths_r, self.ot_props_r = self._load_data(split)
            print(f"--- SBIFaceForensic: Loaded {len(self.image_paths_r)} samples. ---")
            if len(self.image_paths_r) == 0:
                 print(f"CRITICAL WARNING: Loaded 0 samples for split {split}. Evaluation will likely fail.")
        except Exception as e:
            print(f"CRITICAL: Error during self._load_data call: {e}")
            traceback.print_exc()
            self.image_paths_r, self.labels_r, self.mask_paths_r, self.ot_props_r = [], [], [], []

        # --- 6. Initialize Transforms ---
        print(f"--- SBIFaceForensic: Initializing transforms... ---")
        try:
            if self._cfg.TRANSFORM and self._cfg.TRANSFORM.geometry:
                self.geo_transform = build_pipeline(self._cfg.TRANSFORM.geometry, PIPELINES,
                    default_args={"additional_targets": {"image_f": "image", "mask_f": "mask"}})
            else: self.geo_transform = None
        except Exception as e: print(f"Warning: Failed to build geometry transform: {e}"); self.geo_transform = None
        try: self.transforms = get_transforms()
        except NameError: self.transforms = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        print(f"--- SBIFaceForensic: __init__ finished. ---")


    def _load_data(self, split, anno_file=None):
        """Loads data lists either from path or file."""
        image_paths, labels, mask_paths, ot_props = None, None, None, None
        try:
            from_file = self._cfg.DATA[split.upper()].get('FROM_FILE', False)
            if not from_file:
                print(f"--- SBIFaceForensic._load_data: Calling self._load_from_path (routed)... ---")
                # This call routes through MasterDataset based on self.dataset
                loaded_data = self._load_from_path(split) # MasterDataset routes this
                print(f"--- SBIFaceForensic._load_data: Received type {type(loaded_data)} from _load_from_path ---") # DEBUG PRINT
                if isinstance(loaded_data, tuple) and len(loaded_data) == 4:
                    image_paths, labels, mask_paths, ot_props = loaded_data
                elif loaded_data is NotImplemented:
                     # This case should ideally not happen if FF._load_from_path handles all relevant compressions
                     raise NotImplementedError(f"_load_from_path for dataset '{self.dataset}' returned NotImplemented.")
                else:
                     raise TypeError(f"_load_from_path for {self.dataset} returned unexpected type: {type(loaded_data)}")
            else:
                # File loading logic (ensure _load_from_file exists and works)
                print(f"--- SBIFaceForensic._load_data: Calling self._load_from_file... ---")
                anno_file_path = self._cfg.DATA[split.upper()].get('ANNO_FILE', anno_file)
                if not anno_file_path: raise ValueError(f"ANNO_FILE missing for {split.upper()} with FROM_FILE=True.")
                loaded_data = self._load_from_file(split, anno_file=anno_file_path)
                if isinstance(loaded_data, tuple) and len(loaded_data) == 4: image_paths, labels, mask_paths, ot_props = loaded_data
                else: raise TypeError(f"_load_from_file returned unexpected type: {type(loaded_data)}")

        except NotImplementedError as nie: # Catch explicit NotImplementedError
             print(f"ERROR in _load_data for split {split}: Method not implemented - {nie}")
             traceback.print_exc(); return [], [], [], []
        except Exception as e:
            print(f"ERROR in _load_data calling loader method for split {split}: {e}")
            traceback.print_exc(); return [], [], [], []

        # Post-loading checks
        image_paths = image_paths if image_paths is not None else []
        labels = labels if labels is not None else []
        mask_paths = mask_paths if mask_paths is not None else []
        ot_props = ot_props if ot_props is not None else []
        if not image_paths: print(f"Warning: _load_data resulted in 0 image paths for split {split}.")

        # Sampler Logic (Placeholder)
        if self.sampler_active: print('Warning: Sampler is active but logic needs verification.')

        return image_paths, labels, mask_paths, ot_props

    # --- Rest of the methods (__len__, _load_img, __getitem__, etc.) ---
    # --- Keep the simplified __getitem__ from previous version for testing ---
    # --- Remember to replace placeholders in final_transforms, select_encode_method ---
    # --- (The rest of the methods from the previous full rewrite are omitted here for brevity) ---
    # --- but should be included in your actual file.                       ---

    def __len__(self):
        return len(self.image_paths_r) if hasattr(self, 'image_paths_r') and self.image_paths_r else 0

    def _load_img(self, img_path):
        try:
            img = load_image(img_path)
            # if img is None: print(f"Warning: load_image returned None for path: {img_path}") # Reduce verbosity
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

    def final_transforms(self, img_array):
        if img_array.dtype == np.uint8: img_array = img_array.astype(np.float32) / 255.0
        elif img_array.max() > 1.0 and img_array.dtype != np.uint8: img_array = img_array.astype(np.float32) / 255.0
        try: return self.transforms(img_array) # Use initialized transforms
        except Exception as e:
             print(f"Error in final_transforms: {e}. Image shape: {img_array.shape}, dtype: {img_array.dtype}")
             return torch.zeros((3, self.target_h, self.target_w), dtype=torch.float32)

    def select_encode_method(self, version): # Placeholder
        def dummy_encoder(target):
            if target is None: return None, None
            if target.ndim == 3: target = target[..., 0] if target.shape[-1] == 1 else cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            h, w = target.shape[:2]; heatmap = np.zeros((1, h, w), dtype=np.float32)
            cst_c = self._cfg.MODEL.heads.get('cstency', 256)
            cstency_hm = np.transpose(np.repeat(target[..., np.newaxis], cst_c, axis=2), (2, 0, 1)).astype(np.float32)
            return heatmap, cstency_hm
        return dummy_encoder

    def __getitem__(self, idx): # Simplified for testing
        max_retries = 3
        for retry in range(max_retries):
            try:
                if not self.image_paths_r: raise IndexError("Dataset image_paths list is empty.")
                if idx >= len(self.image_paths_r): idx = idx % len(self.image_paths_r)
                img_path = self.image_paths_r[idx]
                if not self.labels_r or idx >= len(self.labels_r): raise IndexError(f"Label list invalid for index {idx}.")
                label = self.labels_r[idx]; vid_id = img_path.split('/')[-2]
                img = self._load_img(img_path)
                if img is None: raise ValueError(f"Failed loading: {img_path}")
                if self.split == 'test':
                    img = crop_by_margin(img, margin=[5, 5])
                    if img is None or img.size == 0: raise ValueError(f"Image empty after cropping: {img_path}")

                if self.train: # --- TRAINING PLACEHOLDER ---
                    c, s = get_center_scale(img.shape[:2], self.aspect_ratio, pixel_std=self.pixel_std)
                    trans = get_affine_transform(c, s, self.rot, self._cfg.IMAGE_SIZE, pixel_std=self.pixel_std)
                    input_img = cv2.warpAffine(img, trans, (self.target_w, self.target_h), flags=cv2.INTER_LINEAR)
                    patch_img_trans = self.final_transforms(input_img)
                    hm_h, hm_w = self._cfg.HEATMAP_SIZE; cst_c = self._cfg.MODEL.heads.get('cstency', 256)
                    dummy_heatmap = torch.zeros((1, hm_h, hm_w), dtype=torch.float32)
                    dummy_target = torch.zeros((1, hm_h, hm_w), dtype=torch.float32)
                    dummy_cstency = torch.zeros((cst_c, hm_h, hm_w), dtype=torch.float32)
                    return (patch_img_trans, dummy_heatmap, dummy_target, dummy_cstency, patch_img_trans, dummy_heatmap, dummy_target, dummy_cstency)
                else: # --- TEST/VAL ---
                    c, s = get_center_scale(img.shape[:2], self.aspect_ratio, pixel_std=self.pixel_std)
                    trans = get_affine_transform(c, s, self.rot, self._cfg.IMAGE_SIZE, pixel_std=self.pixel_std)
                    input_img = cv2.warpAffine(img, trans, (self.target_w, self.target_h), flags=cv2.INTER_LINEAR)
                    img_trans = self.final_transforms(input_img)
                    label_val = label.item() if isinstance(label, torch.Tensor) else float(label)
                    label_np = np.array([label_val], dtype=np.float32)
                    return img_trans, label_np, vid_id
            except Exception as e:
                current_img_path = self.image_paths_r[idx] if hasattr(self, 'image_paths_r') and idx < len(self.image_paths_r) else "Unknown"
                print(f"Error processing index {idx} (path: {current_img_path}) (retry {retry+1}/{max_retries}): {e}")
                # traceback.print_exc() # Uncomment for debugging specific sample errors
                if retry == max_retries - 1: print(f"FATAL: Max retries exceeded for index {idx}."); raise e
                else: new_idx = random.randint(0, len(self.image_paths_r) - 1) if self.image_paths_r else 0; idx = new_idx
        raise RuntimeError(f"__getitem__ failed after retries.") # Should not be reached

    # train_collate_fn, train_worker_init_fn (omitted for brevity, use previous corrected version)
    # ... (Paste the train_collate_fn and train_worker_init_fn from the previous version here) ...
    def train_collate_fn(self, batch):
        batch = [item for item in batch if item is not None and len(item) == 8] # Filter out None or invalid items
        if not batch: return {}
        batch_data = {}
        try:
            img_f, hm_f, target_f, cst_f, img_r, hm_r, target_r, cst_r = zip(*batch)
            img_f_tensor = torch.stack(img_f, dim=0); img_r_tensor = torch.stack(img_r, dim=0)
            hm_f_tensor = torch.stack([torch.as_tensor(t, dtype=torch.float32) for t in hm_f], dim=0)
            hm_r_tensor = torch.stack([torch.as_tensor(t, dtype=torch.float32) for t in hm_r], dim=0)
            target_f_tensor = torch.stack([torch.as_tensor(t, dtype=torch.float32) for t in target_f], dim=0)
            target_r_tensor = torch.stack([torch.as_tensor(t, dtype=torch.float32) for t in target_r], dim=0)
            cst_f_tensor = torch.stack([torch.as_tensor(t, dtype=torch.float32) for t in cst_f], dim=0)
            cst_r_tensor = torch.stack([torch.as_tensor(t, dtype=torch.float32) for t in cst_r], dim=0)
            img = torch.cat([img_r_tensor, img_f_tensor], 0); heatmap = torch.cat([hm_r_tensor, hm_f_tensor], 0)
            target = torch.cat([target_r_tensor, target_f_tensor], 0); cst = torch.cat([cst_r_tensor, cst_f_tensor], 0)
            num_real = img_r_tensor.size(0); num_fake = img_f_tensor.size(0)
            label = torch.tensor([[0.0]] * num_real + [[1.0]] * num_fake, dtype=torch.float32)
            b_size = label.size(0)
            if b_size > 0:
                idxes = torch.randperm(b_size)
                img, label, target, heatmap, cst = img[idxes], label[idxes], target[idxes], heatmap[idxes], cst[idxes]
            batch_data["img"] = img; batch_data["label"] = label; batch_data["target"] = target
            batch_data["heatmap"] = heatmap; batch_data["cstency_heatmap"] = cst
        except Exception as e: print(f"Error in train_collate_fn: {e}"); traceback.print_exc(); return {}
        return batch_data

    def train_worker_init_fn(self, worker_id):
        seed = torch.initial_seed() % (2**32)
        random.seed(seed + worker_id); np.random.seed(seed + worker_id)