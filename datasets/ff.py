# datasets/ff.py
#-*- coding: utf-8 -*-
import os
import sys
import numpy as np
from glob import glob

# Ensure project root is discoverable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
     sys.path.append(project_root)

try:
    from .builder import DATASETS
    from .common import CommonDataset
except ImportError as e:
    print(f"ERROR importing builder/common in ff.py: {e}. Using dummy classes.")
    class CommonDataset:
        def __init__(self, cfg, **kwargs): self._cfg = cfg; self.split=kwargs.get('split')
    _module_dict_dummy = {}
    class DummyRegistry: # Fallback registry
         def __init__(self, name): self._name = name; self._module_dict = _module_dict_dummy
         def register_module(self, module=None, name=None, **kwargs):
              if module: self._module_dict[name or module.__name__] = module
              else: return lambda cls: self._module_dict.setdefault(name or cls.__name__, cls) or cls
         def __contains__(self, key): return key in self._module_dict
         def get(self, key): return self._module_dict.get(key)
    DATASETS = DummyRegistry('Dataset')

class FF(CommonDataset):
    """
    Dataset loader for FaceForensics++. Focuses on c0/c40 structure.
    Relies on `self.compression` being set by the inheriting class instance.
    """
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

    def _load_from_path(self, split):
        """
        Loads image paths and labels by scanning c0/c40 directory structure.
        Structure: ROOT/split/data_type/ft/video_id/*.png
        """
        print("--- ENTERING FF._load_from_path ---") # DEBUG PRINT
        # --- Ensure attributes ---
        if not hasattr(self, '_cfg'): raise AttributeError("FF instance missing '_cfg'.")
        if not hasattr(self, 'split'): raise AttributeError("FF instance missing 'split'.")
        if not hasattr(self, 'compression'): raise AttributeError("FF instance missing 'compression'.")

        # --- Get config values ---
        try:
            split_upper = self.split.upper()
            root_path = self._cfg.DATA[split_upper].ROOT
            data_type = self._cfg.DATA.get('TYPE', 'frames')
            # Use FAKETYPE from the specific split config
            fake_types = self._cfg.DATA[split_upper].FAKETYPE
            image_suffix = self._cfg.get('IMAGE_SUFFIX', 'png')
            self.dataset_name = self._cfg.DATA[split_upper].get('NAME', 'FF++') # For logging
        except (AttributeError, KeyError, TypeError) as e:
             raise ValueError(f"FF Loader: Missing required config keys (ROOT, FAKETYPE, etc.): {e}")

        if not os.path.exists(root_path):
            raise FileNotFoundError(f"FF Loader: Root path does not exist: {root_path}")

        img_paths, labels = [], []
        print(f"FF Loader: Scanning {self.compression} data for split '{split}' at '{root_path}/{split}/{data_type}'...")
        print(f"FF Loader: Looking for fake types: {fake_types}")

        # --- Logic SPECIFICALLY for c0/c40 ---
        if self.compression == 'c0' or self.compression == 'c40':
            for ft in fake_types: # e.g., ['Deepfakes', 'FaceSwap', 'Original']
                data_dir = os.path.join(root_path, self.split, data_type, ft)
                print(f"--- FF Loader: Checking directory: {data_dir}") # DEBUG PRINT
                if not os.path.exists(data_dir):
                    print(f"Warning: Directory not found, skipping: {data_dir}")
                    continue

                found_in_ft = False
                # Iterate through video ID subdirectories
                for sub_dir_name in os.listdir(data_dir):
                    sub_dir_path = os.path.join(data_dir, sub_dir_name)
                    if os.path.isdir(sub_dir_path):
                        # Use glob to find images
                        current_paths = glob(f'{sub_dir_path}/*.{image_suffix}')
                        if current_paths:
                            img_paths.extend(current_paths)
                            # Assign label: 0 if 'Original', 1 otherwise
                            label_value = 0 if ft == 'Original' else 1
                            labels.extend(np.full(len(current_paths), label_value))
                            found_in_ft = True
                            # print(f"--- FF Loader: Found {len(current_paths)} images in {sub_dir_path}") # DEBUG PRINT

                if not found_in_ft:
                    print(f"Warning: No images found for type '{ft}' in subdirs of {data_dir}")

        else:
            # If compression is neither c0 nor c40, return empty lists
            print(f"Warning: FF Loader only supports 'c0' or 'c40' compression in this version, found '{self.compression}'. Returning empty lists.")
            # Explicitly return empty lists to avoid NotImplemented issues downstream
            return [], [], [], []


        print(f'--- FF Loader: Found {len(img_paths)} total image paths. ---')
        # Return the 4 required items
        mask_paths = []
        ot_props = []
        print("--- EXITING FF._load_from_path ---") # DEBUG PRINT
        return img_paths, labels, mask_paths, ot_props

# --- Conditional Registration ---
if 'FF' not in DATASETS:
    DATASETS.register_module(module=FF)