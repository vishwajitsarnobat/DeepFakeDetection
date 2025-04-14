#-*- coding: utf-8 -*-
from .builder import DATASETS
from .celebDF_v1 import CDFV1
from .celebDF_v2 import CDFV2
from .ff import FF
from .dfdcp import DFDCP
from .dfdc import DFDC
from .dfd import DFD
from .dfw import DFW

# Ensure all parent dataset classes are correctly imported above

@DATASETS.register_module()
class MasterDataset(CDFV1, FF, DFDCP, CDFV2, DFDC, DFD, DFW):
    """
    Master dataset class routing calls to specific dataset loaders.
    Requires 'self.dataset' to be set, typically by the inheriting class's __init__.
    """
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs) # Calls parent __init__ methods

        # Ensure self.dataset is available after initialization
        if not hasattr(self, 'dataset'):
             split = kwargs.get('split', 'test')
             try:
                 # Attempt to infer from config if not set by inheriting class
                 self.dataset = cfg.DATA[split.upper()].NAME
             except (AttributeError, KeyError):
                 raise ValueError("MasterDataset requires 'self.dataset' to be set, "
                                  "e.g., from config DATASET.DATA.<SPLIT>.NAME.")

    def _load_from_path(self, split):
        """Routes call to the correct parent _load_from_path based on self.dataset."""
        if not hasattr(self, 'dataset'):
             raise AttributeError("Instance variable 'dataset' is not set.")

        if self.dataset == 'FF++':
            return FF._load_from_path(self, split=split)
        elif self.dataset == 'Celeb-DFv1':
            return CDFV1._load_from_path(self, split=split)
        elif self.dataset == 'DFDCP':
            return DFDCP._load_from_path(self, split=split)
        elif self.dataset == 'Celeb-DFv2':
            return CDFV2._load_from_path(self, split=split)
        elif self.dataset == 'DFDC':
            return DFDC._load_from_path(self, split=split)
        elif self.dataset == 'DFD':
            return DFD._load_from_path(self, split=split)
        elif self.dataset == 'DFW':
            return DFW._load_from_path(self, split=split)
        else:
            raise NotImplementedError(f'Dataset type "{self.dataset}" is not supported by MasterDataset._load_from_path.')