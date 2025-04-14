# -- coding: utf-8 --
import math
import os
import sys
import argparse
import time
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import random

import simplejson as json
from box import Box as edict
from glob import glob
import dlib
import cv2
import numpy as np
from imutils import face_utils
from tqdm import tqdm

# Assuming these modules exist in the project structure
try:
    from configs.get_config import load_config
    from transform import affine_transform, draw_landmarks
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure 'configs' and 'transform' modules are accessible.")
    sys.exit(1)

class LandmarkUtility(object):
    def __init__(self,
                 cfg,
                 load_imgs=False,
                 **kwargs):
        # super().__init__() # Removed super() call as LandmarkUtility doesn't inherit

        assert "DATASET" in cfg, "Dataset can not be None!"
        assert "ROOT" in cfg, "Image Directory need to be provided!"

        if not isinstance(cfg, edict):
            cfg = edict(cfg)

        self.load_imgs = load_imgs
        self.image_root_base = cfg.ROOT
        self.image_suffix = cfg.IMAGE_SUFFIX or 'jpg'
        self.dataset = cfg.DATASET
        self.split = cfg.SPLIT or 'train'
        self.data_type = cfg.DATA_TYPE or 'images'
        self.debug = cfg.DEBUG
        self.fake_types = cfg.FAKETYPE or []
        self.real_types = cfg.get('REALTYPE', ['Original'])
        self.compression = cfg.COMPRESSION
        self.n_landmarks = cfg.get('N_LANDMARKS', 81)

        self.image_root = os.path.join(self.image_root_base, 'processed', self.split, self.data_type)
        print(f"Expecting data within: {self.image_root}")

        if kwargs is not None:
            for k,v in kwargs.items():
                if v is None:
                    raise ValueError(f'{k}:{v} recieve a None value!')
                self.__setattr__(k, v)

    def __contain__(self, key):
        return hasattr(self, key)

    def _load_data(self):
        img_paths = []
        file_names = []
        labels = []

        print(f'Loading data from dataset --- {self.dataset}')
        if self.load_imgs:
            img_paths, file_names, labels = self._load_data_from_path(self.image_root)
        else:
            assert self.__contain__('file_path'), "Loading data from file need a file path"
            img_paths, file_names, labels = self._load_data_from_file(self.__getattribute__('file_path'))

        assert len(img_paths) != 0, "Image paths have not been loaded! Please check image directory!"
        assert len(file_names) != 0, "Image files have not been loaded! Please check image suffixes!"
        assert len(labels) == len(img_paths), "Labels and image paths mismatch!"
        return img_paths, file_names, labels

    def _load_data_from_path(self, data_root):
        assert os.path.exists(data_root), f"Data root path does not exist: {data_root}!"
        fake_types = self.fake_types
        real_types = self.real_types
        img_paths = []
        labels = [] # 0 for real, 1 for fake

        print(f"Loading REAL types: {real_types}")
        for rt in real_types:
            data_dir = os.path.join(data_root, rt)
            if not os.path.exists(data_dir):
                 print(f"Warning: Real data directory not found: {data_dir}")
                 continue
            print(f"Scanning real directory: {data_dir}")
            for sub_dir in os.listdir(data_dir):
                sub_dir_path = os.path.join(data_dir, sub_dir)
                if os.path.isdir(sub_dir_path):
                    img_paths_ = glob(f'{sub_dir_path}/*.{self.image_suffix}')
                    img_paths.extend(img_paths_)
                    labels.extend([0] * len(img_paths_))

        print(f"Loading FAKE types: {fake_types}")
        for ft in fake_types:
            data_dir = os.path.join(data_root, ft)
            if not os.path.exists(data_dir):
                print(f"Warning: Fake data directory not found: {data_dir}")
                continue
            print(f"Scanning fake directory: {data_dir}")
            for sub_dir in os.listdir(data_dir):
                 sub_dir_path = os.path.join(data_dir, sub_dir)
                 if os.path.isdir(sub_dir_path):
                    img_paths_ = glob(f'{sub_dir_path}/*.{self.image_suffix}')
                    img_paths.extend(img_paths_)
                    labels.extend([1] * len(img_paths_))

        print('{} image paths loaded ({real_count} real, {fake_count} fake)!'.format(
            len(img_paths), real_count=labels.count(0), fake_count=labels.count(1)))

        relative_img_paths = []
        prefix_to_remove = self.image_root_base + ('/' if not self.image_root_base.endswith('/') else '')
        for ip in img_paths:
            # Use os.path.relpath for robust relative path calculation
            try:
                rel_path = os.path.relpath(ip, self.image_root_base)
                relative_img_paths.append(rel_path)
            except ValueError: # Handle cases like different drives on Windows
                 print(f"Warning: Path {ip} could not be made relative to {self.image_root_base}. Using absolute path.")
                 relative_img_paths.append(ip)

        file_names = [os.path.basename(ip) for ip in img_paths]

        return relative_img_paths, file_names, labels

    def _load_data_from_file(self, file_path):
        filename, file_extension = os.path.splitext(file_path)
        img_paths, file_names, labels = [], [], []
        if file_extension == '.json':
            print(f"Loading data from JSON: {file_path}")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                obj_data = data["data"]

                for item in obj_data:
                    img_paths.append(item['image_path'])
                    file_names.append(item['file_name'])
                    if 'label' in item:
                         labels.append(item['label'])
                    else:
                         path_parts = item['image_path'].split('/')
                         is_fake = any(ft in path_parts for ft in self.fake_types)
                         labels.append(1 if is_fake else 0)
                         print(f"Warning: 'label' key missing for {item['file_name']}, inferring label.")
            except FileNotFoundError:
                print(f"Error: Annotation file not found at {file_path}"); sys.exit(1)
            except KeyError as e:
                print(f"Error: Missing key {e} in JSON file {file_path}"); sys.exit(1)
            except Exception as e:
                print(f"Error loading JSON file {file_path}: {e}"); sys.exit(1)
        else:
             print(f"Error: Unsupported file extension {file_extension}"); sys.exit(1)

        print(f"{len(img_paths)} entries loaded from {file_path}")
        return img_paths, file_names, labels

    def _img_obj(self, img_path, file_name, label, **kwargs):
        path_parts = img_path.split(os.sep)
        fake_type_str = 'unknown'
        # Use reversed order check in case path contains both (e.g. /real/processed/fake/...)
        if any(ft in path_parts for ft in self.fake_types):
            # Find the specific fake type
            for ft in self.fake_types:
                 if ft in path_parts:
                     fake_type_str = ft
                     break
        elif any(rt in path_parts for rt in self.real_types):
             fake_type_str = self.real_types[0] if self.real_types else 'real'
        else: # Infer based on label if path doesn't contain type markers
             fake_type_str = (self.fake_types[0] if self.fake_types else 'fake') if label == 1 else (self.real_types[0] if self.real_types else 'real')


        obj = dict(image_path=img_path, file_name=file_name, label=label, fake_type=fake_type_str, **kwargs)
        return obj

    def _load_image(self, relative_img_path):
        full_img_path = os.path.join(self.image_root_base, relative_img_path)
        image = cv2.imread(full_img_path)
        if image is None: print(f"Warning: Failed to load image at {full_img_path}")
        return image

    def _facial_landmark(self, image, detector, lm_predictor):
        if image is None: return None
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except cv2.error:
             if len(image.shape) == 2: gray = image
             elif len(image.shape) == 3 and image.shape[2] == 1: gray = image[:, :, 0]
             else: print("Cannot convert image to grayscale."); return None
        except Exception as e: print(f"Unexpected error during color conversion: {e}."); return None

        try:
            f_rect = detector(gray, 1)
            if len(f_rect) > 0:
                face_rect = f_rect[0]
                if face_rect.left() >= face_rect.right() or face_rect.top() >= face_rect.bottom():
                    print(f"Warning: Invalid face rectangle: {face_rect}."); return None
                f_lms = lm_predictor(gray, face_rect)
                f_lms = face_utils.shape_to_np(f_lms)
                return f_lms
            else: return None
        except Exception as e: print(f"Error during dlib detection: {e}"); return None

    def _align_face(self, image, f_lms):
        if image is None or f_lms is None: return image, f_lms, None

        assert f_lms is not None, "Facial Landmarks can not be None!"
        required_lm_count = 48 # Example minimum for eye corners in 81pt model
        if f_lms.shape[0] <= required_lm_count:
            print(f"Warning: Not enough landmarks ({f_lms.shape[0]}) for alignment."); return image, f_lms, None

        le_idx, re_idx = 39, 48 # Indices for 81-point model outer corners
        if f_lms.shape[0] == 68: le_idx, re_idx = 36, 45

        try:
            le_x, le_y = f_lms[le_idx]
            re_x, re_y = f_lms[re_idx]
        except IndexError:
            print(f"Warning: Landmark indices out of bounds. Skipping alignment."); return image, f_lms, None

        delta_y = le_y - re_y; delta_x = le_x - re_x
        angle = math.atan2(delta_y, delta_x) * (180 / math.pi)

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0) # Correct angle sign might depend on coordinate system
        rot_img = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        rot_f_lms = np.empty_like(f_lms)
        for i, p in enumerate(f_lms):
             # Ensure affine_transform handles points correctly
             point_hom = np.array([p[0], p[1], 1.0])
             transformed_point = rot_mat @ point_hom
             rot_f_lms[i] = transformed_point[:2]
             # If using external affine_transform:
             # rot_f_lms[i] = affine_transform(p, rot_mat)


        return rot_img, f_lms, rot_f_lms

    def facial_landmarks(self, img_paths, detector, lm_predictor):
        rot_imgs, f_lmses, rot_f_lmses = [], [], []
        processed_indices = []
        skipped_lm_count, skipped_align_count, load_fail_count = 0, 0, 0

        for i, ip in enumerate(tqdm(img_paths, dynamic_ncols=True)):
            image = self._load_image(ip)
            if image is None: load_fail_count += 1; continue

            f_lms = self._facial_landmark(image, detector, lm_predictor)

            if f_lms is not None:
                rot_img, _f_lms, rot_f_lms = self._align_face(image, f_lms)
                if rot_f_lms is None:
                    skipped_align_count += 1
                    rot_img, _f_lms, rot_f_lms = image, f_lms, []
            else:
                skipped_lm_count += 1
                rot_img, _f_lms, rot_f_lms = image, [], []

            rot_imgs.append(rot_img)
            f_lmses.append(_f_lms)
            rot_f_lmses.append(rot_f_lms)
            processed_indices.append(i)

            if i < 10 and self.debug:
                 os.makedirs('samples', exist_ok=True)
                 if isinstance(rot_f_lms, np.ndarray) and len(rot_f_lms) > 0:
                     vis_img = draw_landmarks(rot_img.copy(), rot_f_lms)
                     cv2.imwrite(f'samples/test_{i}_aligned.jpg', vis_img)
                 elif isinstance(_f_lms, np.ndarray) and len(_f_lms) > 0 :
                      vis_img_orig = draw_landmarks(image.copy(), _f_lms)
                      cv2.imwrite(f'samples/test_{i}_original_lm.jpg', vis_img_orig)

        print(f"LM detect skips: {skipped_lm_count}, Align skips: {skipped_align_count}, Load fails: {load_fail_count}")
        return rot_imgs, f_lmses, rot_f_lmses, processed_indices

    def build_data(self, img_paths, file_names, labels, **kwargs):
        data = dict(data=[])
        orig_lmses = kwargs.get('orig_lmses', None)
        aligned_lmses = kwargs.get('aligned_lmses', None)

        if orig_lmses is not None: assert len(orig_lmses) == len(img_paths)
        if aligned_lmses is not None: assert len(aligned_lmses) == len(img_paths)

        for i, (p, f, l) in enumerate(zip(img_paths, file_names, labels)):
            img_obj = self._img_obj(p, f, label=l, id=i)
            if orig_lmses:
                lms = orig_lmses[i]
                img_obj['orig_lms'] = lms.tolist() if isinstance(lms, np.ndarray) else lms
            if aligned_lmses:
                lms = aligned_lmses[i]
                img_obj['aligned_lms'] = lms.tolist() if isinstance(lms, np.ndarray) else lms
            data["data"].append(img_obj)
        return data

    def save2json(self, data, fn=None):
        assert len(data['data']), "Data can not be empty!"
        target = os.path.join("processed_data", self.compression)
        os.makedirs(target, exist_ok=True)

        if fn is None:
            fake_types_str = "_".join(self.fake_types) if self.fake_types else "NoFake"
            real_types_str = "_".join(self.real_types) if self.real_types else "NoReal"
            if len(fake_types_str) > 30: fake_types_str = f"{len(self.fake_types)}fakes"
            if len(real_types_str) > 30: real_types_str = f"{len(self.real_types)}reals"
            fn = f"{self.split}_{self.dataset}_{real_types_str}_{fake_types_str}_lm{self.n_landmarks}.json"

        fp = os.path.join(target, fn)
        print(f"Saving processed data to: {fp}")
        try:
            with open(fp, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving JSON to {fp}: {e}"); raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Landmarks preprocessing!')
    parser.add_argument('--config', required=True, help='Config file')
    parser.add_argument('--file_path', help='File path to load processed data')
    parser.add_argument('--extract_landmark', action='store_true', help='Extract landmarks')
    parser.add_argument('--save_aligned', action='store_true', help='Save aligned images')
    parser.add_argument('--output_json', help='Specify output JSON filename')
    args = parser.parse_args()
    print(args)

    cfg_path = args.config
    cfg = load_config(cfg_path)
    preprocess_cfg = cfg.get('PREPROCESSING', cfg) # Use PREPROCESSING section or top-level

    extract_landmark = args.extract_landmark
    save_aligned = args.save_aligned and extract_landmark

    kwargs = {}

    if args.file_path:
        print("Initializing LandmarkUtility to load from file...")
        lm_ins = LandmarkUtility(preprocess_cfg, load_imgs=False, file_path=args.file_path, **kwargs)
    else:
        print("Initializing LandmarkUtility to load from image directories...")
        lm_ins = LandmarkUtility(preprocess_cfg, load_imgs=True, **kwargs)

    img_paths, file_names, labels = lm_ins._load_data()
    print(f'{len(img_paths)} items loaded.')

    final_img_paths, final_file_names, final_labels = img_paths, file_names, labels
    f_lmses_to_save, rot_f_lmses_to_save = None, None

    if not args.file_path and extract_landmark:
        lm_pretrained_path = preprocess_cfg.get('facial_lm_pretrained')
        if not lm_pretrained_path or not os.path.exists(lm_pretrained_path):
             print(f"Error: Landmark predictor '{lm_pretrained_path}' not found."); sys.exit(1)

        print(f"Loading landmark predictor: {lm_pretrained_path}")
        try:
            f_detector = dlib.get_frontal_face_detector()
            f_lm_detector = dlib.shape_predictor(lm_pretrained_path)
        except Exception as e: print(f"Error initializing dlib: {e}"); sys.exit(1)

        print("Starting landmark detection and alignment...")
        rot_imgs, f_lmses, rot_f_lmses, processed_indices = lm_ins.facial_landmarks(img_paths, f_detector, f_lm_detector)

        final_img_paths = [img_paths[i] for i in processed_indices]
        final_file_names = [file_names[i] for i in processed_indices]
        final_labels = [labels[i] for i in processed_indices]
        f_lmses_to_save = f_lmses
        rot_f_lmses_to_save = rot_f_lmses
        print(f"Retained {len(final_img_paths)} entries after processing.")

        if save_aligned:
            print("Saving aligned images...")
            aligned_base_dir = os.path.join(lm_ins.image_root_base, 'processed', lm_ins.split, f'aligned_{lm_ins.data_type}_{lm_ins.n_landmarks}')
            os.makedirs(aligned_base_dir, exist_ok=True)
            print(f"Saving aligned images to base directory: {aligned_base_dir}")
            saved_count, failed_count = 0, 0

            for idx, original_index in enumerate(tqdm(processed_indices, dynamic_ncols=True)):
                rot_img = rot_imgs[idx]
                original_rel_path = img_paths[original_index]

                try:
                    sub_path = os.path.relpath(os.path.join(lm_ins.image_root_base, original_rel_path), os.path.join(lm_ins.image_root_base, 'processed', lm_ins.split, lm_ins.data_type))
                except ValueError:
                    sub_path = final_file_names[idx]

                output_dir = os.path.join(aligned_base_dir, os.path.dirname(sub_path))
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, final_file_names[idx])

                try:
                    success = cv2.imwrite(output_path, rot_img)
                    if success:
                        saved_count += 1
                        aligned_relative_path = os.path.relpath(output_path, lm_ins.image_root_base)
                        final_img_paths[idx] = aligned_relative_path
                    else:
                        failed_count += 1
                        print(f"Warning: Failed to save {output_path}")
                except Exception as e:
                    failed_count += 1; print(f"Error saving {output_path}: {e}")

            print(f"Finished saving aligned images: {saved_count} saved, {failed_count} failed.")
            if saved_count > 0 and failed_count == 0:
                 lm_ins.data_type = f'aligned_{lm_ins.data_type}_{lm_ins.n_landmarks}'
                 print(f"Updated data_type to '{lm_ins.data_type}'")

        print('Landmarks processed.')
    elif args.file_path: print("Skipping landmark extraction (loading from file).")
    else: print("Proceeding without landmark extraction.")

    print("Building data structure for JSON...")
    build_kwargs = {}
    if f_lmses_to_save is not None: build_kwargs['orig_lmses'] = f_lmses_to_save
    if rot_f_lmses_to_save is not None: build_kwargs['aligned_lmses'] = rot_f_lmses_to_save

    data = lm_ins.build_data(final_img_paths, final_file_names, final_labels, **build_kwargs)

    try:
        lm_ins.save2json(data, fn=args.output_json)
        print("Processed Data saved successfully!")
    except Exception as e: print(f"Failed to save JSON data: {e}")