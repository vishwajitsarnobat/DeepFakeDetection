# scripts/test.py
#-*- coding: utf-8 -*-
import os
import sys
import time
import argparse
import math
import random
import traceback
import itertools
import shutil # Keep for potential future use, but remove copy call

# Ensure project root is in sys.path
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from box import Box
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, f1_score, accuracy_score, roc_auc_score, average_precision_score, recall_score

# Project specific imports
try:
    from package_utils.transform import final_transform, get_center_scale, get_affine_transform
    from models import MODELS, build_model, load_pretrained
    from package_utils.utils import vis_heatmap
    from package_utils.image_utils import load_image, crop_by_margin
    from losses.losses import _sigmoid
    # Metrics calculated directly in this script now
    from datasets import DATASETS, build_dataset
    from logs.logger import Logger, LOG_DIR # Still attempt to import Logger
    from configs.get_config import load_config
except ImportError as e: print(f"CRITICAL: Import error: {e}. Check paths/deps."); sys.exit(1)
except Exception as e: print(f"CRITICAL: Import error: {e}"); sys.exit(1)


# --- Dummy Logger (Fallback) ---
class DummyLogger:
    # Use only methods that are guaranteed to exist or print directly
    def info(self, msg): print(f"INFO: {msg}")
    # warning and error will be handled by standard print below

# --- Argument Parser ---
def parse_args(args=None):
    arg_parser = argparse.ArgumentParser(description='Model Testing Script')
    arg_parser.add_argument('--cfg', '-c', help='Config file path', required=True)
    arg_parser.add_argument('--image', '-i', type=str, default=None, help='Single image path')
    arg_parser.add_argument('--limit_percent', type=float, default=100.0, help='Percentage of test data (0-100)')
    parsed_args = arg_parser.parse_args(args)
    return parsed_args

# --- Confusion Matrix Plotting Function ---
def plot_confusion_matrix(cm, classes, save_path, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """Prints, plots, and saves the confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        if normalize:
            row_sums = cm.sum(axis=1)[:, np.newaxis]
            with np.errstate(divide='ignore', invalid='ignore'): cm_normalized = cm.astype('float') / row_sums
            cm_normalized[np.isnan(cm_normalized)] = 0.0; cm_plot = cm_normalized; fmt = '.2f'; im = ax.imshow(cm_plot, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
            print("Normalized confusion matrix:")
        else: print('Confusion matrix, without normalization:'); cm_plot = cm; fmt = 'd'; im = ax.imshow(cm_plot, interpolation='nearest', cmap=cmap)
        print(cm_plot); ax.set_title(title); fig.colorbar(im, ax=ax);
        tick_marks = np.arange(len(classes)); ax.set_xticks(tick_marks); ax.set_xticklabels(classes, rotation=45); ax.set_yticks(tick_marks); ax.set_yticklabels(classes);
        thresh = cm_plot.max() / 2.;
        for i, j in itertools.product(range(cm_plot.shape[0]), range(cm_plot.shape[1])):
            ax.text(j, i, format(cm_plot[i, j], fmt), horizontalalignment="center", verticalalignment="center", color="white" if cm_plot[i, j] > thresh else "black", fontsize=12)
        plt.tight_layout(); ax.set_ylabel('True label'); ax.set_xlabel('Predicted label');
        fig.savefig(save_path);
        print(f"Saved confusion matrix plot to: {save_path}")
    except Exception as plot_e: print(f"ERROR: Failed to plot/save CM to {save_path}: {plot_e}")
    finally: plt.close(fig); # Ensure figure is closed

# --- Main Execution Block ---
if __name__=='__main__':
    args = parse_args()

    # --- Initialize Logger (Safely) ---
    logger = DummyLogger() # Use dummy logger
    try:
        logger_instance = Logger(task='testing')
        # Only use the imported logger if it has the 'info' method
        if hasattr(logger_instance, 'info'):
             logger = logger_instance
             logger.info("Main Logger initialized.")
        else:
             print("WARNING: Main Logger class missing 'info' method. Using print for INFO.")
    except Exception as e:
        print(f"WARNING: Logger init failed: {e}. Using print for INFO.")

    # --- Load Configuration ---
    try:
        cfg = load_config(args.cfg); logger.info(f"Loaded config: {args.cfg}")
        config_name = os.path.splitext(os.path.basename(args.cfg))[0]
    except FileNotFoundError: print(f"ERROR: Config file not found: {args.cfg}"); sys.exit(1)
    except Exception as e: print(f"ERROR: Loading config {args.cfg}: {e}"); traceback.print_exc(); sys.exit(1)

    # --- Seed ---
    seed = cfg.get('SEED', random.randint(0, 2**31)); random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed: {seed}")

    # --- Task and Parameters ---
    try: task = cfg.TEST.get('subtask', 'eval'); flip_test = cfg.TEST.get('flip_test', False); logger.info(f"Task: {task}, Flip Test: {flip_test}"); assert task in ['eval', 'test_img']
    except Exception as e: print(f"ERROR: Invalid TEST config: {e}"); sys.exit(1)

    # --- Device Setup ---
    use_cuda = torch.cuda.is_available()
    device_ids, device_count, device = [], 0, torch.device("cpu")
    if use_cuda:
        gpu_ids_cfg = cfg.TEST.get('gpus', []);
        if not isinstance(gpu_ids_cfg, list): gpu_ids_cfg = [gpu_ids_cfg]
        num_available_gpus = torch.cuda.device_count(); device_ids = [i for i in gpu_ids_cfg if isinstance(i, int) and 0 <= i < num_available_gpus]
        if len(device_ids) != len(gpu_ids_cfg): print(f"WARNING: Using valid GPUs {device_ids} from {gpu_ids_cfg}")
        device_count = len(device_ids)
        if device_count > 0: logger.info(f"Using {device_count} GPU(s): {device_ids}"); device = torch.device(f"cuda:{device_ids[0]}")
        else: print("WARNING: CUDA available but no valid GPUs found/specified. Using CPU.")
    else: logger.info("CUDA not available. Using CPU.")

    # --- Model Build and Load ---
    try:
        model_type = cfg.MODEL.get('type', 'N/A'); logger.info(f"Building model: {model_type}")
        precision_str = cfg.get('PRECISION', 'float32'); precision = getattr(torch, precision_str, torch.float32); logger.info(f"Using precision: {precision}")
        model = build_model(cfg.MODEL, MODELS).to(dtype=precision)
        pretrained_path = cfg.TEST.get('pretrained', None)
        if pretrained_path and os.path.exists(pretrained_path): logger.info(f"Loading weights: {pretrained_path}"); model = load_pretrained(model, pretrained_path)
        elif pretrained_path: print(f"WARNING: Pretrained path not found: {pretrained_path}")
        else: print("WARNING: No pretrained model specified.")
        if use_cuda and device_count > 0:
            if device_count > 1: logger.info(f"Using DataParallel: {device_ids}"); model = nn.DataParallel(model, device_ids=device_ids).to(device)
            else: model = model.to(device)
        else: model = model.to(device)
        model.eval()
    except Exception as e: print(f"ERROR: Building/loading model: {e}"); traceback.print_exc(); sys.exit(1)

    # --- Transforms and Other Variables ---
    try:
        transforms = final_transform(cfg.DATASET); img_size = cfg.DATASET.get('IMAGE_SIZE', [256, 256])
        aspect_ratio = img_size[1] * 1.0 / img_size[0]; pixel_std = cfg.DATASET.get('PIXEL_STD', 200)
        metrics_base = cfg.get('METRICS_BASE', 'binary'); fixed_threshold = cfg.TEST.get('threshold', 0.5); rot = 0
    except Exception as e: print(f"ERROR: Setting up params: {e}"); traceback.print_exc(); sys.exit(1)


    # =========================== EXECUTE TASK ===============================

    if task == 'test_img':
        # --- Single Image Testing ---
        if args.image is None: print("ERROR: Image path required."); sys.exit(1)
        logger.info(f"Processing single image: {args.image}")
        try:
            img = load_image(args.image);
            if img is None: raise ValueError("load_image failed")
            img_h, img_w = img.shape[:2]; center, scale = get_center_scale((img_w, img_h), aspect_ratio, pixel_std=pixel_std)
            trans = get_affine_transform(center, scale, rot, img_size, pixel_std=pixel_std)
            input_cv = cv2.warpAffine(img, trans, (img_size[1], img_size[0]), flags=cv2.INTER_LINEAR)
            input_rgb = cv2.cvtColor(input_cv, cv2.COLOR_BGR2RGB)
            img_tensor = transforms(input_rgb).to(dtype=precision)
            img_batch = torch.unsqueeze(img_tensor, 0).to(device)
            with torch.no_grad():
                st = time.time(); outputs = model(img_batch)
                if isinstance(outputs, list): outputs = outputs[0]
                cls_outputs = outputs.get('cls') if isinstance(outputs, dict) else outputs
                if cls_outputs is None: raise ValueError("No 'cls' output from model.")
                if cfg.TEST.get('vis_hm', False) and isinstance(outputs, dict) and 'hm' in outputs:
                    try: vis_heatmap(input_cv, _sigmoid(outputs['hm']).cpu().numpy()[0], 'output_pred_heatmap.jpg')
                    except Exception as vis_e: print(f"WARNING: Heatmap viz failed: {vis_e}") # Use print
                pred_score = cls_outputs.detach().cpu().numpy()[0][-1]
                pred_label = 'Fake' if pred_score > fixed_threshold else 'Real'
                logger.info(f'Inference time: {time.time() - st:.4f}s')
                logger.info(f'Prediction: {pred_label} (Score: {pred_score:.4f})')
        except Exception as e: print(f"ERROR: Single image test: {e}"); traceback.print_exc(); sys.exit(1) # Use print


    elif task == 'eval':
        # --- Dataset Evaluation ---
        logger.info(f"Initiating evaluation: metric_base={metrics_base}")
        video_level = cfg.TEST.get('video_level', False); logger.info(f"Video level: {video_level}")

        # --- Dataset Subset Logic ---
        limit_percent = args.limit_percent
        if not (0 < limit_percent <= 100): print(f"WARNING: Invalid --limit_percent. Using 100%."); limit_percent = 100.0
        use_subset = limit_percent < 100.0

        try: # Setup Dataset and DataLoader
            st = time.time(); logger.info("Building full dataset...")
            full_test_dataset = build_dataset(cfg.DATASET, DATASETS, default_args=dict(split='test'))
            if not full_test_dataset or len(full_test_dataset) == 0: print("ERROR: Dataset empty/failed."); sys.exit(1) # Use print
            n_total = len(full_test_dataset); logger.info(f"Full dataset samples: {n_total}")
            if use_subset:
                n_subset = max(1, int(n_total * limit_percent / 100.0)); logger.info(f"Using subset: {n_subset} samples ({limit_percent:.1f}%)")
                all_indices = list(range(n_total)); random.shuffle(all_indices); subset_indices = all_indices[:n_subset]; test_dataset = Subset(full_test_dataset, subset_indices)
            else: logger.info("Using full dataset."); test_dataset = full_test_dataset
            logger.info(f"Final dataset size for eval: {len(test_dataset)}")
            default_bs = 16; test_bs = cfg.TEST.get('batch_size', default_bs); effective_bs = test_bs * max(1, device_count)
            logger.info(f"DataLoader batch size: {effective_bs} ({test_bs} per GPU)")
            num_workers = cfg.DATASET.get('NUM_WORKERS', 0); pin_memory = cfg.DATASET.get('PIN_MEMORY', False) and use_cuda
            logger.info(f"DataLoader workers: {num_workers}, Pin memory: {pin_memory}")
            test_dataloader = DataLoader(test_dataset, batch_size=effective_bs, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
            logger.info(f'Dataset setup time: {time.time() - st:.2f}s')
        except Exception as e: print(f"ERROR: Setting up dataset/loader: {e}"); traceback.print_exc(); sys.exit(1) # Use print

        # --- Evaluation Loop ---
        total_preds_list, total_labels_list = [], []; vid_preds_dict, vid_labels_dict = {}, {}
        test_dataloader_pbar = tqdm(test_dataloader, dynamic_ncols=True, desc="Evaluating")
        logger.info("Starting evaluation loop...")
        eval_start_time = time.time()
        try:
            with torch.no_grad():
                for b, batch_data in enumerate(test_dataloader_pbar):
                    st_batch = time.time()
                    try: inputs, labels, vid_ids = batch_data; labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
                    except (ValueError, TypeError) as e: print(f"WARNING: Skipping batch {b}: Error unpacking - {e}"); continue # Use print

                    try: # Inference block with OOM handling
                        inputs = inputs.to(device, dtype=precision, non_blocking=pin_memory); outputs = model(inputs); cls_outputs = None
                        if flip_test:
                            inputs_flipped = inputs.flip(dims=(3,)); outputs_1 = model(inputs_flipped)
                            if isinstance(outputs, list): outputs = outputs[0]
                            if isinstance(outputs_1, list): outputs_1 = outputs_1[0]
                            if isinstance(outputs, dict): cls_out=outputs.get('cls'); cls_out_1=outputs_1.get('cls'); cls_outputs = (cls_out + cls_out_1) / 2.0 if cls_out is not None and cls_out_1 is not None else cls_out or cls_out_1
                            else: cls_outputs = (outputs + outputs_1) / 2.0
                        else:
                            if isinstance(outputs, list): outputs = outputs[0]
                            cls_outputs = outputs.get('cls') if isinstance(outputs, dict) else outputs
                        if cls_outputs is None: raise ValueError("No 'cls' output.")
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e): 
                            print(f"WARNING: OOM Error batch {b}. Reduce batch size? Skipping.")
                            if use_cuda: torch.cuda.empty_cache(); continue # Use print
                        else: print(f"WARNING: RuntimeError batch {b}: {e}"); raise e # Use print
                    except Exception as inf_e: print(f"WARNING: Skipping batch {b}: Inference error - {inf_e}"); continue # Use print

                    preds_cpu = cls_outputs.detach().cpu().numpy()
                    if not video_level: total_preds_list.extend(preds_cpu); total_labels_list.extend(labels_np)
                    else:
                        for idx, vid_id in enumerate(vid_ids):
                            if idx < len(preds_cpu) and idx < len(labels_np):
                                if vid_id not in vid_preds_dict: vid_preds_dict[vid_id] = []; vid_labels_dict[vid_id] = labels_np[idx:idx+1]
                                vid_preds_dict[vid_id].append(preds_cpu[idx:idx+1])
                            else: print(f"WARNING: Index mismatch batch {b}.") # Use print
                    batch_time = time.time() - st_batch; test_dataloader_pbar.set_postfix(batch_time=f"{batch_time:.3f}s")

            # --- Final Metric Calculation & Saving ---
            logger.info("Calculating final metrics...")
            eval_loop_time = time.time() - eval_start_time; logger.info(f"Evaluation loop time: {eval_loop_time:.2f}s")
            # Aggregate results
            if video_level:
                final_preds_list, final_labels_list = [], []
                for vid_id in vid_preds_dict:
                    if vid_id in vid_labels_dict:
                        video_avg_pred = np.mean(np.concatenate(vid_preds_dict[vid_id], axis=0), axis=0, keepdims=True)
                        final_preds_list.append(video_avg_pred); final_labels_list.append(vid_labels_dict[vid_id])
                    else: print(f"WARNING: Label missing for vid {vid_id}, skipping.") # Use print
                if not final_preds_list: print("ERROR: No valid video predictions collected."); sys.exit(1) # Use print
                total_preds_np = np.concatenate(final_preds_list, axis=0); total_labels_np = np.concatenate(final_labels_list, axis=0)
            else: total_preds_np = np.array(total_preds_list); total_labels_np = np.array(total_labels_list)
            # Check results
            if total_preds_np.size == 0 or total_labels_np.size == 0: print("ERROR: No results collected."); sys.exit(1) # Use print
            logger.info(f"Final predictions shape: {total_preds_np.shape}, Labels shape: {total_labels_np.shape}")

            try: # Metric calculation, saving, plotting
                if total_labels_np.ndim == 1: total_labels_np = total_labels_np.reshape(-1, 1)
                # Apply sigmoid if predictions look like logits
                epsilon = 1e-6
                if np.any(total_preds_np < -epsilon) or np.any(total_preds_np > 1.0 + epsilon):
                     print("WARNING: Predictions appear to be logits. Applying sigmoid.") # Use print
                     binary_preds_scores = 1 / (1 + np.exp(-total_preds_np))
                else: binary_preds_scores = total_preds_np # Assume probabilities

                binary_preds_scores_flat = binary_preds_scores[:, -1] if binary_preds_scores.ndim > 1 and binary_preds_scores.shape[1] > 1 else binary_preds_scores.ravel()
                labels_flat_np = total_labels_np.ravel().astype(int)

                # Find Optimal Threshold (Maximizing F1)
                precisions, recalls, thresholds_pr = precision_recall_curve(labels_flat_np, binary_preds_scores_flat, pos_label=1)
                f1_scores = np.zeros_like(thresholds_pr)
                valid_pr_indices = (precisions[:-1] + recalls[:-1]) > 0
                f1_scores[valid_pr_indices] = (2 * precisions[:-1][valid_pr_indices] * recalls[:-1][valid_pr_indices]) / (precisions[:-1][valid_pr_indices] + recalls[:-1][valid_pr_indices])
                best_f1_idx = np.argmax(f1_scores); optimal_threshold = thresholds_pr[best_f1_idx]; max_f1 = f1_scores[best_f1_idx]
                logger.info(f"Optimal threshold (max F1): {optimal_threshold:.4f} (Max F1: {max_f1*100:.2f}%)")

                # Calculate Metrics using OPTIMAL threshold
                optimal_binary_preds = (binary_preds_scores_flat >= optimal_threshold).astype(int)
                acc_optimal = accuracy_score(labels_flat_np, optimal_binary_preds)
                auc_, ap_ = 0.0, 0.0
                if len(np.unique(labels_flat_np)) > 1:
                    auc_ = roc_auc_score(labels_flat_np, binary_preds_scores_flat)
                    ap_ = average_precision_score(labels_flat_np, binary_preds_scores_flat)
                ar_optimal = recall_score(labels_flat_np, optimal_binary_preds, pos_label=1, zero_division=0)
                f1_optimal = f1_score(labels_flat_np, optimal_binary_preds, pos_label=1, zero_division=0)

                # Log final results
                logger.info(f'------ FINAL METRICS (Optimal Threshold, Dataset: {config_name}, Limit: {limit_percent}%) ------')
                logger.info(f' Optimal Threshold (Max F1): {optimal_threshold:.4f}')
                logger.info(f' Accuracy (@Optimal Thr):   {acc_optimal*100:.2f}%')
                logger.info(f' AUC:                       {auc_*100:.2f}%')
                logger.info(f' Avg Precision (AP):        {ap_*100:.2f}%')
                logger.info(f' Recall (@Optimal Thr):     {ar_optimal*100:.2f}%')
                logger.info(f' F1 Score (@Optimal Thr):   {f1_optimal*100:.2f}%')
                logger.info(f'----------------------------------------------------------------')

                # --- Save predictions and labels ---
                results_dir = "test_results"; os.makedirs(results_dir, exist_ok=True)
                file_suffix = f"_lim{int(limit_percent)}" if use_subset else ""
                preds_filename = os.path.join(results_dir, f"{config_name}_preds{file_suffix}.npy")
                labels_filename = os.path.join(results_dir, f"{config_name}_labels{file_suffix}.npy")
                np.save(preds_filename, total_preds_np); np.save(labels_filename, total_labels_np)
                logger.info(f"Saved predictions to: {preds_filename}"); logger.info(f"Saved labels to: {labels_filename}")

                # --- Confusion Matrix using OPTIMAL threshold ---
                cm = confusion_matrix(labels_flat_np, optimal_binary_preds)
                logger.info(f'Confusion Matrix (@Optimal Thr={optimal_threshold:.2f}, Limit: {limit_percent}%): TN={cm[0, 0]} FP={cm[0, 1]} FN={cm[1, 0]} TP={cm[1, 1]}')
                class_names = ['Real', 'Fake']

                # Save CM plots ONLY in results directory
                cm_title = f'CM (Optimal Thr {optimal_threshold:.2f}, {config_name} {limit_percent}%)'
                cm_filename_res = os.path.join(results_dir, f"{config_name}_CM_optimal{file_suffix}.png")
                plot_confusion_matrix(cm, classes=class_names, save_path=cm_filename_res, title=cm_title) # Call plot function

                cm_norm_title = f'Norm CM (Optimal Thr {optimal_threshold:.2f}, {config_name} {limit_percent}%)'
                cm_norm_filename_res = os.path.join(results_dir, f"{config_name}_CM_optimal_norm{file_suffix}.png")
                plot_confusion_matrix(cm, classes=class_names, save_path=cm_norm_filename_res, normalize=True, title=cm_norm_title) # Call plot function again

                # --- REMOVED: Copy CM plots to root directory ---
                # root_cm_filename = f"{config_name}_CM_optimal{file_suffix}.png"
                # root_cm_norm_filename = f"{config_name}_CM_optimal_norm{file_suffix}.png"
                # try: shutil.copyfile(cm_filename_res, root_cm_filename); shutil.copyfile(cm_norm_filename_res, root_cm_norm_filename); logger.info(f"Copied CM plots to root dir.")
                # except Exception as copy_e: print(f"WARNING: Could not copy CM plots to root: {copy_e}") # Use print
                # --- END REMOVED ---

            except Exception as e: print(f"ERROR: Metric calculation/saving/plotting: {e}"); traceback.print_exc() # Use print

        except Exception as e: print(f"ERROR: Evaluation loop: {e}"); traceback.print_exc(); sys.exit(1) # Use print
        finally: logger.info('-----------------*** Evaluation Finished ***--------------------')

    logger.info("Script finished.")