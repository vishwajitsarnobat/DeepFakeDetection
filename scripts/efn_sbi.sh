#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py --cfg configs/efn4_fpn_sbi_adv.yaml
