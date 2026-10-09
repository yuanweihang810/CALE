# run_cale_v2_10pct_1fold_efficientnetb0_softlabel0.py
"""
CALE V2 (LE only, SOFT_LABEL=0.0) - 10% subject sampling, 1 fold, 30 epochs
Diagnostic experiment: other condition positions set to 0 during training (only main branch supervised)
"""
import os, json, random, time
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, recall_score
from sklearn.model_selection import GroupKFold

# ==================== Configuration ====================
CUSTOM_DIR = '/root/autodl-tmp/IEEE/data_aligned'
TEST_DIR   = '/root/autodl-tmp/IEEE/test_aligned/test'
LABEL_FILE = '/root/autodl-tmp/IEEE/label.txt'

BACKBONE_NAME = 'efficientnetb0'
VARIANT = 'V2'
SOFT_LABEL      = 0.0          # <<< only change: 0.5 -> 0.0
INFERENCE_SCALE = 0.5
FIXED_THRESHOLD = 0.5

OUTPUT_DIR = '/root/autodl-tmp/IEEE/cale_v2_10
