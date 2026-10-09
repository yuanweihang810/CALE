"""
7-dim sigmoid baseline - fair multi-label comparison
Uses the same 10% subject sampling, 1 fold, ResNet-18 as CALE
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
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GroupKFold

CUSTOM_DIR = '/root/autodl-tmp/IEEE/data_aligned'
TEST_DIR   = '/root/autodl-tmp/IEEE/test_aligned/test'
LABEL_FILE = '/root/autodl-tmp/IEEE/label.txt'
OUTPUT_DIR = '/root/autodl-tmp/IEEE/baseline_ml_10pct_1fold'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 128
LR = 1e-5
EPOCHS = 30
PATIENCE = 10
SEED = 42
WD = 5e-3
NUM_WORKERS = 8

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

EMOTIONS = ['angry','disgust','fear','happy','sad','surprise','natural']
NUM_CLASSES = 7
LOW_B, HIGH_B, BLUR = 80, 120, 10

def classify_condition(p):
    try:
        with Image.open(p) as img:
            arr = np.array(img.convert('L'), dtype=np.float64)
        b = arr.mean(); v = cv2.Laplacian(arr, cv2.CV_64F).var()
        if v < BLUR: return 2
        if b < LOW_B: return 0
        if b > HIGH_B: return 1
        return 3
    except: return 3

def folder_to_emotion(n):
    m = {'angry':'angry','anger':'angry','disgust':'disgust','disgusted':'disgust',
         'fear':'fear','scared':'fear','happy':'happy','happiness':'happy',
         'sad':'sad','sadness':'sad','surprise':'surprise','surprised':'surprise',
         'natural':'natural','neutral':'natural'}
    return m.get(n.strip().lower())

def sid(p): return os.path.basename(p).split('_')[0]

transform_train = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
transform_test = transform_train

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split, transform, precomputed=None, subset_idx=None):
        self.transform = transform; self.samples = []
        if precomputed is not None: self.samples = precomputed
        elif split == 'Training': self._load_train(CUSTOM_DIR)
        else: self._load_test(TEST_DIR, LABEL_FILE)
        if subset_idx is not None: self.samples = [self.samples[i] for i in subset_idx]

    def _load_train(self, root):
        for folder in sorted(Path(root).iterdir()):
            if not folder.is_dir(): continue
            emo = folder_to_emotion(folder.name)
            if emo is None: continue
            idx = EMOTIONS.index(emo)
            label = np.zeros(NUM_CLASSES, dtype=np.float32); label[idx] = 1.0
            for f in sorted(folder.glob('*.*')):
                if f.suffix.lower() in ('.jpg','.jpeg','.png'):
                    self.samples.append((str(f), label, classify_condition(str(f))))
        print(f'Train: {len(self.samples)}')

    def _load_test(self, img_dir, lp):
        with open(lp) as f: lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 8: continue
            img_no = parts[0]
            labels = np.array(list(map(int, parts[1:8])), dtype=np.float32)
            for ext in ['.png','.jpg','.jpeg']:
                p = os.path.join(img_dir, f'{img_no}{ext}')
                if os.path.exists(p):
                    self.samples.append((p, labels, classify_condition(p)))
                    break
        self.samples.sort(key=lambda x: x[0])
        print(f'Test: {len(self.samples)}')

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        p, l, c = self.samples[i]
        img = Image.open(p).convert('RGB')
        return self.transform(img), torch.tensor(l), c

class BaseML(nn.Module):
    def __init__(self, nc=7):
        super().__init__()
        bb = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = bb
        in_f = bb.fc.in_features
        bb.fc = nn.Linear(in_f, nc)
    def forward(self, x): return self.backbone(x)

def train_epoch(model, loader, opt):
    model.train(); total = 0
    for x, y, _ in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward(); opt.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

def evaluate(model, loader, threshold=0.5, mode='thresh'):
    model.eval()
    P, T = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(DEVICE)
            out = model(x)
            p = torch.sigmoid(out)
            P.append(p.cpu().numpy()); T.append(y.numpy())
    P = np.concatenate(P); T = np.concatenate(T)
    if mode == 'thresh':
        preds = (P > threshold).astype(np.float32)
    else:
        idx = np.argmax(P, axis=1)
        preds = np.zeros_like(T); preds[np.arange(len(idx)), idx] = 1
    return {
        'macro_f1': float(f1_score(T, preds, average='macro', zero_division=0)),
        'uar': float(recall_score(T, preds, average='macro', zero_division=0)),
        'avg_labels': float(preds.sum(axis=1).mean()),
    }

def main():
    full = Dataset('Training', transform_train)
    all_s = full.samples
    sids = [sid(s[0]) for s in all_s]
    test_set = Dataset('Testing', transform_test)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    gkf = GroupKFold(n_splits=5)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(np.zeros(len(sids)), groups=sids)):
        if fold >= 1: break
        # 10% subject sampling
        tr_subj = list(set([sids[i] for i in tr_idx]))
        rng = np.random.RandomState(SEED)
        rng.shuffle(tr_subj)
        keep = set(tr_subj[:max(1, int(len(tr_subj) * 0.1))])
        tr_idx = [i for i in tr_idx if sids[i] in keep]

        tr_set = Dataset('Training', transform_train, precomputed=all_s, subset_idx=tr_idx)
        va_set = Dataset('Validation', transform_test, precomputed=all_s, subset_idx=va_idx)
        tr_loader = DataLoader(tr_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
        va_loader = DataLoader(va_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        model = BaseML().to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-7)

        best = -1; patience = 0
        for epoch in range(EPOCHS):
            loss = train_epoch(model, tr_loader, opt)
            vm = evaluate(model, va_loader, mode='thresh')
            print(f'Epoch {epoch+1}: loss={loss:.4f} | Val F1={vm["macro_f1"]:.4f} | AvgLabels={vm["avg_labels"]:.2f}')
            if vm['macro_f1'] > best:
                best = vm['macro_f1']
                torch.save(model.state_dict(), f'{OUTPUT_DIR}/best_model.pth')
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE: break
            sched.step()

        model.load_state_dict(torch.load(f'{OUTPUT_DIR}/best_model.pth'))
        tm_th = evaluate(model, test_loader, mode='thresh')
        tm_arg = evaluate(model, test_loader, mode='argmax')
        print(f'\n7-dim sigmoid baseline:')
        print(f'  Val (thresh) F1: {best:.4f}')
        print(f'  Test thresh F1: {tm_th["macro_f1"]:.4f}, AvgLabels: {tm_th["avg_labels"]:.2f}')
        print(f'  Test argmax F1: {tm_arg["macro_f1"]:.4f}')

        with open(f'{OUTPUT_DIR}/summary.json', 'w') as f:
            json.dump({'val_f1': best, 'test_thresh': tm_th, 'test_argmax': tm_arg}, f, indent=2)

if __name__ == '__main__':
    main()
