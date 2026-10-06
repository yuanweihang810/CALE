# CALE: Condition-Aware Label Expansion for FER

Code and data for the paper "Condition-Aware Label Expansion for Facial Expression Recognition: An Empirical Study of When It Works and When It Fails".

## Data

- Training set: https://drive.google.com/file/d/1ZcVy0gQ59Sw704AxI2s99ZuSg5h4O38M/view?usp=sharing
- Test set: https://drive.google.com/file/d/1oQP91wG6OMKK5C9Cob-4iYVCpDuTXUQZ/view?usp=sharing
- RAF-DB: http://www.whdeng.cn/RAF/model1.html
- RAF-ML: http://www.whdeng.cn/RAF/model2.html

## Installation

    pip install -r requirements.txt

## Repository Layout

- train/ : main training scripts
- diagnostic/ : diagnostic experiments in 10 percent data regime
- eval/ : evaluation scripts
- paper/ : LaTeX source and figure generation code

## License

MIT License.

## Reproducing the Main Results

Main 5-fold single-label cross-validation (Table IV):
    python train/train_cale_singlelabel.py

Diagnostic experiments in 10% data regime (Tables X-XIII):
    python diagnostic/run_cale_v2_10pct_1fold.py
    python diagnostic/run_cale_v2_10pct_1fold_softlabel0.py
    python diagnostic/run_baseline_ml_10pct_1fold.py

Fair multi-label baseline:
    python diagnostic/run_baseline_ml_10pct_1fold.py

Figures 9-11:
    cd paper/code
    python fig9_architecture.py
    python fig10_fair_baseline.py
    python fig11_condition_sensitivity.py
