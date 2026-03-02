# Machine Failure Prediction -- Deep Learning with PyTorch
<!-- mashrou3 lel tanabo2 be fashal el makana besti5dam deep learning w PyTorch -->

A binary classification project that predicts whether an industrial machine
will fail based on sensor readings (temperature, torque, speed, tool wear, etc.).
<!-- mashrou3 classification binary, beyet-naba2 law el makana hateb0z wala la2 men el sensor readings -->

Built entirely with **PyTorch** and follows the **Golden Rules** of neural network
training -- starting from a sanity check all the way to a regularized production model.
<!-- mabny koloh be PyTorch w biyetbe3 el Golden Rules lel NN training -->
<!-- men el sanity check le7ad el regularized model el gahiz lel production -->

---

## Why This Project Exists
<!-- leh el mashrou3 da mawgood -->

The original version used classical ML (Logistic Regression, Decision Trees, Random Forest).
This version replaces all of that with deep learning to practice:
<!-- el version el adima kanet biste5dem ML 3ady. el version di 8ayartha le deep learning 3ashan net-darrab 3ala: -->

- Building custom PyTorch `Dataset` and `DataLoader` pipelines
  <!-- bena2 PyTorch Dataset w DataLoader pipelines ma5soosa -->
- Designing neural architectures of increasing complexity
  <!-- tasmim network architectures btetkaber f el ta32eed -->
- Applying the systematic 4-step training methodology
  <!-- tatbi2 el Golden Rules el arba3 7atawat -->
- Writing reproducible, well-structured experiment code
  <!-- ketabt code monazam w reproducible -->

---

## The Golden Rules (Training Strategy)
<!-- el Golden Rules (istratejyet el training) -- el 2awa3ed el zahabi lel training -->

We follow a bottom-up approach, where each step validates the previous one:
<!-- bnemshi men ta7t le fo2, kol step btet2akad en elly 2ablaha sha8al -->

| Step | Goal | What We Do |
|------|------|------------|
| **1. Sanity Check** | Verify the pipeline | Train on **1 sample** with a SimpleNN. Loss must go to 0. |
| **2. Baseline** | Establish a reference | Train SimpleNN on **all data**. Get baseline metrics. |
| **3. Reduce Bias** | Fix underfitting | Switch to DeeperNN (more layers, more neurons). |
| **4. Reduce Variance** | Fix overfitting | Add Dropout, BatchNorm, Weight Decay (L2). |

<!-- step 1: sanity check -- darrab 3ala sample wa7da w et2akad el loss benzil le 0 -->
<!-- step 2: baseline -- darrab SimpleNN 3al data kolaha w 5od el metrics -->
<!-- step 3: reduce bias -- 8ayar le DeeperNN (layers w neurons akthar) -->
<!-- step 4: reduce variance -- deed Dropout w BatchNorm w Weight Decay -->

If Step 1 fails, we stop immediately -- no point training on the full dataset
with a broken pipeline.
<!-- law step 1 feshlet, benwaqaf 3ala tool -- mafesh fayda nekamel training 3ala pipeline bayza -->

---

## Project Structure
<!-- haykal el mashrou3 -->

```
DeepLearning_Task2/
├── main.py              # Entry point -- runs the full experiment
│                        # el entry point -- beyeshghal el experiment koloh
├── data_loader.py       # CSV loading + PyTorch Dataset/DataLoader
│                        # ta7mil el CSV + PyTorch Dataset w DataLoader
├── eda.py               # Exploratory plots (saved to outputs/eda/)
│                        # el plots el isti3lameyya (btet7efaz f outputs/eda/)
├── preprocessor.py      # Balancing, encoding, scaling, splitting
│                        # mowazna, encoding, scaling, ta2sim el data
├── model.py             # 3 NN architectures (Simple to Regularized)
│                        # 3 architectures lel neural networks
├── trainer.py           # Training loops, evaluation, loss curves
│                        # loops el training, el evaluation, w loss curves
├── requirements.txt     # All dependencies with pinned versions
│                        # kol el dependencies be versions mo7addada
├── README.md            # You're reading it -- enta bte2rah da delwa2ti
├── data/
│   └── train.csv        # Raw dataset (136K rows, 14 columns)
│                        # el data el kham (136K saff, 14 3amood)
└── outputs/             # Auto-created at runtime -- beyet3emel otomatik
    ├── eda/             # Feature distribution & correlation plots
    ├── training/        # Loss curves for each training stage
    ├── results_comparison.csv
    └── best_model.pth   # Saved weights of the best model
```

---

## Dataset
<!-- el dataset -->

- **Source**: Machine predictive maintenance dataset
  <!-- masdar el data: dataset lel seyyana el tanabo2eyya -->
- **Rows**: 136,429
- **Target**: `Machine failure` (0 = OK, 1 = Failure)
- **Imbalance**: 134,281 vs 2,148 (~98:2) -- handled via downsampling
  <!-- el imbalance: 134K vs 2K (98:2) -- 7aleinaha bel downsampling -->
- **Features**: Type (L/M/H), Air temperature, Process temperature,
  Rotational speed, Torque, Tool wear, plus failure type indicators

---

## How to Run
<!-- ezzay teshghal el mashrou3 -->

```bash
# 1. Install dependencies
# nasab el libraries el matlooba
pip install -r requirements.txt

# 2. Run the full experiment
# shaghal el experiment kamel
python main.py
```

That's it. Everything else (folders, plots, model saving) happens automatically.
<!-- keda khalas. kol 7aga tanya (el folders, el plots, 7efz el model) bte7sal lewa7daha. -->

---

## Reproducibility
<!-- el reproducibility -- ezzay el nata2eg tetkarrar -->

Every source of randomness is locked down with a fixed seed (`42`):
<!-- kol masdar lel 3ashwa2eyya mat2afal be seed sabit (42): -->

- `random.seed(42)` for Python's built-in RNG
- `np.random.seed(42)` for NumPy
- `torch.manual_seed(42)` for PyTorch
- `torch.cuda.manual_seed_all(42)` for multi-GPU
- `torch.backends.cudnn.deterministic = True` for CUDA ops
- Stratified train/val split with `random_state=42`
- Downsampling with `random_state=42`

**Same code + same seed = same numbers, every single time.**
<!-- nafs el code + nafs el seed = nafs el ar2am, kol mara. -->

---

## Models
<!-- el models -->

| Model | Layers | Parameters | Purpose |
|-------|--------|------------|---------|
| `SimpleNN` | 2 hidden (32, 16) | ~700 | Sanity check + baseline |
| `DeeperNN` | 4 hidden (128, 64, 32, 16) | ~12K | More capacity to reduce bias |
| `RegularizedNN` | 4 hidden + BN + Dropout | ~13K | Reduce variance / overfitting |

<!-- SimpleNN: network so8ayara lel sanity check w el baseline -->
<!-- DeeperNN: network a3ma2 3ashan ne2allel el bias -->
<!-- RegularizedNN: nafs el capacity bs ma3 BN w Dropout 3ashan ne2allel el overfitting -->

All models output a single raw logit. We use `BCEWithLogitsLoss` which
applies sigmoid internally -- numerically more stable than `sigmoid + BCELoss`.
<!-- kol el models betraga3 logit wa7ed. bnesta5dem BCEWithLogitsLoss
elly beye3mel sigmoid gowah -- a7san 3adadiyyan men sigmoid + BCELoss -->

---

## Dependencies
<!-- el libraries el matlooba -->

| Package | Version |
|---------|---------|
| PyTorch | >= 2.7.0 |
| pandas | >= 2.2.3 |
| numpy | >= 2.2.3 |
| scikit-learn | >= 1.6.2 |
| matplotlib | >= 3.10.1 |
| seaborn | >= 0.13.2 |

---

## Key Design Decisions
<!-- 2ararat el tasmim el mohimma -->

1. **Downsampling over SMOTE** -- Simpler, and keeps the focus on the NN side.
   <!-- downsampling a7san men SMOTE -- absat, w bey5ali el focus 3al NN -->
2. **StandardScaler fitted on train only** -- Prevents data leakage.
   <!-- el scaler yet-fit 3al train bass -- 3ashan nemna3 el data leakage -->
3. **BCEWithLogitsLoss** -- Combines sigmoid + BCE for numerical stability.
   <!-- BCEWithLogitsLoss -- bey-game3 sigmoid + BCE 3ashan yestaqrar 3adadiyyan -->
4. **No early stopping in this version** -- We want to observe the full
   training dynamics (overfitting, convergence) for educational purposes.
   <!-- mafe4 early stopping -- 3ayzeen neshof el training dynamics kamla lel ta3leem -->
5. **Plots saved to disk** -- No `plt.show()` calls that block execution.
   Everything goes to `outputs/` for easy review.
   <!-- el plots btet7efaz 3al disk -- mafe4 plt.show() 3ashan mayw2afsh el execution -->
