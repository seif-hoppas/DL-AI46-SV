"""
main.py -- Machine Failure Prediction using Deep Learning (PyTorch)
===================================================================
This is the entry point.  It orchestrates the full experiment
following the Golden Rules of NN training:
# da el file el asasi. bey-nazem el tagr0ba kol-ha 3ala 7asab el Golden Rules

  Step 1 -- Sanity Check:   overfit 1 sample, verify pipeline works
  Step 2 -- Baseline:       train SimpleNN on all data
  Step 3 -- Reduce Bias:    train DeeperNN (more capacity)
  Step 4 -- Reduce Variance: train RegularizedNN + weight decay + dropout

Each step builds on the previous one.  If step 1 fails, we stop
immediately -- no point in training for hours on broken code.
# kol step mabniyya 3ala elly 2ablaha. law step 1 feshlet, benwaqaf
# mafesh fayda nekamel training 3ala code bay0z

Reproducibility:  We set ALL random seeds (Python, NumPy, PyTorch)
at the very start.  Same code + same seed = same numbers, every time.
# kol el seeds bet7ased fl awel. nafs el code + nafs el seed = nafs el nateega dayman

Usage:
    python main.py
"""

import os
import random
import numpy as np
import pandas as pd
import torch

# Our project modules
# el modules beta3tna el ma7alleyya
from data_loader  import load_csv, build_loaders
from eda          import plot_features, plot_target, plot_correlation
from preprocessor import downsample, encode_features, prepare_data
from model        import SimpleNN, DeeperNN, RegularizedNN, count_parameters
from trainer      import sanity_check, train_model, evaluate_model


# ===================================================================
#  Reproducibility -- lock down EVERY source of randomness
#  ne2fel kol masdar lel 3ashwa2eyya 3ashan el nata2eg tetkarrar
# ===================================================================
def set_seed(seed=42):
    """
    Seeds Python, NumPy, and PyTorch so results are identical
    across runs.  Also disables CuDNN non-deterministic ops.
    This is non-negotiable for scientific reproducibility.
    # bne7ot seeds le Python w NumPy w PyTorch 3ashan el nata2eg tefdal wa7da
    # kaman bne2fal el CuDNN operations el mosh deterministic
    # da lazem lazem lel reproducibility el 3elmeya
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # These two lines make CUDA ops deterministic (at a small speed cost)
    # el satrein dol biye5lello el CUDA operations deterministic (3ala 7esab shwayyet speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[main] Random seed set to {seed} for full reproducibility")


# ===================================================================
#  Device selection -- use GPU if available, else CPU
#  e5tar el GPU law mawgood, 8er keda CPU
# ===================================================================
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] Using device: {device}")
    return device


# ===================================================================
#  MAIN PIPELINE
#  el pipeline el ra2eseya beta3t el experiment
# ===================================================================
def main():

    # Create output directories
    # e3mel el folders elly hanesta3melha lel output
    os.makedirs("outputs", exist_ok=True)

    # ---------------------------------------------------------------
    #  0. REPRODUCIBILITY & DEVICE SETUP
    #  tazbit el seeds w e5teyar el device
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  MACHINE FAILURE PREDICTION -- DEEP LEARNING EXPERIMENT")
    print("=" * 60)

    SEED = 42
    set_seed(SEED)
    device = get_device()

    # ---------------------------------------------------------------
    #  1. LOAD DATA
    #  ta7mil el data men el CSV
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 0: LOADING RAW DATA")
    print("=" * 60)

    df = load_csv("data/train.csv")

    # A quick peek at the data to see what we're working with
    # naza basita 3al data 3ashan ne3raf bene3mel eih
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nColumn types:\n{df.dtypes}")
    print(f"\nBasic stats:\n{df.describe()}")

    # ---------------------------------------------------------------
    #  2. EXPLORATORY DATA ANALYSIS
    #  ta7lil isti3lami lel data (neshof el distributions w el imbalance)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 0: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    plot_target(df)           # See the class imbalance right away
                              # neshof el imbalance ben el classes 3ala tool
    plot_features(df)         # Feature distributions coloured by class
                              # tawzi3 el features mel2awn 7asab el class
    plot_correlation(df)      # Correlation heatmap for numerics
                              # 5aretat el irtibatat bein el features el ra2ameyya

    # ---------------------------------------------------------------
    #  3. PREPROCESSING
    #  ta7dir el data (balancing, encoding, scaling, splitting)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 0: PREPROCESSING")
    print("=" * 60)

    # Balance the dataset first (before encoding or splitting)
    # wazzen el data el awel (abl el encoding aw el splitting)
    df_balanced = downsample(df, random_state=SEED)

    # Encode categoricals and drop identifiers
    # 7awel el categorical columns le ar2am w esheel el ID columns
    df_encoded, label_encoders = encode_features(df_balanced)

    # Split into train/val and scale -- scaler fitted on train ONLY
    # e2sem le train w val w e3mel scaling -- el scaler yet-fit 3al train BASS
    X_train, X_val, y_train, y_val, scaler = prepare_data(
        df_encoded, test_size=0.2, random_state=SEED
    )

    input_dim = X_train.shape[1]   # Number of features going into the network
                                    # 3adad el features elly hade5al bihom el network
    print(f"\nInput dimension for NN: {input_dim}")

    # ---------------------------------------------------------------
    #  4. THE GOLDEN RULES -- Step 1: SANITY CHECK
    # ---------------------------------------------------------------
    #  "Train on a single sample and make sure loss goes to 0."
    #  This catches bugs in model architecture, loss function,
    #  data pipeline, and tensor shapes.  ALWAYS do this first.
    #  # "darrab 3ala sample wa7da w et2akad en el loss benzil le 0."
    #  # da beyakshed el bugs f el model, el loss function, el data pipeline,
    #  # w el tensor shapes. DAYMAN e3mel da el awel.
    # ---------------------------------------------------------------

    # Reset seed before each model to ensure fair comparison
    # reset el seed abl kol model 3ashan el m2arna tekoon 3adla
    set_seed(SEED)
    sanity_model = SimpleNN(input_dim)
    count_parameters(sanity_model)

    sanity_check(
        model    = sanity_model,
        X_sample = X_train[0],     # Just one sample!  -- sample wa7da bass!
        y_sample = y_train[0],
        device   = device,
        epochs   = 200,
        lr       = 1e-2,
    )

    # ---------------------------------------------------------------
    #  5. THE GOLDEN RULES -- Step 2: ESTABLISH A BASELINE
    # ---------------------------------------------------------------
    #  "Train a simple model on the full training data."
    #  This gives us a reference point.  Any fancier model MUST
    #  beat this, or we're making things worse for no reason.
    #  # "darrab model basit 3al data kolaha."
    #  # da beyedeena nu2tet marg3eyya. ay model a7san LAZEM
    #  # ye8leb da, we law la2 yeb2a e7na bnekhrab el mawdoo3.
    # ---------------------------------------------------------------

    set_seed(SEED)
    train_loader, val_loader = build_loaders(
        X_train, y_train, X_val, y_val, batch_size=64
    )

    baseline_model = SimpleNN(input_dim)
    count_parameters(baseline_model)

    baseline_model, baseline_history = train_model(
        model          = baseline_model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        device         = device,
        epochs         = 50,
        lr             = 1e-3,
        weight_decay   = 0.0,           # No regularization yet -- lessa men 8er regularization
        stage_name     = "Step 2: Baseline (SimpleNN)",
        filename_prefix= "step2_baseline",
    )

    # How does the baseline do?
    # el baseline 3amel eih?
    baseline_metrics = evaluate_model(
        baseline_model, val_loader, device,
        stage_name="Step 2: Baseline (SimpleNN)"
    )

    # ---------------------------------------------------------------
    #  6. THE GOLDEN RULES -- Step 3: REDUCE BIAS (fix underfitting)
    # ---------------------------------------------------------------
    #  "Use a more complex model to push training accuracy higher."
    #  If the baseline underfits (accuracy is mediocre on training
    #  data), we need more model capacity: deeper + wider layers.
    #  # "esta5dem model a3qad 3ashan el training accuracy ye3la."
    #  # law el baseline 3amel underfitting (el accuracy mesh kweisa),
    #  # me7tagen model akbar: layers akthar w awsa3.
    # ---------------------------------------------------------------

    set_seed(SEED)
    train_loader, val_loader = build_loaders(
        X_train, y_train, X_val, y_val, batch_size=64
    )

    deeper_model = DeeperNN(input_dim)
    count_parameters(deeper_model)

    deeper_model, deeper_history = train_model(
        model          = deeper_model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        device         = device,
        epochs         = 80,
        lr             = 1e-3,
        weight_decay   = 0.0,           # Still no regularization -- lessa men 8er regularization
        stage_name     = "Step 3: Reduce Bias (DeeperNN)",
        filename_prefix= "step3_reduce_bias",
    )

    deeper_metrics = evaluate_model(
        deeper_model, val_loader, device,
        stage_name="Step 3: Reduce Bias (DeeperNN)"
    )

    # ---------------------------------------------------------------
    #  7. THE GOLDEN RULES -- Step 4: REDUCE VARIANCE (fix overfitting)
    # ---------------------------------------------------------------
    #  "Add regularization to improve generalization."
    #  If the gap between train loss and val loss is large,
    #  the model is memorizing noise.  We fight this with:
    #    - Dropout (randomly kill neurons during training)
    #    - BatchNorm (stabilize layer inputs)
    #    - Weight Decay / L2 (penalize large weights)
    #  # "deed regularization 3ashan el model ye-generalize a7san."
    #  # law el far2 bein el train loss w el val loss kbeer,
    #  # el model biye7faz noise. bne7arebha be:
    #  #   - Dropout (ne2tel neurons 3ashwa2i wa2t el training)
    #  #   - BatchNorm (nestaqrar el inputs beta3t kol layer)
    #  #   - Weight Decay / L2 (ne3a2eb el weights el kbeera)
    # ---------------------------------------------------------------

    set_seed(SEED)
    train_loader, val_loader = build_loaders(
        X_train, y_train, X_val, y_val, batch_size=64
    )

    reg_model = RegularizedNN(input_dim, dropout_rate=0.3)
    count_parameters(reg_model)

    reg_model, reg_history = train_model(
        model          = reg_model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        device         = device,
        epochs         = 100,
        lr             = 1e-3,
        weight_decay   = 1e-4,          # L2 regularization via optimizer
                                        # L2 regularization men 5elal el optimizer
        stage_name     = "Step 4: Reduce Variance (RegularizedNN)",
        filename_prefix= "step4_reduce_variance",
    )

    reg_metrics = evaluate_model(
        reg_model, val_loader, device,
        stage_name="Step 4: Reduce Variance (RegularizedNN)"
    )

    # ---------------------------------------------------------------
    #  8. FINAL COMPARISON -- Which model won?
    #  el m2arna el akhira -- anhi model kasab?
    # ---------------------------------------------------------------

    print("\n" + "=" * 60)
    print("  FINAL RESULTS COMPARISON")
    print("=" * 60)

    results = pd.DataFrame({
        "Step 2 -- Baseline (SimpleNN)":       baseline_metrics,
        "Step 3 -- Reduce Bias (DeeperNN)":    deeper_metrics,
        "Step 4 -- Reduce Variance (RegNN)":   reg_metrics,
    }).T

    print(f"\n{results.to_string()}")

    # Highlight the winner
    # warre el model el kaseb
    best_model_name = results["f1"].idxmax()
    best_f1 = results["f1"].max()
    print(f"\n  >> Best model by F1: {best_model_name} (F1 = {best_f1:.4f})")

    # Save the comparison table to CSV
    # e7faz gadwel el m2arna f CSV file
    results.to_csv("outputs/results_comparison.csv")
    print("  -> Results saved to outputs/results_comparison.csv")

    # ---------------------------------------------------------------
    #  9. SAVE BEST MODEL WEIGHTS
    #  e7faz el weights beta3t a7san model
    # ---------------------------------------------------------------

    # We'll save the regularized model since it should generalize best
    # hane7faz el regularized model 3ashan el mafrood yekon a7san f el generalization
    model_path = "outputs/best_model.pth"
    torch.save(reg_model.state_dict(), model_path)
    print(f"  -> Best model saved to {model_path}")

    print("\n" + "=" * 60)
    print("  EXPERIMENT COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()