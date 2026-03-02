"""
trainer.py — Training Loops & Evaluation (The Golden Rules Engine)
==================================================================
This module contains everything related to training and evaluating
our neural networks.  It follows the Golden Rules progression:
# ya3ni hena kol 7aga beta3t el training w el evaluation

  Step 1 -- Sanity Check:  overfit a SINGLE sample.
  Step 2 -- Baseline:      train SimpleNN on real data.
  Step 3 -- Reduce Bias:   train DeeperNN (more capacity).
  Step 4 -- Reduce Variance: train RegularizedNN + weight decay.

Each step has its own function so the flow is crystal clear.
# kol step leha function l wa7daha 3ashan el code yeb2a wade7

Reproducibility note:  all random seeds are set externally in main.py
via the `set_seed()` utility, so every run gives identical results.
# el seeds bet7ased fl main.py, fa kol mara teshghal el code haygib nafs el nateega
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

import matplotlib.pyplot as plt

# Where training curves get saved
# hena bne7aded el folder elly hay7ot feeh el loss curves
PLOTS_DIR = "outputs/training"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ===================================================================
#  CORE TRAINING LOOP  (used by every step)
#  el training loop el asasi elly kol step beyestakhdemo
# ===================================================================
def _train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Standard PyTorch training loop for one epoch.
    Returns average loss over all batches.
    # loop 3ady betrain epoch wa7da w betregga3 el average loss
    """
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Zero the gradients, forward pass, compute loss, backward, update weights
        # saffar el gradients, 3addi el data, e7seb el loss, back-propagation, 7addes el weights
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    return running_loss / len(loader.dataset)


# ===================================================================
#  VALIDATION LOOP
#  hena bne-evaluate el model 3ala el validation data men 8er training
# ===================================================================
@torch.no_grad()
def _validate(model, loader, criterion, device):
    """
    Runs the model in eval mode (no dropout, BN uses running stats).
    Returns average loss over the validation set.
    # benshghal el model f eval mode (el dropout beyetwaqaf w el BatchNorm biyesta5dem el running stats)
    """
    model.eval()
    running_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        running_loss += loss.item() * X_batch.size(0)

    return running_loss / len(loader.dataset)


# ===================================================================
#  FULL EVALUATION  (metrics + classification report)
#  hena bne7seb kol el metrics w bentaba3 el report el kamla
# ===================================================================
@torch.no_grad()
def evaluate_model(model, loader, device, stage_name=""):
    """
    Computes accuracy, precision, recall, F1 and prints a full
    classification report.  This is what we use to compare stages.
    # bne7seb el accuracy, precision, recall, f1 w bentabe3 ta2rir kamel 3ashan n2aren ben el models
    """
    model.eval()
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits  = model(X_batch)
        # Apply sigmoid then threshold at 0.5 to get binary predictions
        # bente3mel sigmoid w ba3den law >= 0.5 yeb2a 1, 8er keda 0
        preds   = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(y_batch.numpy().flatten().astype(int))

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)

    print(f"\n{'='*50}")
    print(f"  Evaluation -- {stage_name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=['No Failure', 'Failure'])}")
    print(f"Confusion Matrix:\n{confusion_matrix(all_labels, all_preds)}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ===================================================================
#  TRAINING CURVE PLOTTER
#  btersem el loss curve (train vs val) w te7fazha f disk
# ===================================================================
def _plot_curves(train_losses, val_losses, title, filename):
    """Saves a train vs validation loss curve to disk.
    # bte7faz sora lel train loss w el val loss 3ashan netabe3 el training"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train Loss", linewidth=2)
    ax.plot(val_losses,   label="Val Loss",   linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (BCE)")
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=120)
    plt.close(fig)
    print(f"[trainer] Saved loss curve -> {PLOTS_DIR}/{filename}")


# ===================================================================
#  STEP 1: SANITY CHECK  --  Overfit a single sample
#  awal 7aga: ne5od sample WA7DA w ne7awel el model ye7fazha
# ===================================================================
def sanity_check(model, X_sample, y_sample, device, epochs=200, lr=1e-2):
    """
    THE most important first step.  Take ONE sample and try to
    memorize it.  If the loss doesn't reach ~0, something is broken:
    wrong loss function, wrong shapes, bad data pipeline, etc.
    # ahm 7aga tebi2a: 5od sample WA7DA w 7awel el model ye7fazha.
    # law el loss ma nazlsh le 0, yeb2a fi 7aga 8alat: loss function, shapes, aw data pipeline

    We expect the loss to drop from ~0.7 to nearly 0 in <200 epochs.
    If it doesn't, stop and debug before wasting hours on full training.
    # el mafrood el loss yenzil men 0.7 le 2oreeb men 0 f a2al men 200 epoch.
    # law ma7asalsh keda, wa2af w debug abl ma ted3 wa2t f training kamel.
    """
    print("\n" + "=" * 50)
    print("  STEP 1: SANITY CHECK -- Overfit a single sample")
    print("=" * 50)

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Single sample as tensors
    # na2el el sample el wa7da le tensor
    X_t = torch.tensor(X_sample, dtype=torch.float32).unsqueeze(0).to(device)
    y_t = torch.tensor([y_sample], dtype=torch.float32).unsqueeze(1).to(device)

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_t)
        loss   = criterion(logits, y_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 50 == 0 or epoch == 1:
            pred = (torch.sigmoid(logits) >= 0.5).int().item()
            print(f"  Epoch {epoch:>3d} | Loss: {loss.item():.6f} | "
                  f"Pred: {pred} | True: {int(y_sample)}")

    # Did it work?  Final loss should be close to zero
    # hal nege7? el loss el akhir lazem yekoon 2oreib men el sefr
    final_loss = losses[-1]
    final_pred = (torch.sigmoid(model(X_t)) >= 0.5).int().item()

    if final_loss < 0.01 and final_pred == int(y_sample):
        print("\n  [PASSED] SANITY CHECK PASSED -- The model can memorize a single sample.")
        print("    Pipeline is working correctly. Safe to proceed!\n")
        # 3azama! el model 2eder ye7faz el sample. el pipeline sha8ala tamam, kamel!
    else:
        print("\n  [FAILED] SANITY CHECK FAILED -- Loss didn't converge or prediction is wrong.")
        print("    Debug before continuing! Check model, loss, data shapes.\n")
        # el model feshel ye7faz sample wa7da. lazem te-debug abl ma tekamel!

    # Save a mini loss curve for the sanity check
    # e7faz sora lel loss curve beta3t el sanity check
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, linewidth=2, color="crimson")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Sanity Check -- Single Sample Overfitting", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "step1_sanity_check.png"), dpi=120)
    plt.close(fig)

    return model, losses


# ===================================================================
#  GENERIC TRAIN FUNCTION (for Steps 2-4)
#  el training function el 3amma elly benstakhdmha lel steps 2 w 3 w 4
# ===================================================================
def train_model(model, train_loader, val_loader, device,
                epochs=50, lr=1e-3, weight_decay=0.0,
                stage_name="Model", filename_prefix="model"):
    """
    Full training loop with validation tracking.
      - Logs train + val loss every 5 epochs
      - Saves loss curves to disk
      - Returns the trained model and history
    # loop kamel lel training ma3 validation tracking
    # beyekteb el loss kol 5 epochs, w beye7faz el loss curves, w beyregga3 el model w el history
    """
    print(f"\n{'='*50}")
    print(f"  Training: {stage_name}")
    print(f"  Epochs: {epochs} | LR: {lr} | Weight Decay: {weight_decay}")
    print(f"{'='*50}")

    model     = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    val_losses   = []

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = _validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save the training curves
    # e7faz el loss curves 3ala el disk
    _plot_curves(train_losses, val_losses, stage_name,
                 f"{filename_prefix}_loss_curve.png")

    return model, {"train_losses": train_losses, "val_losses": val_losses}