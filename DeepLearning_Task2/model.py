"""
model.py -- Neural Network Architectures (Simple to Complex)
==========================================================
We define THREE models here, each corresponding to a stage
in the Golden Rules of NN training:
# 3andena 3 models hena, kol wa7ed bey-maseel marhala fl Golden Rules

  1. SimpleNN   -- Tiny network for the Sanity Check & Baseline.
                   If this can't overfit a single sample, something
                   is fundamentally broken in our pipeline.
                   # network so8ayara lel sanity check. law mesh 2adra
                   # te7faz sample wa7da, yeb2a fi 7aga 8alat gedan.

  2. DeeperNN   -- More layers & neurons to Reduce Bias.
                   This is where we throw capacity at the problem
                   to push training accuracy higher.
                   # layers w neurons akthar 3ashan ne2allel el bias.
                   # hena bne-zawed el capacity 3ashan el accuracy ye3la.

  3. RegularizedNN -- Same capacity as DeeperNN but with Dropout
                      and BatchNorm to Reduce Variance (overfitting).
                      This is the "production-ready" architecture.
                      # nafs el capacity bs ma3 Dropout w BatchNorm
                      # 3ashan ne2allel el overfitting. da el model el gahiz lel production.

All models output a SINGLE raw logit (no sigmoid!) because we
use BCEWithLogitsLoss which applies sigmoid internally -- this is
numerically more stable than doing sigmoid + BCELoss.
# kol el models betraga3 logit wa7ed men 8er sigmoid, 3ashan
# BCEWithLogitsLoss beye3mel sigmoid gowah -- w da a7san 3adadiyyan.
"""

import torch
import torch.nn as nn


# ------------------------------------------------------------------
# Stage 1 & 2: Simple shallow network (Sanity Check + Baseline)
# network basita lel sanity check w el baseline
# ------------------------------------------------------------------
class SimpleNN(nn.Module):
    """
    Two hidden layers, nothing fancy.
    Good enough to verify the pipeline works and establish a baseline.
    If this can't memorize ONE sample, we have a bug.
    # etnen hidden layers, 7aga baseeta gidan.
    # kefaya 3ashan net2aked en el pipeline sha8ala w ne3mel baseline.
    # law mesh 2adra te7faz sample WA7DA, yeb2a 3andena bug.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),     # Single logit output for binary classification
                                  # logit wa7ed lel binary classification
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------
# Stage 3: Deeper network to reduce bias (fix underfitting)
# network a3ma2 3ashan ne2allel el bias (nehal el underfitting)
# ------------------------------------------------------------------
class DeeperNN(nn.Module):
    """
    Four hidden layers with increasing-then-decreasing width.
    More parameters = more capacity = can learn harder patterns.
    We use this when SimpleNN plateaus and underfits.
    # arba3 hidden layers, el 3ard beyezyed ba3den bye2el.
    # parameters akthar = capacity a3la = ye2dar yet3alem patterns as3ab.
    # bnesta5demo lama el SimpleNN yewsal le plateau w ye3mel underfitting.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------
# Stage 4: Regularized network to reduce variance (fix overfitting)
# network ma3 regularization 3ashan ne2allel el overfitting
# ------------------------------------------------------------------
class RegularizedNN(nn.Module):
    """
    Same depth as DeeperNN, but with BatchNorm + Dropout after
    each hidden layer.
    # nafs 3om2 el DeeperNN, bs ma3 BatchNorm + Dropout ba3d kol hidden layer.

    - BatchNorm stabilizes training and lets us use higher learning rates.
    - Dropout (p=0.3) randomly zeroes neurons during training,
      forcing the network to not rely on any single feature too much.
    # - BatchNorm biyestaqrar el training w beyesma7lena nesta5dem learning rates a3la.
    # - Dropout (p=0.3) biye-saffar neurons 3ashwa2i wa2t el training,
    #   3ashan el network maya3temedsh 3ala feature wa7da awi.

    Combined with weight_decay in the optimizer (L2 regularization),
    this is our best shot at generalizing well on unseen data.
    # ma3 el weight_decay fl optimizer (L2 regularization),
    # da a7san model 3andena yegeneralize 3ala data gdeeda.
    """

    def __init__(self, input_dim, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------
# Helper: count total trainable parameters
# tool basita: bte7seb 3adad el parameters elly el model beyedarab 3aleha
# ------------------------------------------------------------------
def count_parameters(model):
    """Quick util to print how many parameters a model has.
    # beteba3 kam parameter el model 3ando"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {model.__class__.__name__} -- {total:,} trainable parameters")
    return total
