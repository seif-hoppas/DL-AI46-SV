"""
data_loader.py -- Loading & PyTorch Dataset Utilities
=====================================================
Why a separate file?  Keeps data I/O isolated so the rest of
the codebase never worries about *where* the data lives.
# leh file l wa7do? 3ashan el data I/O yeb2a ma3zool w ba2y el code
# maya7taresh yefakar el data gaya menen.

We also define a custom PyTorch Dataset here so the DataLoader
can feed batches directly to our neural network.
# kaman bene3aref PyTorch Dataset hena 3ashan el DataLoader
# ye2dar yewaddi batches lel neural network 3ala tool.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ------------------------------------------------------------------
# 1. Raw CSV loader (nothing fancy, just a clean read)
# ta7mil el CSV (7aga baseeta, bass 2eraya nadeefa)
# ------------------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame:
    """
    Reads the CSV into a DataFrame.
    We keep this trivial on purpose -- any heavy lifting
    (encoding, scaling, balancing) belongs in preprocessor.py.
    # beye2ra el CSV f DataFrame.
    # 5alleinaha baseeta 3an 3amd -- ay sho8l te2il
    # (encoding, scaling, balancing) makano f preprocessor.py.
    """
    df = pd.read_csv(path)
    print(f"[data_loader] Loaded {len(df)} rows from {path}")
    return df


# ------------------------------------------------------------------
# 2. PyTorch Dataset wrapper
# el wrapper beta3 PyTorch Dataset
# ------------------------------------------------------------------
class FailureDataset(Dataset):
    """
    Wraps NumPy arrays (features + labels) so PyTorch's DataLoader
    can shuffle, batch, and iterate over them automatically.
    Nothing magical -- it just converts rows into tensors on the fly.
    # byeleff NumPy arrays (features + labels) 3ashan el DataLoader
    # beta3 PyTorch ye2dar ye-shuffle, ye-batch, w yeleff 3alehom automatic.
    # mafesh se7r -- bass bye7awel el rows le tensors wa2t el ta7mil.
    """

    def __init__(self, X, y):
        # float32 for features (required by Linear layers)
        # float32 lel features (el Linear layers btehtago)
        self.X = torch.tensor(X, dtype=torch.float32)
        # float32 for labels too (BCEWithLogitsLoss expects matching types)
        # float32 lel labels kaman (BCEWithLogitsLoss 3ayez nafs el type)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ------------------------------------------------------------------
# 3. Convenience: build DataLoaders in one call
# tool sareee3a: eb2ni el DataLoaders f call wa7da
# ------------------------------------------------------------------
def build_loaders(X_train, y_train, X_val, y_val, batch_size=64):
    """
    Returns ready-to-use train & validation DataLoaders.
    Shuffling is ON for training, OFF for validation (as it should be).
    # beyregga3 DataLoaders gehzeen lel isti5dam.
    # el shuffling mashghool lel training, w ma2fool lel validation (zay ma el mafrood).
    """
    train_ds = FailureDataset(X_train, y_train)
    val_ds   = FailureDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"[data_loader] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader
