"""
eda.py -- Exploratory Data Analysis
====================================
A quick visual tour of the dataset before we do anything else.
The goal is simple:  understand distributions, spot imbalance,
and catch anything weird *before* it bites us during training.
# gawla basareyya sari3a 3al dataset abl ma ne3mel ay 7aga tanya.
# el hadaf basit: nefham el distributions, nelaqy el imbalance,
# w nemsek ay 7aga 8ariba *abl* ma te2azina wa2t el training.

All plots are saved to `outputs/eda/` so they're reproducible
and easy to include in reports.
# kol el plots btet7efaz f `outputs/eda/` 3ashan tekoon reproducible
# w sahla nede5alha f ay report.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

# Where we'll dump our plots -- no manual folder creation needed
# hena el folder elly hane7ot feeh el plots -- mesh me7tagen ne3melo manually
EDA_DIR = "outputs/eda"
os.makedirs(EDA_DIR, exist_ok=True)

# Seaborn styling -- makes everything look nicer with zero effort
# 7atena style 7elw le seaborn -- kol 7aga hatetla men 8er magehood
sns.set_theme(style="whitegrid", palette="muted")


# ------------------------------------------------------------------
# 1. Feature distributions split by target class
# tawzi3 kol feature m2asem 7asab el class (failure wla la2)
# ------------------------------------------------------------------
def plot_features(df):
    """
    For each feature (skipping id & Product ID):
      - Numeric cols with >=5 unique values  -> histogram + KDE
      - Everything else                      -> count plot
    # le kol feature (benskip el id w Product ID):
    #   - el columns el ra2ameyya elly 3andaha >=5 unique values -> histogram + KDE
    #   - ay 7aga tanya -> count plot

    Every plot is coloured by 'Machine failure' so we can
    immediately see if a feature separates the classes.
    # kol plot mel2awn 7asab 'Machine failure' 3ashan neshof
    # 3ala tool law el feature deh btefre2 bein el classes.
    """
    print("[eda] Plotting feature distributions...")

    # We skip the first two columns (id, Product ID) -- they're identifiers, not features
    # bnekamel el awel 2 columns (id, Product ID) -- dol identifiers mesh features
    feature_cols = df.columns[2:]

    for col in feature_cols:
        fig, ax = plt.subplots(figsize=(8, 4))

        if df[col].dtype.kind in "if" and df[col].nunique() >= 5:
            # Continuous-ish feature -> histogram with KDE overlay
            # feature ra2ameyya -> histogram ma3 KDE fo2aha
            sns.histplot(data=df, x=col, hue="Machine failure",
                         kde=True, ax=ax, palette="Set2")
        else:
            # Categorical / low-cardinality -> bar chart
            # categorical aw 2elelet el values -> bar chart
            sns.countplot(data=df, x=df[col].astype(str), hue="Machine failure",
                          dodge=True, ax=ax, palette="Set2")

        ax.set_title(f"Distribution of {col}", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, f"feature_{col.replace(' ', '_')}.png"), dpi=120)
        plt.close(fig)   # Close to free memory (important when looping!)
                         # e2fel el figure 3ashan ne-fady el memory (mohim lama ne3mel loop!)

    print(f"[eda] Saved feature plots to {EDA_DIR}/")


# ------------------------------------------------------------------
# 2. Target distribution (the big-picture class imbalance check)
# tawzi3 el target (fah el sora el kbeera lel class imbalance)
# ------------------------------------------------------------------
def plot_target(df):
    """
    Simple bar chart of the target column.
    If one bar towers over the other, we know we need to handle
    class imbalance before training (spoiler: it does).
    # bar chart basit lel target column.
    # law bar wa7ed 3ali awy 3an el tany, yeb2a lazem ne3aleg
    # el class imbalance abl el training (spoiler: laa2 da mawgood).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Machine failure", hue="Machine failure",
                  ax=ax, palette="Set2", legend=False)
    ax.set_title("Machine Failure Distribution (0 = OK, 1 = Failure)", fontsize=13)

    # Print the counts right on the bars -- visual confirmation
    # ekteb el 3adad fo2 el bars -- ta2kid basari
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "target_distribution.png"), dpi=120)
    plt.close(fig)
    print(f"[eda] Saved target distribution plot to {EDA_DIR}/")


# ------------------------------------------------------------------
# 3. Correlation heatmap (numerics only)
# 5aretat el irtibatat (lel ar2am bass)
# ------------------------------------------------------------------
def plot_correlation(df):
    """
    Quick heatmap of Pearson correlations between numeric features.
    Helps spot multicollinearity and strong predictors at a glance.
    # heatmap sari3a lel Pearson correlations bein el features el ra2ameyya.
    # btesa3ed nelaqy el multicollinearity w el predictors el 2aweyya be naza.
    """
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close(fig)
    print(f"[eda] Saved correlation heatmap to {EDA_DIR}/")