"""Q4.2 – LLM embeddings for ICU mortality classification (linear probe + visualization).

Pipeline
--------
1. Load Set A (train) and Set C (test) from output/processed_sets/.
2. Build a compact clinical text summary for each patient (same as Q4.1).
3. Query a local Ollama embedding model to get one vector per patient summary.
4. Train a LogisticRegression linear probe on Set A embeddings; evaluate on Set C.
5. Report AuROC / AuPRC and compare against section 2 & 3 results.
6. Extract pooled encoder representations from the Task 2.3a SimpleTransformer
   (domain-specific supervised model) for Set C patients.
7. Visualize both embedding spaces side-by-side with UMAP (t-SNE fallback),
   coloured by in-hospital mortality label, and compare clustering structure.

Embeddings are cached to disk (.npz) so expensive re-computation is avoided on
subsequent runs.

Usage
-----
    python utilities/Task4_2.py
    python utilities/Task4_2.py --embed-model mxbai-embed-large  # different model
    python utilities/Task4_2.py --limit 200                      # quick test
    python utilities/Task4_2.py --no-domain                      # skip domain embeddings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import ollama

# ---------------------------------------------------------------------------
# Project imports  (script is run from the repo root)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utilities.Task4_1 import build_patient_summary, PROCESSED_DIR
from utilities.Task2_3a import (
    SimpleTransformer,
    load_and_preprocess_dataset,
    create_dataloader,
    CLIP,
    BATCH_SIZE,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR      = Path("output/llm")
PREPROCESSED_DIR = Path("output/preprocessed_imputed_datasets")
CHECKPOINT_3A   = Path("model_checkpoints/task_2_3a/run_20260329_164728/best_model.pt")

DEFAULT_EMBED_MODEL = "mxbai-embed-large"
SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Task 2.3a SimpleTransformer hyperparameters (must match training configuration)
_3A_D_MODEL       = 64
_3A_NHEAD         = 1
_3A_NUM_LAYERS    = 2
_3A_DIM_FFORWARD  = 128
_3A_DROPOUT       = 0.1

# ---------------------------------------------------------------------------
# LLM embedding helpers
# ---------------------------------------------------------------------------

def get_embedding(text: str, model: str) -> np.ndarray:
    """Return a 1-D vector embedding from the Ollama embedding endpoint."""
    response = ollama.embed(model=model, input=text)
    return np.array(response["embeddings"][0], dtype=np.float32)


def build_llm_embeddings(
    df: pd.DataFrame,
    model: str,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed every patient in *df* using the Ollama embedding model.

    Returns
    -------
    X : (n_patients, embed_dim) float32 array
    y : (n_patients,) int array  (0 = survived, 1 = died)
    """
    patient_ids    = df["PatientID"].unique()
    labels_series  = df.groupby("PatientID")["Label"].last().astype(int)

    if limit is not None:
        patient_ids = patient_ids[:limit]

    X, y = [], []
    for i, pid in enumerate(patient_ids):
        pat_df  = df[df["PatientID"] == pid]
        summary = build_patient_summary(pat_df)
        emb     = get_embedding(summary, model)
        X.append(emb)
        y.append(int(labels_series[pid]))

        if (i + 1) % 50 == 0 or (i + 1) == len(patient_ids):
            print(f"  [{i + 1}/{len(patient_ids)}] embedded (dim={len(emb)})")

    return np.vstack(X), np.array(y, dtype=int)


# ---------------------------------------------------------------------------
# Domain-specific (Task 2.3a SimpleTransformer) embedding extraction
# ---------------------------------------------------------------------------

def extract_domain_embeddings(
    df: pd.DataFrame,
    feature_cols: list[str],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the Task 2.3a SimpleTransformer and return the pooled encoder
    representations (the layer *before* the linear classifier head) for
    every patient in *df*.

    Returns
    -------
    emb    : (n_patients, d_model) float32 array
    labels : (n_patients,) float32 array
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    num_features = len(feature_cols)

    model = SimpleTransformer(
        num_features    = num_features,
        d_model         = _3A_D_MODEL,
        nhead           = _3A_NHEAD,
        num_layers      = _3A_NUM_LAYERS,
        dim_feedforward = _3A_DIM_FFORWARD,
        dropout         = _3A_DROPOUT,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    loader = create_dataloader(df, feature_cols, batch_size=BATCH_SIZE, shuffle=False)

    all_embs, all_labels = [], []
    with torch.no_grad():
        for x, mask, labels in loader:
            x, mask = x.to(device), mask.to(device)

            # Reproduce the SimpleTransformer forward pass up to the pooling step,
            # stopping before the final classifier Linear layer.
            x_proj = model.input_projection(x)                                      # (B, T, d_model)
            x_enc  = model.transformer_encoder(x_proj, src_key_padding_mask=mask)   # (B, T, d_model)

            inv_mask = 1.0 - mask.unsqueeze(-1).float()                             # (B, T, 1)
            sum_x    = torch.sum(x_enc.to(torch.float32) * inv_mask, dim=1)         # (B, d_model)
            count_x  = torch.clamp(inv_mask.sum(dim=1), min=1e-9)                   # (B, 1)
            pooled   = (sum_x / count_x)                                             # (B, d_model)

            all_embs.append(pooled.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.vstack(all_embs), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_embeddings(
    emb_llm:    np.ndarray,
    y_llm:      np.ndarray,
    emb_domain: np.ndarray,
    y_domain:   np.ndarray,
    save_path:  Path | None = None,
) -> None:
    """
    Side-by-side scatter plot of LLM embeddings vs domain-specific embeddings,
    projected to 2-D with UMAP (falls back to t-SNE if umap-learn is absent).
    Points are coloured by in-hospital mortality label.
    """
    try:
        import umap  # noqa: PLC0415
        print("  Fitting UMAP on LLM embeddings …")
        proj_llm = umap.UMAP(n_components=2, random_state=SEED).fit_transform(emb_llm)
        print("  Fitting UMAP on domain-specific embeddings …")
        proj_dom = umap.UMAP(n_components=2, random_state=SEED).fit_transform(emb_domain)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE  # noqa: PLC0415
        print("  umap-learn not installed – falling back to t-SNE …")
        print("  Fitting t-SNE on LLM embeddings …")
        proj_llm = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(emb_llm)
        print("  Fitting t-SNE on domain-specific embeddings …")
        proj_dom = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(emb_domain)
        method = "t-SNE"

    COLORS = {0: "#4C9BE8", 1: "#E8524C"}  # blue = survived, red = died

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Patient Embedding Visualisation: LLM vs Domain-Specific (Task 2.3a)",
        fontsize=13,
    )

    for ax, proj, y, title in [
        (axes[0], proj_llm, y_llm,    f"LLM Embeddings ({method})"),
        (axes[1], proj_dom, y_domain, f"Domain-Specific Embeddings ({method})"),
    ]:
        colors = [COLORS[int(lb)] for lb in y]
        ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=6, alpha=0.5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f"{method} 1")
        ax.set_ylabel(f"{method} 2")
        ax.grid(True, linestyle="--", alpha=0.3)

    legend_handles = [
        mpatches.Patch(color=COLORS[0], label="Survived (label = 0)"),
        mpatches.Patch(color=COLORS[1], label="Died     (label = 1)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.04),
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Q4.2 LLM embedding linear probe + visualisation")
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help=f"Ollama embedding model name (default: {DEFAULT_EMBED_MODEL}). "
             f"Pull it first with:  ollama pull {DEFAULT_EMBED_MODEL}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N patients per split (useful for quick testing)",
    )
    parser.add_argument(
        "--no-domain",
        action="store_true",
        help="Skip domain-specific embedding extraction and visualisation",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    # -----------------------------------------------------------------------
    # 1. Load datasets
    # -----------------------------------------------------------------------
    print("Loading datasets …")
    df_a = pd.read_parquet(PROCESSED_DIR / "processed_set_a.parquet")
    df_c = pd.read_parquet(PROCESSED_DIR / "processed_set_c.parquet")
    print(f"  Set A: {df_a['PatientID'].nunique()} patients")
    print(f"  Set C: {df_c['PatientID'].nunique()} patients")

    # -----------------------------------------------------------------------
    # 2 & 3. Build / load LLM embeddings  (cached as .npz)
    # -----------------------------------------------------------------------
    model_tag = args.embed_model.replace(":", "_").replace("/", "_")
    cache_a = OUTPUT_DIR / f"llm_emb_a_{model_tag}.npz"
    cache_c = OUTPUT_DIR / f"llm_emb_c_{model_tag}.npz"

    if cache_a.exists() and cache_c.exists():
        print("\nLoading cached LLM embeddings …")
        d = np.load(cache_a); X_train, y_train = d["X"], d["y"]
        d = np.load(cache_c); X_test,  y_test  = d["X"], d["y"]
        print(f"  Train: {X_train.shape}   Test: {X_test.shape}")
    else:
        print(f"\nEmbedding Set A via '{args.embed_model}' …")
        X_train, y_train = build_llm_embeddings(df_a, args.embed_model, args.limit)
        np.savez(cache_a, X=X_train, y=y_train)
        print(f"  Saved → {cache_a}")

        print(f"\nEmbedding Set C via '{args.embed_model}' …")
        X_test, y_test = build_llm_embeddings(df_c, args.embed_model, args.limit)
        np.savez(cache_c, X=X_test, y=y_test)
        print(f"  Saved → {cache_c}")

    # -----------------------------------------------------------------------
    # 4. Linear probe: LogisticRegression trained on LLM embeddings
    # -----------------------------------------------------------------------
    print("\nTraining linear probe (LogisticRegression) on LLM embeddings …")
    probe = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
    probe.fit(X_train, y_train)

    probs = probe.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)

    print(f"\n{'='*55}")
    print(f"  Q4.2 LLM Embedding Results  (model: {args.embed_model})")
    print(f"{'='*55}")
    print(f"  Embedding dim  : {X_train.shape[1]}")
    print(f"  Train patients : {len(y_train)}")
    print(f"  Test  patients : {len(y_test)}")
    print(f"  AuROC          : {auroc:.4f}")
    print(f"  AuPRC          : {auprc:.4f}")
    print(f"{'='*55}")

    # Persist summary row
    summary_path = OUTPUT_DIR / "q4_2_summary.csv"
    row = pd.DataFrame([{
        "embed_model" : args.embed_model,
        "n_train"     : len(y_train),
        "n_test"      : len(y_test),
        "embed_dim"   : int(X_train.shape[1]),
        "auroc_probe" : auroc,
        "auprc_probe" : auprc,
    }])
    if summary_path.exists():
        pd.concat([pd.read_csv(summary_path), row], ignore_index=True).to_csv(summary_path, index=False)
    else:
        row.to_csv(summary_path, index=False)
    print(f"\n  Results appended → {summary_path}")

    # -----------------------------------------------------------------------
    # 5. Domain-specific embeddings (Task 2.3a SimpleTransformer)
    # -----------------------------------------------------------------------
    if args.no_domain:
        print("\n--no-domain flag set; skipping domain embedding extraction.")
        return

    domain_cache = OUTPUT_DIR / "domain_emb_c_task23a.npz"
    emb_domain, y_domain = None, None

    if domain_cache.exists():
        print("\nLoading cached domain-specific (Task 2.3a) embeddings …")
        d = np.load(domain_cache)
        emb_domain, y_domain = d["X"], d["y"]
        print(f"  Shape: {emb_domain.shape}")
    elif CHECKPOINT_3A.exists():
        print(f"\nExtracting Task 2.3a SimpleTransformer embeddings for Set C …")
        df_c_prep, feature_cols = load_and_preprocess_dataset(
            PREPROCESSED_DIR / "preprocessed_set_c.parquet", "Set C (3a)", CLIP
        )
        emb_domain, y_domain = extract_domain_embeddings(
            df_c_prep, feature_cols, CHECKPOINT_3A, DEVICE
        )
        np.savez(domain_cache, X=emb_domain, y=y_domain)
        print(f"  Saved → {domain_cache}  (shape={emb_domain.shape})")
    else:
        print(f"\nWARNING: Task 2.3a checkpoint not found at {CHECKPOINT_3A}")
        print("  Skipping domain embedding extraction and visualisation.")

    # -----------------------------------------------------------------------
    # 6. Visualise both embedding spaces
    # -----------------------------------------------------------------------
    if emb_domain is not None:
        print("\nGenerating embedding visualisations …")
        # Align sample counts so both plots show the same number of patients
        n = min(len(X_test), len(emb_domain))
        plot_embeddings(
            X_test[:n],      y_test[:n],
            emb_domain[:n],  y_domain[:n],
            save_path=OUTPUT_DIR / "q4_2_embedding_comparison.png",
        )
    else:
        print("\nSkipping visualisation (no domain embeddings available).")

    print("\nDone.")


if __name__ == "__main__":
    main()