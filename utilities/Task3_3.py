from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def load_embeddings(embeddings_path: str | Path, labels_path: str | Path | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads embeddings and labels from file.

    Supported formats:
    - .npy or .npz (expects "embeddings" and optionally "labels")
    - .csv or .parquet (expects columns prefixed with emb_ or all numeric columns; labels in "Label" if present)
    """
    embeddings_path = Path(embeddings_path)

    if embeddings_path.suffix in {".npy"}:
        embeddings = np.load(embeddings_path)
        labels = None
    elif embeddings_path.suffix in {".npz"}:
        data = np.load(embeddings_path)
        embeddings = data["embeddings"]
        labels = data["labels"] if "labels" in data else None
    elif embeddings_path.suffix in {".csv"}:
        df = pd.read_csv(embeddings_path)
        labels = df["Label"].values if "Label" in df.columns else None
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        if not emb_cols:
            emb_cols = df.select_dtypes(include=[np.number]).columns.difference(["Label"]).tolist()
        embeddings = df[emb_cols].to_numpy()
    elif embeddings_path.suffix in {".parquet"}:
        df = pd.read_parquet(embeddings_path)
        labels = df["Label"].values if "Label" in df.columns else None
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        if not emb_cols:
            emb_cols = df.select_dtypes(include=[np.number]).columns.difference(["Label"]).tolist()
        embeddings = df[emb_cols].to_numpy()
    else:
        raise ValueError(f"Unsupported embeddings format: {embeddings_path.suffix}")

    if labels is None and labels_path is not None:
        labels_path = Path(labels_path)
        if labels_path.suffix == ".npy":
            labels = np.load(labels_path)
        elif labels_path.suffix == ".csv":
            labels = pd.read_csv(labels_path).iloc[:, 0].values
        elif labels_path.suffix == ".parquet":
            labels = pd.read_parquet(labels_path).iloc[:, 0].values
        else:
            raise ValueError(f"Unsupported labels format: {labels_path.suffix}")

    if labels is None:
        raise ValueError("Labels not found. Provide labels_path or include Label column / labels array.")

    return embeddings, labels


def run_tsne(embeddings: np.ndarray, seed: int = 42, perplexity: int = 30) -> np.ndarray:
    return TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(embeddings)


def run_umap(embeddings: np.ndarray, seed: int = 42, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    try:
        import umap  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("UMAP is not installed. Add umap-learn or use --method tsne.") from exc

    reducer = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
    )
    return reducer.fit_transform(embeddings)


def clustering_metrics(embeddings_2d: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    return {
        "silhouette": float(silhouette_score(embeddings_2d, labels)),
        "davies_bouldin": float(davies_bouldin_score(embeddings_2d, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(embeddings_2d, labels)),
    }


def plot_embeddings(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str | Path | None = None,
) -> None:
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="coolwarm", s=12, alpha=0.7)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(*scatter.legend_elements(), title="Label", loc="best")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def run_visualization(
    embeddings_path: str | Path,
    labels_path: str | Path | None,
    method: str = "tsne",
    seed: int = 42,
    output_dir: str | Path = "output/representations",
    **kwargs: Any,
) -> Dict[str, float]:
    embeddings, labels = load_embeddings(embeddings_path, labels_path)

    if method == "tsne":
        emb_2d = run_tsne(embeddings, seed=seed, perplexity=int(kwargs.get("perplexity", 30)))
    elif method == "umap":
        emb_2d = run_umap(
            embeddings,
            seed=seed,
            n_neighbors=int(kwargs.get("n_neighbors", 15)),
            min_dist=float(kwargs.get("min_dist", 0.1)),
        )
    else:
        raise ValueError("method must be 'tsne' or 'umap'")

    metrics = clustering_metrics(emb_2d, labels)
    method_name = method.upper()
    plot_embeddings(emb_2d, labels, f"{method_name} Visualization (Task 3.3)", Path(output_dir) / f"{method}_plot.png")
    return metrics
