from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def daily_weather_signatures(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    value_columns: tuple[str, ...] = ("temperature", "humidity", "pressure", "wind_speed"),
) -> pd.DataFrame:
    """Aggregate hourly weather into compact daily statistical signatures."""
    out = frame.copy()
    ts = pd.to_datetime(out[timestamp_col], utc=True)
    out["_day"] = ts.dt.floor("D")
    grouped = out.groupby("_day")[list(value_columns)].agg(["mean", "std", "min", "max"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    return grouped.dropna()


def cluster_weather_regimes(
    signatures: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
):
    scaler = StandardScaler()
    z = scaler.fit_transform(signatures)
    n_components = min(6, z.shape[1], max(2, z.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=random_state)
    embedding = pca.fit_transform(z)
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = model.fit_predict(embedding)
    result = pd.DataFrame(
        {"regime": labels, "pc1": embedding[:, 0], "pc2": embedding[:, 1]},
        index=signatures.index,
    )
    return result, {"scaler": scaler, "pca": pca, "clusterer": model}
