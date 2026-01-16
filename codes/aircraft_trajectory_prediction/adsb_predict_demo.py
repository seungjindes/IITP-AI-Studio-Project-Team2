#!/usr/bin/env python3
"""
ADS-B 3D Trajectory Prediction Demo (HexIdent -> latest segment -> future prediction)

You asked: "처음부터 다시 써 줘" + altitude 중요(2D 전환 X) + 정확도 생략.

This script is built to be:
- Simple
- Robust against FlightID=0
- Robust against mixed dtypes / missing values
- Clear failure messages when *3D altitude itself is unusable* (e.g., all NULL or all 0)

What it does:
1) Pull raw points for a HexIdent from BigQuery within a date range (or auto-lookback).
2) Preprocess strictly for 3D: lat/lon/alt must exist after resample+interpolation.
3) Segment by time gaps to approximate flights.
4) Train a Bi-LSTM next-step predictor on past segments; if none, train on the latest segment itself.
5) Predict future points from the latest window and save plots + print a short preview.

Install:
  pip install google-cloud-bigquery pandas numpy scikit-learn torch matplotlib

Run:
  python adsb_predict_demo.py --hexident 43BF95 --train-start 2025-10-01 --train-end 2026-01-01

Or auto-lookback:
  python adsb_predict_demo.py --hexident 43BF95
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter

from google.cloud import bigquery
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# Utilities
# =========================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def utc_now() -> pd.Timestamp:
    # Safe across pandas versions (tz-aware)
    return pd.Timestamp.now(tz="UTC")

def date_str(ts: pd.Timestamp) -> str:
    return ts.date().isoformat()

def parse_date_utc(d: str) -> pd.Timestamp:
    # d: "YYYY-MM-DD"
    return pd.Timestamp(d).tz_localize("UTC")

def parse_ts_utc(s: str) -> pd.Timestamp:
    # s: ISO-like timestamp, e.g. "2025-01-01T12:34:56Z"
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def load_airports(csv_path: str, source_url: str, quiet: bool) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not quiet:
            print(f"[Airports] downloading from {source_url}")
        df = pd.read_csv(source_url)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)
    return df

def infer_airport_pair(
    lat_start: float,
    lon_start: float,
    lat_end: float,
    lon_end: float,
    airports: pd.DataFrame,
    max_km: float
) -> Tuple[Optional[pd.Series], Optional[pd.Series], float, float]:
    use = airports[
        airports["type"].isin(["large_airport", "medium_airport", "small_airport"])
        & airports["latitude_deg"].notna()
        & airports["longitude_deg"].notna()
    ].copy()
    if use.empty:
        return None, None, float("inf"), float("inf")

    def nearest(lat: float, lon: float) -> Tuple[pd.Series, float]:
        dists = use.apply(
            lambda r: haversine_km(lat, lon, float(r["latitude_deg"]), float(r["longitude_deg"])),
            axis=1,
        )
        idx = dists.idxmin()
        return use.loc[idx], float(dists.loc[idx])

    a1, d1 = nearest(lat_start, lon_start)
    a2, d2 = nearest(lat_end, lon_end)
    if d1 > max_km:
        a1 = None
    if d2 > max_km:
        a2 = None
    return a1, a2, d1, d2


# =========================
# Model & data
# =========================

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self) -> int:
        return len(self.X)
    def __getitem__(self, i: int):
        return self.X[i], self.y[i]

class BiLSTM(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 80, num_layers: int = 1, out_dim: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        out_dim: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.fc = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.fc(x)

def make_windows(arr: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i:i + window])
        y.append(arr[i + window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def make_windows_seq(features: np.ndarray, targets: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(features) - window):
        X.append(features[i:i + window])
        y.append(targets[i + window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_model(model: nn.Module, loader: DataLoader, device: str, epochs: int, lr: float) -> None:
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)

        if e == 1 or e % 5 == 0:
            print(f"Epoch {e:02d} | train MSE: {total / max(n,1):.6f}")

def rolling_predict_scaled(model: nn.Module, seed_window_scaled: np.ndarray, steps: int, device: str) -> np.ndarray:
    model.eval()
    w = seed_window_scaled.copy()
    preds = []
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(w[None, :, :], dtype=torch.float32).to(device)
            yhat = model(x).cpu().numpy()[0]
            preds.append(yhat)
            w = np.vstack([w[1:], yhat])
    return np.array(preds, dtype=np.float32)

def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    eps = 1e-6
    diff = pred - gt
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    mape = float(np.mean(np.abs(diff) / (np.abs(gt) + eps)))
    return {"MSE": mse, "MAE": mae, "MAPE": mape}

def compute_metrics_per_dim(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    Returns per-dimension metrics: lat, lon, alt.
    """
    out = {}
    for i, name in enumerate(["lat", "lon", "alt"]):
        out[name] = compute_metrics(gt[:, i], pred[:, i])
    return out

def make_windows_flat(features: np.ndarray, targets: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(features) - window):
        X.append(features[i:i + window].reshape(-1))
        y.append(targets[i + window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def build_features(coords: np.ndarray, feature_set: str, step_seconds: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    features: [lat, lon, alt] or + [dlat/dt, dlon/dt, dalt/dt]
    targets: [lat, lon, alt]
    """
    base = coords.astype(np.float32)
    if feature_set == "vel":
        diffs = np.diff(base, axis=0, prepend=base[0:1])
        vel = diffs / max(step_seconds, 1.0)
        feats = np.concatenate([base, vel], axis=1)
    else:
        feats = base
    return feats.astype(np.float32), base.astype(np.float32)

def fit_predict_baselines(
    coords: np.ndarray,
    window: int,
    train_ratio: float = 0.7,
    teacher_forcing: bool = True,
    scale_mode: str = "train",
    feature_set: str = "basic",
    step_seconds: float = 5.0
) -> Tuple[np.ndarray, dict]:
    """
    Train baseline regressors on the first 70% and predict the remaining 30%.
    Returns: (ground_truth_test, {name: pred_points})
    """
    n = len(coords)
    split = int(n * train_ratio)
    split = max(split, window + 5)
    train = coords[:split]
    test = coords[split:]

    scaler = MinMaxScaler()
    if scale_mode == "full":
        scaler.fit(coords)
    else:
        scaler.fit(train)
    feats_full, targets_full = build_features(coords, feature_set, step_seconds)
    feats_train, targets_train = build_features(train, feature_set, step_seconds)
    feats_test, targets_test = build_features(test, feature_set, step_seconds)

    train_s = scaler.transform(targets_train).astype(np.float32)
    full_s = scaler.transform(targets_full).astype(np.float32)
    # Scale features by applying same scaler to base coords and velocities separately
    if feature_set == "vel":
        feats_train_s = np.concatenate([train_s, feats_train[:, 3:]], axis=1)
        feats_full_s = np.concatenate([full_s, feats_full[:, 3:]], axis=1)
        feats_test_s = np.concatenate([scaler.transform(targets_test), feats_test[:, 3:]], axis=1)
    else:
        feats_train_s = train_s
        feats_full_s = full_s
        feats_test_s = scaler.transform(targets_test)

    X_train, y_train = make_windows_flat(feats_train_s, train_s, window)

    models = {
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "SVR": MultiOutputRegressor(
            SVR(C=10.0, epsilon=0.01, kernel="rbf")
        ),
        "KNN": MultiOutputRegressor(
            KNeighborsRegressor(n_neighbors=8, weights="distance")
        ),
        "MLP": MultiOutputRegressor(
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                learning_rate_init=1e-3,
                max_iter=400,
                random_state=42,
            )
        ),
    }

    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        if teacher_forcing:
            X_test, y_test = make_windows_flat(
                feats_full_s[split - window:], full_s[split - window:], window
            )
            pred_s = model.predict(X_test).astype(np.float32)
        else:
            def _rolling_predict(model, seed_window: np.ndarray, steps: int) -> np.ndarray:
                w = seed_window.copy()
                preds = []
                for _ in range(steps):
                    x = w.reshape(1, -1)
                    y = model.predict(x).astype(np.float32)[0]
                    preds.append(y)
                    w = np.vstack([w[1:], y])
                return np.array(preds, dtype=np.float32)
            seed = feats_train_s[-window:]
            pred_s = _rolling_predict(model, seed, len(test))
        preds[name] = scaler.inverse_transform(pred_s)

    if teacher_forcing:
        gt = scaler.inverse_transform(y_test).astype(np.float32)
    else:
        gt = targets_test.astype(np.float32)
    return gt, preds

def fit_predict_bilstm(
    coords: np.ndarray,
    window: int,
    epochs: int,
    lr: float,
    batch_size: int,
    train_ratio: float = 0.7,
    teacher_forcing: bool = True,
    scale_mode: str = "train",
    hidden_size: int = 80,
    num_layers: int = 1,
    feature_set: str = "basic",
    step_seconds: float = 5.0
) -> np.ndarray:
    n = len(coords)
    split = int(n * train_ratio)
    split = max(split, window + 5)
    train = coords[:split]
    test = coords[split:]

    scaler = MinMaxScaler()
    if scale_mode == "full":
        scaler.fit(coords)
    else:
        scaler.fit(train)
    feats_full, targets_full = build_features(coords, feature_set, step_seconds)
    feats_train, targets_train = build_features(train, feature_set, step_seconds)
    feats_test, targets_test = build_features(test, feature_set, step_seconds)

    train_s = scaler.transform(targets_train).astype(np.float32)
    full_s = scaler.transform(targets_full).astype(np.float32)
    if feature_set == "vel":
        feats_train_s = np.concatenate([train_s, feats_train[:, 3:]], axis=1)
        feats_full_s = np.concatenate([full_s, feats_full[:, 3:]], axis=1)
        feats_test_s = np.concatenate([scaler.transform(targets_test), feats_test[:, 3:]], axis=1)
    else:
        feats_train_s = train_s
        feats_full_s = full_s
        feats_test_s = scaler.transform(targets_test)

    X_train, y_train = make_windows_seq(feats_train_s, train_s, window)
    loader = DataLoader(SeqDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = BiLSTM(input_size=feats_train_s.shape[1], hidden_size=hidden_size, num_layers=num_layers, out_dim=3).to(device)
    train_model(model, loader, device=device, epochs=epochs, lr=lr)

    if teacher_forcing:
        X_test, _ = make_windows_seq(
            feats_full_s[split - window:], full_s[split - window:], window
        )
        with torch.no_grad():
            x = torch.tensor(X_test, dtype=torch.float32).to(device)
            pred_s = model(x).cpu().numpy()
    else:
        # Test: recursive rollout from last train window
        seed = feats_train_s[-window:]
        pred_s = rolling_predict_scaled(model, seed, len(test), device)
    pred = scaler.inverse_transform(pred_s)
    return pred

def fit_predict_transformer(
    coords: np.ndarray,
    window: int,
    epochs: int,
    lr: float,
    batch_size: int,
    train_ratio: float = 0.7,
    teacher_forcing: bool = True,
    scale_mode: str = "train",
    feature_set: str = "basic",
    step_seconds: float = 5.0,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
) -> np.ndarray:
    n = len(coords)
    split = int(n * train_ratio)
    split = max(split, window + 5)
    train = coords[:split]
    test = coords[split:]

    scaler = MinMaxScaler()
    if scale_mode == "full":
        scaler.fit(coords)
    else:
        scaler.fit(train)

    feats_full, targets_full = build_features(coords, feature_set, step_seconds)
    feats_train, targets_train = build_features(train, feature_set, step_seconds)
    feats_test, targets_test = build_features(test, feature_set, step_seconds)

    train_s = scaler.transform(targets_train).astype(np.float32)
    full_s = scaler.transform(targets_full).astype(np.float32)
    if feature_set == "vel":
        feats_train_s = np.concatenate([train_s, feats_train[:, 3:]], axis=1)
        feats_full_s = np.concatenate([full_s, feats_full[:, 3:]], axis=1)
        feats_test_s = np.concatenate([scaler.transform(targets_test), feats_test[:, 3:]], axis=1)
    else:
        feats_train_s = train_s
        feats_full_s = full_s
        feats_test_s = scaler.transform(targets_test)

    X_train, y_train = make_windows_seq(feats_train_s, train_s, window)
    loader = DataLoader(SeqDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerRegressor(
        input_size=feats_train_s.shape[1],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        out_dim=3,
    ).to(device)
    train_model(model, loader, device=device, epochs=epochs, lr=lr)

    if teacher_forcing:
        model.eval()
        X_test, _ = make_windows_seq(feats_full_s[split - window:], full_s[split - window:], window)
        with torch.no_grad():
            x = torch.tensor(X_test, dtype=torch.float32).to(device)
            pred_s = model(x).cpu().numpy()
    else:
        seed = feats_train_s[-window:]
        pred_s = rolling_predict_scaled(model, seed, len(test), device)
    pred = scaler.inverse_transform(pred_s)
    return pred


# =========================
# BigQuery + preprocessing
# =========================

def bq_client(project: str) -> bigquery.Client:
    return bigquery.Client(project=project)

def load_hex_points(
    client: bigquery.Client,
    table: str,
    hexident: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Pull raw points for a hexident in date window.
    Do not filter NULLs here; preprocessing handles it.
    """
    q = f"""
    SELECT
      TIMESTAMP(DATETIME(Date_MSG_Logged, Time_MSG_Logged), "UTC") AS ts,
      Latitude, Longitude, Altitude
    FROM `{table}`
    WHERE HexIdent = '{hexident}'
      AND Date_MSG_Logged BETWEEN DATE('{start_date}') AND DATE('{end_date}')
    ORDER BY ts;
    """
    return client.query(q).to_dataframe()

def preprocess_3d(
    df_raw: pd.DataFrame,
    resample: str,
    smooth_w: int,
    interp_limit: int,
    quiet: bool,
    use_outlier_filter: bool = True,
    use_resample: bool = True,
    use_interp: bool = True,
    use_smooth: bool = True
) -> pd.DataFrame:
    """
    Make preprocessing consistent with the working FlightID pipeline.

    Key changes:
    - Do NOT drop duplicate timestamps (that can kill valid data).
    - Aggregate duplicates by timestamp using mean().
    - Treat Altitude==0 as missing and interpolate.
    """
    if df_raw.empty:
        return df_raw

    d = df_raw.copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
    d = d.dropna(subset=["ts"]).sort_values("ts").set_index("ts")

    # Force numeric
    for c in ["Latitude", "Longitude", "Altitude"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Keep only coords
    d = d[["Latitude", "Longitude", "Altitude"]]

    # ✅ 핵심: 중복 ts를 drop 하지 말고 집계
    d = d.groupby(level=0).mean(numeric_only=True)

    # Basic sanity for lat/lon
    if use_outlier_filter:
        d = d[(d["Latitude"].between(-90, 90)) & (d["Longitude"].between(-180, 180))]

    if not quiet:
        alt_raw = d["Altitude"]
        nonnull = int(alt_raw.notna().sum())
        zratio = float((alt_raw.fillna(0) == 0).mean()) if len(alt_raw) else 1.0
        print(f"[Raw] rows={len(d):,} | alt_nonnull={nonnull:,} | alt_zero_ratio≈{zratio:.3f}")

    rs = resample.replace("S", "s")

    # Treat altitude==0 as missing (if interpolation enabled)
    if use_interp:
        d["Altitude"] = d["Altitude"].mask(d["Altitude"] == 0, np.nan)

    # Resample
    if use_resample:
        d = d.resample(rs).mean(numeric_only=True)

    # Interpolate small gaps for lat/lon
    if use_interp:
        d[["Latitude", "Longitude"]] = d[["Latitude", "Longitude"]].interpolate(
            method="time", limit=interp_limit, limit_direction="both"
        )
        # Altitude: interpolate then fill edges (prevents "all dropped")
        d["Altitude"] = d["Altitude"].interpolate(method="time", limit_direction="both").ffill().bfill()

    # Smooth
    if use_smooth:
        for c in ["Latitude", "Longitude", "Altitude"]:
            d[c] = d[c].rolling(smooth_w, center=True, min_periods=1).mean()

    d3 = d.dropna(subset=["Latitude", "Longitude", "Altitude"]).reset_index()

    if not quiet:
        if len(d3) == 0:
            print("[Preprocess3D] produced 0 rows after dropna(lat/lon/alt).")
        else:
            alt = d3["Altitude"].to_numpy(np.float32)
            span = float(alt.max() - alt.min())
            print(f"[Preprocess3D] rows={len(d3):,} | alt_span={span:.3f} | resample={rs} | interp_limit={interp_limit}")

    return d3


# =========================
# Segmentation
# =========================

def segment_by_gap(df: pd.DataFrame, gap_minutes: int) -> pd.DataFrame:
    """
    Create seg_id by time gaps >= gap_minutes.
    """
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d = d.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    dt = d["ts"].diff().dt.total_seconds().fillna(0)
    d["seg_id"] = (dt >= gap_minutes * 60).astype(int).cumsum()
    return d

@dataclass
class SegInfo:
    seg_id: int
    n: int
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp

def list_segments(seg_df: pd.DataFrame, min_seg_points: int) -> List[SegInfo]:
    g = seg_df.groupby("seg_id")["ts"].agg(["min", "max", "count"]).reset_index()
    g = g.rename(columns={"min": "start_ts", "max": "end_ts", "count": "n"})
    g = g[g["n"] >= min_seg_points].sort_values("end_ts", ascending=False)
    out: List[SegInfo] = []
    for _, r in g.iterrows():
        out.append(
            SegInfo(
                seg_id=int(r["seg_id"]),
                n=int(r["n"]),
                start_ts=pd.to_datetime(r["start_ts"], utc=True),
                end_ts=pd.to_datetime(r["end_ts"], utc=True),
            )
        )
    return out


# =========================
# Output
# =========================

def print_preview(hexident: str, seg: SegInfo, resample: str, horizon_steps: int, observed: np.ndarray, predicted: Optional[np.ndarray]):
    seconds_per_step = pd.to_timedelta(resample.replace("S", "s")).total_seconds()
    minutes = (horizon_steps * seconds_per_step) / 60.0 if horizon_steps else 0.0

    print("\n============================")
    print("3D Trajectory Preview")
    print("============================")
    print(f"HexIdent: {hexident}")
    print(f"Selected segment: seg_id={seg.seg_id} | points={seg.n} | {seg.start_ts} -> {seg.end_ts}")
    print(f"Resample: {resample} (~{seconds_per_step:.0f}s/step)")
    if horizon_steps and predicted is not None and len(predicted) > 0:
        print(f"Horizon: {horizon_steps} steps (~{minutes:.1f} min)")

    last_obs = observed[-1]
    print(f"\nLast observed: lat={last_obs[0]:.6f}, lon={last_obs[1]:.6f}, alt={last_obs[2]:.1f}")

    if predicted is not None and len(predicted) > 0:
        n_preview = min(8, len(predicted))
        print(f"\nNext {n_preview} predicted points:")
        for i in range(n_preview):
            p = predicted[i]
            tmin = (i + 1) * seconds_per_step / 60.0
            print(f"  +{tmin:5.1f}m  lat={p[0]:.6f}, lon={p[1]:.6f}, alt={p[2]:.1f}")
        print("")

def save_plots(out_dir: str, hexident: str, seg: SegInfo, observed: np.ndarray, predicted: Optional[np.ndarray]):
    plt.figure()
    plt.plot(observed[:, 1], observed[:, 0], label="Observed (latest)")
    if predicted is not None and len(predicted) > 0:
        plt.plot(predicted[:, 1], predicted[:, 0], label="Predicted (future)")
    plt.legend()
    plt.title(f"Trajectory | {hexident} | seg={seg.seg_id}")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    p1 = f"{out_dir}/trajectory_{hexident}_seg{seg.seg_id}.png"
    plt.savefig(p1, dpi=140, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(observed[:, 2], label="Observed Alt")
    if predicted is not None and len(predicted) > 0:
        x_pred = np.arange(len(observed), len(observed) + len(predicted))
        plt.plot(x_pred, predicted[:, 2], label="Pred Alt")
    plt.legend()
    plt.title(f"Altitude | {hexident} | seg={seg.seg_id}")
    plt.xlabel("Step"); plt.ylabel("Altitude")
    p2 = f"{out_dir}/altitude_{hexident}_seg{seg.seg_id}.png"
    plt.savefig(p2, dpi=140, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {p1}")
    print(f"[Saved] {p2}")

    # 3D trajectory plot (Latitude, Longitude, Altitude)
    def _downsample(arr: np.ndarray, max_points: int = 400) -> np.ndarray:
        if len(arr) <= max_points:
            return arr
        idx = np.linspace(0, len(arr) - 1, max_points).astype(int)
        return arr[idx]

    def _smooth_for_plot(arr: np.ndarray, window: int = 5) -> np.ndarray:
        if len(arr) < 3:
            return arr
        df = pd.DataFrame(arr)
        return df.rolling(window, center=True, min_periods=1).mean().to_numpy(np.float32)

    obs_plot = _smooth_for_plot(_downsample(observed))
    pred_plot = _smooth_for_plot(_downsample(predicted)) if predicted is not None and len(predicted) > 0 else None

    # Use raw degrees for Lat/Lon; scale 3D box aspect to avoid flattening.
    obs_geo = obs_plot.astype(np.float32)
    pred_geo = pred_plot.astype(np.float32) if pred_plot is not None else None

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    # x=Latitude, y=Longitude, z=Altitude (scaled for display)
    xr = float(np.ptp(obs_geo[:, 0])) or 1e-6
    yr = float(np.ptp(obs_geo[:, 1])) or 1e-6
    zr = float(np.ptp(obs_geo[:, 2])) or 1.0
    xy = max(xr, yr)
    # Stretch lat/lon so the 3D grid is cubic (degree tick labels preserved)
    xy_scale = max(1.0, zr / xy)

    x_obs = obs_geo[:, 0] * xy_scale
    y_obs = obs_geo[:, 1] * xy_scale
    x_pred = pred_geo[:, 0] * xy_scale if pred_geo is not None else None
    y_pred = pred_geo[:, 1] * xy_scale if pred_geo is not None else None
    z_obs = obs_geo[:, 2]
    z_pred = pred_geo[:, 2] if pred_geo is not None else None

    ax.plot(x_obs, y_obs, z_obs, label="Observed (latest)")
    if pred_geo is not None:
        ax.plot(x_pred, y_pred, z_pred, label="Predicted (future)")
    ax.set_title(f"3D Trajectory | {hexident} | seg={seg.seg_id}")
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Longitude (°)")
    ax.set_zlabel("Altitude / m", labelpad=12)
    # Match z visual scale to lat/lon range to avoid flattening.
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v / xy_scale:.2f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v / xy_scale:.2f}"))
    ax.view_init(elev=12, azim=110)
    ax.legend()
    p3 = f"{out_dir}/trajectory3d_{hexident}_seg{seg.seg_id}.png"
    plt.savefig(p3, dpi=140, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {p3}")

def save_compare_plot(
    out_dir: str,
    hexident: str,
    seg: SegInfo,
    gt: np.ndarray,
    preds: dict,
    tag: Optional[str] = None
) -> None:
    # Align scales with cubic grid

    xr = float(np.ptp(gt[:, 0])) or 1e-6
    yr = float(np.ptp(gt[:, 1])) or 1e-6
    zr = float(np.ptp(gt[:, 2])) or 1.0
    xy = max(xr, yr)
    xy_scale = max(1.0, zr / xy)

    def _scale(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return arr[:, 0] * xy_scale, arr[:, 1] * xy_scale, arr[:, 2]

    x_gt, y_gt, z_gt = _scale(gt)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x_gt, y_gt, z_gt, label="Ground truth", linewidth=2.4, color="#111111")

    styles = {
        "Bi-LSTM": dict(color="#17becf", linewidth=1.8, linestyle="--"),
        "Transformer": dict(color="#8c564b", linewidth=1.6, linestyle="--"),
        "SVR": dict(color="#d62728", linewidth=1.6, linestyle="-."),
        "MLP": dict(color="#ff7f0e", linewidth=1.6, linestyle=":"),
        "DecisionTree": dict(color="#2ca02c", linewidth=1.3, linestyle="--"),
        "RandomForest": dict(color="#7f7f7f", linewidth=1.3, linestyle="--"),
        "KNN": dict(color="#9467bd", linewidth=1.3, linestyle=":"),
    }
    for name, arr in preds.items():
        x, y, z = _scale(arr)
        ax.plot(x, y, z, label=name, **styles.get(name, {}))

    title_tag = f" | {tag}" if tag else ""
    ax.set_title(f"3D Trajectory (Test) | {hexident} | seg={seg.seg_id}{title_tag}")
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Longitude (°)")
    ax.set_zlabel("Altitude / m", labelpad=12)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=12, azim=110)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v / xy_scale:.2f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v / xy_scale:.2f}"))
    # Expand limits to include all predictions
    all_arrs = [gt] + list(preds.values())
    all_xyz = np.vstack(all_arrs)
    x_all, y_all, z_all = _scale(all_xyz)
    pad_x = (x_all.max() - x_all.min()) * 0.05 or 1e-6
    pad_y = (y_all.max() - y_all.min()) * 0.05 or 1e-6
    pad_z = (z_all.max() - z_all.min()) * 0.05 or 1.0
    ax.set_xlim(x_all.min() - pad_x, x_all.max() + pad_x)
    ax.set_ylim(y_all.min() - pad_y, y_all.max() + pad_y)
    ax.set_zlim(z_all.min() - pad_z, z_all.max() + pad_z)
    ax.legend()
    tag_suffix = f"_{tag}" if tag else ""
    p = f"{out_dir}/trajectory3d_compare_{hexident}_seg{seg.seg_id}{tag_suffix}.png"
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {p}")


# =========================
# CLI / main
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ADS-B 3D trajectory prediction demo (strict altitude).")
    p.add_argument("--hexident", required=True)
    p.add_argument("--project", default="iitp-class-team-2-473114")
    p.add_argument("--table", default="iitp-class-team-2-473114.SBS_Data.FirstRun")

    p.add_argument("--train-start", default=None, help="YYYY-MM-DD; optional (if omitted, auto lookback)")
    p.add_argument("--train-end", default=None, help="YYYY-MM-DD; optional (if omitted, uses now UTC)")
    p.add_argument("--lookbacks", default="7,30,120,365", help="days to try if train-start omitted")

    p.add_argument("--gap-minutes", type=int, default=10)
    p.add_argument("--min-seg-points", type=int, default=120)
    p.add_argument("--max-train-segs", type=int, default=60)

    p.add_argument("--segment-id", type=int, default=None, help="Pick a specific seg_id from the listed segments")
    p.add_argument("--segment-start", default=None, help='UTC timestamp, e.g. "2025-01-01T12:34:56Z"')
    p.add_argument("--segment-end", default=None, help='UTC timestamp, e.g. "2025-01-01T13:04:56Z"')
    p.add_argument("--segment-duration-min", type=float, default=None, help="Take only first N minutes of the chosen segment")
    p.add_argument("--plot-only", action="store_true", help="Skip prediction; only visualize observed segment")

    p.add_argument("--resample", default="5s")
    p.add_argument("--smooth-w", type=int, default=3)
    p.add_argument("--interp-limit", type=int, default=6)
    p.add_argument("--prep-mode", default="full", choices=["full", "minimal"], help="Legacy preprocessing level")
    p.add_argument(
        "--prep-profile",
        default="full",
        choices=["full", "rs_smooth", "rs_smooth_outlier", "rs_smooth_outlier_interp", "minimal"],
        help="Preprocessing profile for incremental comparison",
    )

    p.add_argument("--window", type=int, default=10)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden-size", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)

    p.add_argument("--horizon-steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out-dir", default=".")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--infer-airports", action="store_true", help="Infer origin/destination airports from segment start/end")
    p.add_argument("--airports-csv", default="data/airports.csv")
    p.add_argument("--airports-source", default="https://ourairports.com/data/airports.csv")
    p.add_argument("--airport-max-km", type=float, default=80.0)
    p.add_argument("--compare-models", action="store_true", help="Plot GT vs multiple model predictions on test split")
    p.add_argument("--compare-train-ratio", type=float, default=0.7)
    p.add_argument("--compare-scale", default="train", choices=["train", "full"])
    p.add_argument("--compare-window", type=int, default=None)
    p.add_argument("--compare-epochs", type=int, default=None)
    p.add_argument("--compare-hidden-size", type=int, default=None)
    p.add_argument("--compare-num-layers", type=int, default=None)
    p.add_argument("--compare-feature-set", default="basic", choices=["basic", "vel"])
    p.add_argument("--compare-tag", default=None)
    p.add_argument("--compare-transformer", action="store_true", help="Include Transformer in compare plots")
    p.add_argument("--compare-models-keep", default="Bi-LSTM,SVR,MLP,Transformer", help="Comma list of models to include")
    p.add_argument("--compare-transformer-d-model", type=int, default=64)
    p.add_argument("--compare-transformer-nhead", type=int, default=4)
    p.add_argument("--compare-transformer-layers", type=int, default=2)
    p.add_argument("--compare-transformer-ff", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    client = bq_client(args.project)

    end_ts = parse_date_utc(args.train_end) if args.train_end else utc_now()
    end_date = date_str(end_ts)

    ranges: List[Tuple[str, str]] = []
    if args.train-start if False else None:
        pass  # placeholder to keep syntax highlighting sane

    if args.train_start:
        ranges = [(args.train_start, end_date)]
    else:
        lookbacks = [int(x.strip()) for x in args.lookbacks.split(",") if x.strip()]
        for days in lookbacks:
            start_ts = end_ts - pd.Timedelta(days=days)
            ranges.append((date_str(start_ts), end_date))

    chosen = None
    df_p = pd.DataFrame()

    def _profile_flags(profile: str) -> Tuple[bool, bool, bool, bool]:
        if profile in ["minimal"]:
            return False, False, False, False
        if profile == "rs_smooth":
            return False, True, False, True
        if profile == "rs_smooth_outlier":
            return True, True, False, True
        if profile in ["rs_smooth_outlier_interp", "full"]:
            return True, True, True, True
        return True, True, True, True

    profile = args.prep_profile or args.prep_mode
    use_outlier_filter, use_resample, use_interp, use_smooth = _profile_flags(profile)

    for start_date, end_date in ranges:
        if not args.quiet:
            print(f"[Load] {args.hexident} | trying {start_date} -> {end_date}")
        df_raw = load_hex_points(client, args.table, args.hexident, start_date, end_date)
        if df_raw.empty:
            if not args.quiet:
                print("  - no rows returned")
            continue

        df_p = preprocess_3d(
            df_raw,
            resample=args.resample,
            smooth_w=args.smooth_w,
            interp_limit=args.interp_limit,
            quiet=args.quiet,
            use_outlier_filter=use_outlier_filter,
            use_resample=use_resample,
            use_interp=use_interp,
            use_smooth=use_smooth,
        )
        if df_p.empty:
            if not args.quiet:
                print("  - rows exist but no usable 3D rows after preprocessing")
            continue

        chosen = (start_date, end_date)
        break

    if df_p.empty or chosen is None:
        raise RuntimeError(
            "No usable 3D data after preprocessing.\n"
            "This usually means Altitude is missing or unusable for this HexIdent in this dataset.\n"
            "Try another HexIdent or verify Altitude column quality (is it always NULL or always 0?)."
        )

    # Segment
    seg_df_all = segment_by_gap(df_p, gap_minutes=args.gap_minutes)
    segs = list_segments(seg_df_all, min_seg_points=args.min_seg_points)
    if not segs:
        raise RuntimeError(
            "Usable 3D rows exist but no segment has enough points.\n"
            "Lower --min-seg-points or lower --gap-minutes."
        )

    latest = segs[0]

    if not args.quiet:
        print(f"\n[Range chosen] {chosen[0]} -> {chosen[1]}")
        print("[Segments] newest first (up to 10):")
        for s in segs[:10]:
            print(f"  seg_id={s.seg_id} | points={s.n} | {s.start_ts} -> {s.end_ts}")

    # Select segment / range
    selected_seg = latest
    if args.segment_id is not None:
        hit = [s for s in segs if s.seg_id == args.segment_id]
        if not hit:
            raise RuntimeError("segment-id not found in listed segments.")
        selected_seg = hit[0]

    if args.segment_start or args.segment_end:
        ts_start = parse_ts_utc(args.segment_start) if args.segment_start else None
        ts_end = parse_ts_utc(args.segment_end) if args.segment_end else None
        sd = seg_df_all.copy()
        if ts_start is not None:
            sd = sd[sd["ts"] >= ts_start]
        if ts_end is not None:
            sd = sd[sd["ts"] <= ts_end]
        if sd.empty:
            raise RuntimeError("No rows in the specified segment-start/segment-end window.")
        selected_seg = SegInfo(
            seg_id=-1,
            n=len(sd),
            start_ts=sd["ts"].min(),
            end_ts=sd["ts"].max(),
        )
        seg_df_view = sd.reset_index(drop=True)
    else:
        seg_df_view = seg_df_all[seg_df_all["seg_id"] == selected_seg.seg_id].reset_index(drop=True)

    if args.segment_duration_min is not None:
        cut_end = selected_seg.start_ts + pd.Timedelta(minutes=args.segment_duration_min)
        seg_df_view = seg_df_view[seg_df_view["ts"] <= cut_end].reset_index(drop=True)
        if seg_df_view.empty:
            raise RuntimeError("segment-duration-min produced empty segment.")
        selected_seg = SegInfo(
            seg_id=selected_seg.seg_id,
            n=len(seg_df_view),
            start_ts=seg_df_view["ts"].min(),
            end_ts=seg_df_view["ts"].max(),
        )

    latest_coords = seg_df_view[["Latitude", "Longitude", "Altitude"]].to_numpy(np.float32)

    if args.infer_airports:
        airports = load_airports(args.airports_csv, args.airports_source, args.quiet)
        lat_start = float(seg_df_view.iloc[0]["Latitude"])
        lon_start = float(seg_df_view.iloc[0]["Longitude"])
        lat_end = float(seg_df_view.iloc[-1]["Latitude"])
        lon_end = float(seg_df_view.iloc[-1]["Longitude"])
        dep, arr, d_dep, d_arr = infer_airport_pair(
            lat_start, lon_start, lat_end, lon_end, airports, args.airport_max_km
        )
        print("\n[Airports] inferred from segment endpoints")
        if dep is None:
            print(f"  Departure: not found within {args.airport_max_km:.0f} km")
        else:
            dep_code = dep["iata_code"] if pd.notna(dep.get("iata_code")) else dep.get("ident")
            print(f"  Departure: {dep_code} | {dep.get('name')} | {d_dep:.1f} km")
        if arr is None:
            print(f"  Arrival: not found within {args.airport_max_km:.0f} km")
        else:
            arr_code = arr["iata_code"] if pd.notna(arr.get("iata_code")) else arr.get("ident")
            print(f"  Arrival: {arr_code} | {arr.get('name')} | {d_arr:.1f} km")

    if args.plot_only:
        if not args.quiet:
            print("[Mode] plot-only (no prediction)")
        print_preview(args.hexident, selected_seg, args.resample, 0, latest_coords, None)
        if not args.no_plots:
            save_plots(args.out_dir, args.hexident, selected_seg, latest_coords, None)
        return

    if args.compare_models:
        if not args.quiet:
            print("[Mode] compare-models (70/30 train/test)")
        n = len(latest_coords)
        window = args.compare_window or args.window
        epochs = args.compare_epochs or args.epochs
        hidden_size = args.compare_hidden_size or args.hidden_size
        num_layers = args.compare_num_layers or 1
        step_seconds = pd.to_timedelta(args.resample.replace("S", "s")).total_seconds()
        split = int(n * args.compare_train_ratio)
        split = max(split, window + 5)

        gt, preds = fit_predict_baselines(
            latest_coords,
            window=window,
            train_ratio=args.compare_train_ratio,
            teacher_forcing=True,
            scale_mode=args.compare_scale,
            feature_set=args.compare_feature_set,
            step_seconds=step_seconds,
        )
        bilstm_pred = fit_predict_bilstm(
            latest_coords,
            window=window,
            epochs=epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            train_ratio=args.compare_train_ratio,
            teacher_forcing=True,
            scale_mode=args.compare_scale,
            hidden_size=hidden_size,
            num_layers=num_layers,
            feature_set=args.compare_feature_set,
            step_seconds=step_seconds,
        )
        preds = {"Bi-LSTM": bilstm_pred, **preds}
        if args.compare_transformer:
            t_pred = fit_predict_transformer(
                latest_coords,
                window=window,
                epochs=epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                train_ratio=args.compare_train_ratio,
                teacher_forcing=True,
                scale_mode=args.compare_scale,
                feature_set=args.compare_feature_set,
                step_seconds=step_seconds,
                d_model=args.compare_transformer_d_model,
                nhead=args.compare_transformer_nhead,
                num_layers=args.compare_transformer_layers,
                dim_feedforward=args.compare_transformer_ff,
            )
            preds = {"Transformer": t_pred, **preds}

        keep = [s.strip() for s in args.compare_models_keep.split(",") if s.strip()]
        preds = {k: v for k, v in preds.items() if k in keep}

        metrics = {}
        for name, arr in preds.items():
            metrics[name] = compute_metrics_per_dim(gt, arr)
        print("\n[Metrics] per-dimension (MSE / MAE / MAPE)")
        for name, m in metrics.items():
            lat = m["lat"]; lon = m["lon"]; alt = m["alt"]
            print(f"  {name:12s} | lat {lat['MSE']:.6f}/{lat['MAE']:.6f}/{lat['MAPE']:.6f} | "
                  f"lon {lon['MSE']:.6f}/{lon['MAE']:.6f}/{lon['MAPE']:.6f} | "
                  f"alt {alt['MSE']:.6f}/{alt['MAE']:.6f}/{alt['MAPE']:.6f}")
        if not args.no_plots:
            save_compare_plot(
                args.out_dir,
                args.hexident,
                selected_seg,
                gt,
                preds,
                tag=args.compare_tag,
            )
        return

    # Training segments: prefer past; fallback to latest itself (front part)
    train_segments: List[np.ndarray] = []
    base_seg_id = selected_seg.seg_id if selected_seg.seg_id >= 0 else None
    past_ids = [s.seg_id for s in segs if base_seg_id is None or s.seg_id != base_seg_id]
    past_ids = past_ids[: args.max_train_segs]
    for sid in past_ids:
        coords = seg_df_all[seg_df_all["seg_id"] == sid][["Latitude", "Longitude", "Altitude"]].to_numpy(np.float32)
        if len(coords) >= (args.window + 10):
            train_segments.append(coords)

    if not train_segments:
        if not args.quiet:
            print("[TrainData] No past segments usable. Training on latest segment's first 70%.")
        cut = int(0.7 * len(latest_coords))
        cut = max(cut, args.window + 10)
        train_segments = [latest_coords[:cut]]

    # Build scaler & windows
    scaler = MinMaxScaler()
    scaler.fit(np.vstack(train_segments))

    X_list, y_list = [], []
    for coords in train_segments:
        coords_s = scaler.transform(coords).astype(np.float32)
        X, y = make_windows(coords_s, args.window)
        if len(X) > 0:
            X_list.append(X); y_list.append(y)

    X_train = np.vstack(X_list)
    y_train = np.vstack(y_list)

    if len(X_train) < 128:
        raise RuntimeError(
            "Not enough training windows.\n"
            "Try lowering --window, lowering --min-seg-points, or using a longer date range/lookback."
        )

    loader = DataLoader(SeqDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    if not args.quiet:
        print(f"\n[Train] device={device} | windows={len(X_train):,} | resample={args.resample} | window={args.window}")

    model = BiLSTM(input_size=3, hidden_size=args.hidden_size, out_dim=3).to(device)
    train_model(model, loader, device=device, epochs=args.epochs, lr=args.lr)

    # Predict future from last WINDOW
    if len(latest_coords) < args.window:
        raise RuntimeError("Latest segment too short to seed prediction. Lower --window or widen date range/lookback.")

    latest_scaled = scaler.transform(latest_coords).astype(np.float32)
    seed = latest_scaled[-args.window:]
    pred_scaled = rolling_predict_scaled(model, seed, args.horizon_steps, device)
    pred = scaler.inverse_transform(pred_scaled)

    print_preview(args.hexident, selected_seg, args.resample, args.horizon_steps, latest_coords, pred)

    if not args.no_plots:
        save_plots(args.out_dir, args.hexident, selected_seg, latest_coords, pred)


if __name__ == "__main__":
    main()