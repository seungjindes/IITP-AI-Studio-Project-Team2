#!/usr/bin/env python3
"""
End-to-end ADS-B Trajectory Prediction (Bi-LSTM) using BigQuery
- CLI: --hexident required
- Optional: --flight-id to force a specific flight
- 2D fallback when altitude is mostly 0/unusable

A) If --flight-id is not provided:
   Finds "good" (HexIdent, FlightID) candidates for the given HexIdent.
B) Loads ONE chosen trajectory, preprocesses it, then:
   - If altitude mostly 0 / constant -> 2D prediction (lat, lon)
   - Else -> 3D prediction (lat, lon, alt)
   Train on first 70% and autoregressive-rollout last 30%.

Install:
  pip install google-cloud-bigquery pandas numpy scikit-learn torch matplotlib

Run:
  python adsb_predict_demo.py --hexident 424A71
  python adsb_predict_demo.py --hexident 424A71 --flight-id 12345
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.cloud import bigquery
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os


# =========================
# Defaults (can override via CLI)
# =========================

PROJECT_DEFAULT = "iitp-class-team-2-473114"
TABLE_DEFAULT   = "iitp-class-team-2-473114.SBS_Data.FirstRun"

CANDIDATE_START_DEFAULT = "2025-12-01"
CANDIDATE_END_DEFAULT   = "2025-12-31"

GAP_SEC_THRESHOLD_DEFAULT = 30
MIN_POINTS_DEFAULT = 800
MIN_DURATION_SEC_DEFAULT = 900
MAX_BIG_GAP_RATIO_DEFAULT = 0.05
TOP_K_CANDIDATES_DEFAULT = 20

RESAMPLE_DEFAULT = "5S"
SMOOTH_W_DEFAULT = 3
WINDOW_DEFAULT = 10
BATCH_SIZE_DEFAULT = 256
EPOCHS_DEFAULT = 10
LR_DEFAULT = 1e-3
HIDDEN_SIZE_DEFAULT = 80
NUM_LAYERS_DEFAULT = 1
SEED_DEFAULT = 42

ALT_POSITIVE_MIN_RATIO_DEFAULT = 0.05
ALT_SPAN_MIN_DEFAULT = 1e-3


# =========================
# Utilities
# =========================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class CandidateFlight:
    hexident: str
    flight_id: int
    n_points: int
    duration_sec: int
    big_gap_ratio: float
    quality_score: float
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp


def query_candidate_flights(
    client: bigquery.Client,
    table: str,
    start_date: str,
    end_date: str,
    gap_sec_threshold: int,
    min_points: int,
    min_duration_sec: int,
    max_big_gap_ratio: float,
    top_k: int,
    hexident: str,
) -> List[CandidateFlight]:
    """
    Candidate search for ONE hexident.

    Fixes:
    - dt computed per (HexIdent, FlightID)
    - require Altitude > 0 at candidate stage to avoid all-zero altitude candidates
    """
    query = f"""
    -- Standard SQL
    DECLARE start_date DATE DEFAULT DATE('{start_date}');
    DECLARE end_date   DATE DEFAULT DATE('{end_date}');
    DECLARE gap_sec_threshold INT64 DEFAULT {gap_sec_threshold};

    WITH base AS (
      SELECT
        HexIdent,
        FlightID,
        TIMESTAMP(DATETIME(Date_MSG_Logged, Time_MSG_Logged), "UTC") AS ts,
        Latitude, Longitude, Altitude
      FROM `{table}`
      WHERE
        HexIdent = '{hexident}'
        AND Date_MSG_Logged BETWEEN start_date AND end_date
        AND Latitude IS NOT NULL AND Longitude IS NOT NULL
        AND Altitude IS NOT NULL AND Altitude > 0
        AND FlightID IS NOT NULL
    ),
    ordered AS (
      SELECT
        HexIdent, FlightID, ts,
        TIMESTAMP_DIFF(
          ts,
          LAG(ts) OVER(PARTITION BY HexIdent, FlightID ORDER BY ts),
          SECOND
        ) AS dt_sec
      FROM base
    ),
    stats AS (
      SELECT
        HexIdent,
        FlightID,
        COUNT(*) AS n_points,
        TIMESTAMP_DIFF(MAX(ts), MIN(ts), SECOND) AS duration_sec,
        SAFE_DIVIDE(
          SUM(CASE WHEN dt_sec IS NULL THEN 0 WHEN dt_sec > gap_sec_threshold THEN 1 ELSE 0 END),
          GREATEST(COUNT(*) - 1, 1)
        ) AS big_gap_ratio,
        MIN(ts) AS start_ts,
        MAX(ts) AS end_ts
      FROM ordered
      GROUP BY HexIdent, FlightID
    )
    SELECT
      HexIdent, FlightID,
      n_points, duration_sec, big_gap_ratio, start_ts, end_ts,
      (LOG(1 + n_points) * (1 - big_gap_ratio) * LOG(1 + duration_sec)) AS quality_score
    FROM stats
    WHERE
      n_points >= {min_points}
      AND duration_sec >= {min_duration_sec}
      AND big_gap_ratio <= {max_big_gap_ratio}
    ORDER BY quality_score DESC
    LIMIT {top_k};
    """

    df = client.query(query).to_dataframe()
    if df.empty:
        return []

    out: List[CandidateFlight] = []
    for _, r in df.iterrows():
        out.append(
            CandidateFlight(
                hexident=str(r["HexIdent"]),
                flight_id=int(r["FlightID"]),
                n_points=int(r["n_points"]),
                duration_sec=int(r["duration_sec"]),
                big_gap_ratio=float(r["big_gap_ratio"]),
                quality_score=float(r["quality_score"]),
                start_ts=pd.to_datetime(r["start_ts"], utc=True),
                end_ts=pd.to_datetime(r["end_ts"], utc=True),
            )
        )
    return out


def query_flight_trajectory(
    client: bigquery.Client,
    table: str,
    flight_id: int,
    hexident: str,
) -> pd.DataFrame:
    """
    MUST filter by HexIdent AND FlightID (FlightID can be 0/shared).
    """
    query = f"""
    SELECT
      TIMESTAMP(DATETIME(Date_MSG_Logged, Time_MSG_Logged), "UTC") AS ts,
      Latitude, Longitude, Altitude
    FROM `{table}`
    WHERE
      HexIdent = '{hexident}'
      AND FlightID = {flight_id}
      AND Latitude IS NOT NULL AND Longitude IS NOT NULL
      AND Altitude IS NOT NULL
    ORDER BY ts;
    """
    return client.query(query).to_dataframe()


def preprocess_trajectory(df: pd.DataFrame, resample: str, smooth_w: int) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Empty dataframe: no trajectory data returned from BigQuery.")

    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
    d = d.dropna(subset=["ts"]).sort_values("ts").set_index("ts")

    for c in ["Latitude", "Longitude", "Altitude"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d[["Latitude", "Longitude", "Altitude"]]
    d = d.groupby(level=0).mean(numeric_only=True)

    d = d[(d["Latitude"].between(-90, 90)) & (d["Longitude"].between(-180, 180))]

    rs = resample.replace("S", "s")
    d = d.resample(rs).mean(numeric_only=True)

    d[["Latitude", "Longitude"]] = d[["Latitude", "Longitude"]].interpolate(method="time", limit_direction="both")
    d["Altitude"] = d["Altitude"].interpolate(method="time", limit_direction="both")

    for c in ["Latitude", "Longitude", "Altitude"]:
        d[c] = d[c].rolling(smooth_w, center=True, min_periods=1).mean()

    d = d.dropna(subset=["Latitude", "Longitude"]).reset_index()
    return d


def choose_dims(df: pd.DataFrame, alt_positive_min_ratio: float, alt_span_min: float) -> int:
    alt = pd.to_numeric(df["Altitude"], errors="coerce")
    if alt.notna().sum() == 0:
        return 2

    pos_ratio = float((alt.fillna(0) > 0).mean())
    span = float(alt.max() - alt.min()) if alt.notna().sum() else 0.0

    if pos_ratio < alt_positive_min_ratio or span < alt_span_min:
        return 2
    return 3


def make_windows(coords_scaled: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(coords_scaled) - window):
        X.append(coords_scaled[i:i + window])
        y.append(coords_scaled[i + window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self) -> int:
        return len(self.X)
    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 80, num_layers: int = 1, out_dim: int = 3):
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


def train_model(model: nn.Module, train_loader: DataLoader, device: str, epochs: int, lr: float) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            n += len(xb)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | train MSE: {total / max(n,1):.6f}")


def autoregressive_rollout(
    model: nn.Module,
    coords_scaled: np.ndarray,
    window: int,
    device: str,
    cutoff_ratio: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    T = len(coords_scaled)
    cut = int(cutoff_ratio * T)
    seed_start = cut - window
    if seed_start < 0:
        raise ValueError("Not enough points to seed the rolling window. Reduce WINDOW or use longer trajectory.")

    window_buf = coords_scaled[seed_start:cut].copy()
    steps = T - cut
    preds_scaled = []

    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(window_buf[None, :, :], dtype=torch.float32).to(device)
            yhat = model(x).cpu().numpy()[0]
            preds_scaled.append(yhat)
            window_buf = np.vstack([window_buf[1:], yhat])

    preds_scaled = np.array(preds_scaled, dtype=np.float32)
    true_scaled = coords_scaled[cut:cut + steps]
    return preds_scaled, true_scaled


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mse = np.mean((pred - true) ** 2, axis=0)
    mae = np.mean(np.abs(pred - true), axis=0)
    mape = np.mean(np.abs((pred - true) / (true + 1e-6)), axis=0) * 100
    return mse, mae, mape


def plot_results(true_coords: np.ndarray, pred_coords: np.ndarray, title_suffix: str, out_dir: str, no_show: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)

    safe_name = (
        title_suffix.replace(" ", "_")
        .replace("=", "-")
        .replace("|", "_")
        .replace("/", "_")
        .replace(":", "_")
    )

    # Lat/Lon plot
    plt.figure()
    plt.plot(true_coords[:, 1], true_coords[:, 0], label="True (withheld)")
    plt.plot(pred_coords[:, 1], pred_coords[:, 0], label="Pred (rolling)")
    plt.legend()
    plt.title(f"Trajectory (Lat/Lon) - {title_suffix}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    traj_path = os.path.join(out_dir, f"trajectory_{safe_name}.png")
    plt.savefig(traj_path, dpi=150, bbox_inches="tight")
    if not no_show:
        plt.show()
    plt.close()
    print(f"[Saved] {traj_path}")

    # Altitude plot (only if 3D)
    if true_coords.shape[1] == 3:
        plt.figure()
        plt.plot(true_coords[:, 2], label="True Alt (withheld)")
        plt.plot(pred_coords[:, 2], label="Pred Alt (rolling)")
        plt.legend()
        plt.title(f"Altitude - {title_suffix}")
        plt.xlabel("Step")
        plt.ylabel("Altitude")

        alt_path = os.path.join(out_dir, f"altitude_{safe_name}.png")
        plt.savefig(alt_path, dpi=150, bbox_inches="tight")
        if not no_show:
            plt.show()
        plt.close()
        print(f"[Saved] {alt_path}")

# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hexident", required=True, help="HexIdent to run (required)")
    p.add_argument("--flight-id", type=int, default=None, help="Optional: force a specific FlightID")
    p.add_argument("--project", default=PROJECT_DEFAULT)
    p.add_argument("--table", default=TABLE_DEFAULT)

    p.add_argument("--candidate-start", default=CANDIDATE_START_DEFAULT)
    p.add_argument("--candidate-end", default=CANDIDATE_END_DEFAULT)

    p.add_argument("--resample", default=RESAMPLE_DEFAULT)
    p.add_argument("--smooth-w", type=int, default=SMOOTH_W_DEFAULT)
    p.add_argument("--window", type=int, default=WINDOW_DEFAULT)

    p.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    p.add_argument("--lr", type=float, default=LR_DEFAULT)
    p.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE_DEFAULT)
    p.add_argument("--num-layers", type=int, default=NUM_LAYERS_DEFAULT)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)

    p.add_argument("--min-points", type=int, default=MIN_POINTS_DEFAULT)
    p.add_argument("--min-duration-sec", type=int, default=MIN_DURATION_SEC_DEFAULT)
    p.add_argument("--gap-sec-threshold", type=int, default=GAP_SEC_THRESHOLD_DEFAULT)
    p.add_argument("--max-gap-ratio", type=float, default=MAX_BIG_GAP_RATIO_DEFAULT)
    p.add_argument("--top-k", type=int, default=TOP_K_CANDIDATES_DEFAULT)

    p.add_argument("--alt-pos-min-ratio", type=float, default=ALT_POSITIVE_MIN_RATIO_DEFAULT)
    p.add_argument("--alt-span-min", type=float, default=ALT_SPAN_MIN_DEFAULT)
    p.add_argument("--out-dir", default=".", help="Directory to save plots")
    p.add_argument("--no-show", action="store_true", help="Do not call plt.show(); only save files")
    return p.parse_args()


# =========================
# Main
# =========================

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    client = bigquery.Client(project=args.project)

    chosen_hexident = args.hexident

    # ----- Choose flight -----
    if args.flight_id is not None:
        chosen_flight_id = args.flight_id
        print(f"[A] Forced: HexIdent={chosen_hexident} FlightID={chosen_flight_id}")
    else:
        print(f"[A] Searching candidates for HexIdent={chosen_hexident} ...")
        candidates = query_candidate_flights(
            client=client,
            table=args.table,
            start_date=args.candidate_start,
            end_date=args.candidate_end,
            gap_sec_threshold=args.gap_sec_threshold,
            min_points=args.min_points,
            min_duration_sec=args.min_duration_sec,
            max_big_gap_ratio=args.max_gap_ratio,
            top_k=args.top_k,
            hexident=chosen_hexident,
        )
        if not candidates:
            raise RuntimeError("No candidates found for this hexident in that date range. Widen dates or lower thresholds.")

        print("[A] Top candidates:")
        for i, c in enumerate(candidates[:10], 1):
            print(f"  {i:02d}. FlightID={c.flight_id} points={c.n_points} dur={c.duration_sec}s gap={c.big_gap_ratio:.3f} score={c.quality_score:.3f}")

        chosen_flight_id = candidates[0].flight_id
        print(f"[A] Chosen FlightID={chosen_flight_id}")

    # ----- Load -----
    print("[Load] Querying trajectory...")
    df_raw = query_flight_trajectory(client, args.table, chosen_flight_id, chosen_hexident)
    print(f"[Load] Rows returned: {len(df_raw):,}")

    # ----- Preprocess -----
    print("[Preprocess] Cleaning/resampling/interpolating/smoothing...")
    df = preprocess_trajectory(df_raw, resample=args.resample, smooth_w=args.smooth_w)
    print(f"[Preprocess] Rows after preprocess: {len(df):,}")

    if len(df) < (args.window + 50):
        raise RuntimeError("Trajectory too short after preprocessing. Try smaller window or different flight.")

    # ----- 2D/3D decision -----
    dims = choose_dims(df, alt_positive_min_ratio=args.alt_pos_min_ratio, alt_span_min=args.alt_span_min)
    cols = ["Latitude", "Longitude"] if dims == 2 else ["Latitude", "Longitude", "Altitude"]
    print(f"[Mode] {'3D' if dims==3 else '2D fallback'} | cols={cols}")

    coords = df[cols].to_numpy(np.float32)
    scaler = MinMaxScaler()
    coords_scaled = scaler.fit_transform(coords).astype(np.float32)

    X, y = make_windows(coords_scaled, window=args.window)
    split = int(0.7 * len(X))
    X_train, y_train = X[:split], y[:split]
    print(f"[Data] Train windows: {len(X_train):,} | Total windows: {len(X):,}")

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)

    # ----- Train -----
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] device={device}")
    model = BiLSTM(input_size=dims, hidden_size=args.hidden_size, num_layers=args.num_layers, out_dim=dims).to(device)
    train_model(model, train_loader, device=device, epochs=args.epochs, lr=args.lr)

    # ----- Rollout -----
    print("[Predict] Autoregressive rollout on withheld last 30%...")
    pred_scaled, true_scaled = autoregressive_rollout(model, coords_scaled, args.window, device, cutoff_ratio=0.7)

    pred = scaler.inverse_transform(pred_scaled)
    true = scaler.inverse_transform(true_scaled)

    mse, mae, mape = compute_metrics(pred, true)
    print(f"[Metrics] dims={dims}")
    print("  MSE :", mse)
    print("  MAE :", mae)
    print("  MAPE%:", mape)

    title = f"HexIdent={chosen_hexident} FlightID={chosen_flight_id} | dims={dims} | resample={args.resample} | window={args.window}"
    plot_results(true, pred, title_suffix=title, out_dir=args.out_dir, no_show=args.no_show)

if __name__ == "__main__":
    main()