"""
End-to-end ADS-B Trajectory Prediction (Bi-LSTM) using BigQuery

What this script does (A + B combined):
A) Finds "good" FlightID candidates automatically (optional) based on point count + big-gap ratio.
B) Trains a Bi-LSTM on the first 70% of a selected trajectory, then simulates ADS-B failure by
   withholding the last 30% and performing autoregressive (rolling) prediction to reconstruct it.

Install:
  pip install google-cloud-bigquery pandas numpy scikit-learn torch matplotlib

Auth (one of):
  - export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
  - gcloud auth application-default login
"""

from __future__ import annotations

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


# =========================
# Config
# =========================

PROJECT = "iitp-class-team-2-473114"
TABLE = "iitp-class-team-2-473114.SBS_Data.FirstRun"

# Option 1: Use a specific HexIdent and auto-pick best FlightID for that HexIdent
# HEXIDENT = "71C700"
HEXIDENT = None

# Option 2: Force a specific FlightID (recommended once you find a good one)
FORCE_FLIGHT_ID: Optional[int] = None  # e.g., 123456

# Date filter for candidate search (A). Keep narrow for cost control.
CANDIDATE_START_DATE = "2025-12-01"   # YYYY-MM-DD
CANDIDATE_END_DATE = "2025-12-31"     # YYYY-MM-DD

# "Big gap" threshold in seconds when scoring continuity
GAP_SEC_THRESHOLD = 30

# Minimum quality filters
MIN_POINTS = 800
MIN_DURATION_SEC = 900
MAX_BIG_GAP_RATIO = 0.05
TOP_K_CANDIDATES = 20

# Preprocessing / modeling
RESAMPLE = "5S"     # e.g., "1S", "5S", "10S"
SMOOTH_W = 3        # moving average window
WINDOW = 10         # sequence length
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
HIDDEN_SIZE = 80
NUM_LAYERS = 1
SEED = 42


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
    hexident: Optional[str] = None,
) -> List[CandidateFlight]:
    hex_filter = f"AND HexIdent = '{hexident}'" if hexident else ""

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
        Date_MSG_Logged BETWEEN start_date AND end_date
        AND Latitude IS NOT NULL AND Longitude IS NOT NULL AND Altitude IS NOT NULL AND Altitude > 0
        AND FlightID IS NOT NULL
        {hex_filter}
    ),
    ordered AS (
      SELECT
        HexIdent, FlightID, ts,
        TIMESTAMP_DIFF(ts, LAG(ts) OVER(PARTITION BY HexIdent, FlightID ORDER BY ts), SECOND) AS dt_sec
      FROM base
    ),
    stats AS (
      SELECT
        HexIdent,
        FlightID,
        COUNT(*) AS n_points,
        TIMESTAMP_DIFF(MAX(ts), MIN(ts), SECOND) AS duration_sec,
        SUM(CASE WHEN dt_sec IS NULL THEN 0 WHEN dt_sec > gap_sec_threshold THEN 1 ELSE 0 END) AS n_big_gaps,
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
) -> pd.DataFrame:
    query = f"""
    WITH base AS (
      SELECT
        HexIdent,
        FlightID,
        Callsign,
        TIMESTAMP(DATETIME(Date_MSG_Logged, Time_MSG_Logged), "UTC") AS ts,
        Latitude, Longitude, Altitude,
        GroundSpeed, Track, VerticalRate, IsOnGround
      FROM `{table}`
      WHERE Latitude IS NOT NULL AND Longitude IS NOT NULL AND Altitude IS NOT NULL
    )
    SELECT *
    FROM base
    WHERE FlightID = {flight_id}
    ORDER BY ts;
    """
    df = client.query(query).to_dataframe()
    return df


def preprocess_trajectory(
    df: pd.DataFrame,
    resample: str,
    smooth_w: int,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Empty dataframe: no trajectory data returned from BigQuery.")

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").set_index("ts")

    # Basic outlier filters
    df = df[
        (df["Latitude"].between(-90, 90)) &
        (df["Longitude"].between(-180, 180)) &
        (df["Altitude"] > 0)
    ]

    # Resample to fixed interval
    df = df.resample(resample).mean(numeric_only=True)

    # Interpolate (time-based)
    df[["Latitude", "Longitude", "Altitude"]] = df[["Latitude", "Longitude", "Altitude"]].interpolate(method="time")

    # Smooth
    for c in ["Latitude", "Longitude", "Altitude"]:
        df[c] = df[c].rolling(smooth_w, center=True).mean()

    df = df.dropna(subset=["Latitude", "Longitude", "Altitude"]).reset_index()
    return df


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
    def __init__(self, input_size: int = 3, hidden_size: int = 80, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
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
    """
    Simulate ADS-B failure:
      - observe only first cutoff_ratio of trajectory
      - predict the remaining steps autoregressively
    Returns:
      pred_scaled (T-cut, 3), true_scaled (T-cut, 3)
    """
    T = len(coords_scaled)
    cut = int(cutoff_ratio * T)
    seed_start = cut - window
    if seed_start < 0:
        raise ValueError("Not enough points to seed the rolling window. Reduce WINDOW or use longer trajectory.")

    window_buf = coords_scaled[seed_start:cut].copy()  # (window, 3)
    steps = T - cut
    preds_scaled = []

    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(window_buf[None, :, :], dtype=torch.float32).to(device)
            yhat = model(x).cpu().numpy()[0]  # (3,)
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


def plot_results(true_coords: np.ndarray, pred_coords: np.ndarray, title_suffix: str) -> None:
    # Lat/Lon plot (x=lon, y=lat)
    plt.figure()
    plt.plot(true_coords[:, 1], true_coords[:, 0], label="True (withheld)")
    plt.plot(pred_coords[:, 1], pred_coords[:, 0], label="Pred (rolling)")
    plt.legend()
    plt.title(f"Trajectory (Lat/Lon) - {title_suffix}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

    # Altitude plot
    plt.figure()
    plt.plot(true_coords[:, 2], label="True Alt (withheld)")
    plt.plot(pred_coords[:, 2], label="Pred Alt (rolling)")
    plt.legend()
    plt.title(f"Altitude - {title_suffix}")
    plt.xlabel("Step")
    plt.ylabel("Altitude")
    plt.show()


# =========================
# Main
# =========================

def main() -> None:
    set_seed(SEED)

    client = bigquery.Client(project=PROJECT)

    # ----- A) Select a good FlightID -----
    if FORCE_FLIGHT_ID is not None:
        chosen_flight_id = FORCE_FLIGHT_ID
        print(f"[A] Using FORCE_FLIGHT_ID={chosen_flight_id}")
    else:
        print("[A] Searching for high-quality FlightID candidates...")
        candidates = query_candidate_flights(
            client=client,
            table=TABLE,
            start_date=CANDIDATE_START_DATE,
            end_date=CANDIDATE_END_DATE,
            gap_sec_threshold=GAP_SEC_THRESHOLD,
            min_points=MIN_POINTS,
            min_duration_sec=MIN_DURATION_SEC,
            max_big_gap_ratio=MAX_BIG_GAP_RATIO,
            top_k=TOP_K_CANDIDATES,
            hexident=HEXIDENT,  # restrict to your aircraft; set to None to search all
        )
        if not candidates:
            raise RuntimeError(
                "No candidates found. Try widening date range, lowering MIN_POINTS/MIN_DURATION_SEC, "
                "or setting hexident=None."
            )

        print("[A] Top candidates:")
        for i, c in enumerate(candidates[:10], 1):
            print(
                f"  {i:02d}. HexIdent={c.hexident} FlightID={c.flight_id} "
                f"points={c.n_points} dur={c.duration_sec}s gap_ratio={c.big_gap_ratio:.3f} "
                f"score={c.quality_score:.3f}"
            )

        chosen = candidates[0]
        chosen_flight_id = chosen.flight_id
        print(f"[A] Chosen FlightID={chosen_flight_id} (HexIdent={chosen.hexident})")

    # ----- Load trajectory for chosen FlightID -----
    print("[Load] Querying trajectory...")
    df_raw = query_flight_trajectory(client, TABLE, chosen_flight_id)
    print(f"[Load] Rows returned: {len(df_raw):,}")

    # ----- Preprocess -----
    print("[Preprocess] Cleaning/resampling/interpolating/smoothing...")
    df = preprocess_trajectory(df_raw, resample=RESAMPLE, smooth_w=SMOOTH_W)
    print(f"[Preprocess] Rows after preprocess: {len(df):,}")

    if len(df) < (WINDOW + 50):
        raise RuntimeError("Trajectory too short after preprocessing. Try smaller WINDOW or different FlightID.")

    # ----- Prepare data -----
    coords = df[["Latitude", "Longitude", "Altitude"]].to_numpy(dtype=np.float32)
    scaler = MinMaxScaler()
    coords_scaled = scaler.fit_transform(coords).astype(np.float32)

    X, y = make_windows(coords_scaled, window=WINDOW)
    split = int(0.7 * len(X))
    X_train, y_train = X[:split], y[:split]
    print(f"[Data] Train windows: {len(X_train):,} | Total windows: {len(X):,}")

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    # ----- Train model -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] device={device}")
    model = BiLSTM(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    train_model(model, train_loader, device=device, epochs=EPOCHS, lr=LR)

    # ----- B) Autoregressive rolling prediction after ADS-B failure -----
    print("[Predict] Autoregressive rollout on withheld last 30%...")
    pred_scaled, true_scaled = autoregressive_rollout(
        model=model,
        coords_scaled=coords_scaled,
        window=WINDOW,
        device=device,
        cutoff_ratio=0.7,
    )

    pred_coords = scaler.inverse_transform(pred_scaled)
    true_coords = scaler.inverse_transform(true_scaled)

    mse, mae, mape = compute_metrics(pred_coords, true_coords)
    print("[Metrics] (lat, lon, alt)")
    print("  MSE :", mse)
    print("  MAE :", mae)
    print("  MAPE%:", mape)

    # ----- Visualize -----
    title_suffix = f"FlightID={chosen_flight_id} | RESAMPLE={RESAMPLE} | WINDOW={WINDOW}"
    plot_results(true_coords, pred_coords, title_suffix=title_suffix)


if __name__ == "__main__":
    main()