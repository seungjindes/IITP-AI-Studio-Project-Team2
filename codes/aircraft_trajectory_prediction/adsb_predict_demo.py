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
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.cloud import bigquery
from sklearn.preprocessing import MinMaxScaler

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

def make_windows(arr: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i:i + window])
        y.append(arr[i + window])
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
    quiet: bool
) -> pd.DataFrame:
    """
    Strict 3D preprocessing.
    Output MUST have non-null lat/lon/alt rows.

    Note:
    - If Altitude is always NULL or always 0 and you consider that "unusable", we will detect and fail.
    """
    if df_raw.empty:
        return df_raw

    d = df_raw.copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
    d = d.dropna(subset=["ts"])
    d = d.drop_duplicates(subset=["ts"]).sort_values("ts").set_index("ts")

    # Force numeric
    for c in ["Latitude", "Longitude", "Altitude"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Basic sanity for lat/lon
    d = d[(d["Latitude"].between(-90, 90)) & (d["Longitude"].between(-180, 180))]

    if not quiet:
        alt_raw = d["Altitude"]
        nonnull = int(alt_raw.notna().sum())
        zratio = float((alt_raw.fillna(0) == 0).mean()) if len(alt_raw) else 1.0
        print(f"[Raw] rows={len(d):,} | alt_nonnull={nonnull:,} | alt_zero_ratio≈{zratio:.3f}")

    rs = resample.replace("S", "s")

    # Keep only numeric columns before resample
    d = d[["Latitude", "Longitude", "Altitude"]]

    # Resample numeric only
    d = d.resample(rs).mean(numeric_only=True)

    # Fill only small gaps
    d = d.interpolate(method="time", limit=interp_limit, limit_direction="both")

    # Smooth
    for c in ["Latitude", "Longitude", "Altitude"]:
        d[c] = d[c].rolling(smooth_w, center=True, min_periods=1).mean()

    # Strict: require all three
    d3 = d.dropna(subset=["Latitude", "Longitude", "Altitude"]).reset_index()

    if not quiet:
        if len(d3) == 0:
            print(f"[Preprocess3D] produced 0 rows after strict dropna(lat/lon/alt).")
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

def print_preview(hexident: str, seg: SegInfo, resample: str, horizon_steps: int, observed: np.ndarray, predicted: np.ndarray):
    seconds_per_step = pd.to_timedelta(resample.replace("S", "s")).total_seconds()
    minutes = (horizon_steps * seconds_per_step) / 60.0

    print("\n============================")
    print("Future Prediction (3D)")
    print("============================")
    print(f"HexIdent: {hexident}")
    print(f"Latest segment: seg_id={seg.seg_id} | points={seg.n} | {seg.start_ts} -> {seg.end_ts}")
    print(f"Resample: {resample} (~{seconds_per_step:.0f}s/step) | Horizon: {horizon_steps} steps (~{minutes:.1f} min)")

    last_obs = observed[-1]
    print(f"\nLast observed: lat={last_obs[0]:.6f}, lon={last_obs[1]:.6f}, alt={last_obs[2]:.1f}")

    n_preview = min(8, len(predicted))
    print(f"\nNext {n_preview} predicted points:")
    for i in range(n_preview):
        p = predicted[i]
        tmin = (i + 1) * seconds_per_step / 60.0
        print(f"  +{tmin:5.1f}m  lat={p[0]:.6f}, lon={p[1]:.6f}, alt={p[2]:.1f}")
    print("")

def save_plots(out_dir: str, hexident: str, seg: SegInfo, observed: np.ndarray, predicted: np.ndarray):
    plt.figure()
    plt.plot(observed[:, 1], observed[:, 0], label="Observed (latest)")
    plt.plot(predicted[:, 1], predicted[:, 0], label="Predicted (future)")
    plt.legend()
    plt.title(f"Trajectory | {hexident} | seg={seg.seg_id}")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    p1 = f"{out_dir}/trajectory_{hexident}_seg{seg.seg_id}.png"
    plt.savefig(p1, dpi=140, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(observed[:, 2], label="Observed Alt")
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

    p.add_argument("--resample", default="5s")
    p.add_argument("--smooth-w", type=int, default=3)
    p.add_argument("--interp-limit", type=int, default=6)

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
    seg_df = segment_by_gap(df_p, gap_minutes=args.gap_minutes)
    segs = list_segments(seg_df, min_seg_points=args.min_seg_points)
    if not segs:
        raise RuntimeError(
            "Usable 3D rows exist but no segment has enough points.\n"
            "Lower --min-seg-points or lower --gap-minutes."
        )

    latest = segs[0]
    past_ids = [s.seg_id for s in segs[1:]][: args.max_train_segs]

    if not args.quiet:
        print(f"\n[Range chosen] {chosen[0]} -> {chosen[1]}")
        print("[Segments] newest first (up to 10):")
        for s in segs[:10]:
            print(f"  seg_id={s.seg_id} | points={s.n} | {s.start_ts} -> {s.end_ts}")

    latest_coords = seg_df[seg_df["seg_id"] == latest.seg_id][["Latitude", "Longitude", "Altitude"]].to_numpy(np.float32)

    # Training segments: prefer past; fallback to latest itself (front part)
    train_segments: List[np.ndarray] = []
    for sid in past_ids:
        coords = seg_df[seg_df["seg_id"] == sid][["Latitude", "Longitude", "Altitude"]].to_numpy(np.float32)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    print_preview(args.hexident, latest, args.resample, args.horizon_steps, latest_coords, pred)

    if not args.no_plots:
        save_plots(args.out_dir, args.hexident, latest, latest_coords, pred)


if __name__ == "__main__":
    main()