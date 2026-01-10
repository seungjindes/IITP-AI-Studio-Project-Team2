# pip install google-cloud-bigquery pandas numpy scikit-learn torch matplotlib

from google.cloud import bigquery
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -----------------------
# 1) BigQuery 로드
# -----------------------
PROJECT = "iitp-class-team-2-473114"
TABLE = "iitp-class-team-2-473114.SBS_Data.FirstRun"
HEXIDENT = "71C700"  # TODO

query = f"""
WITH base AS (
  SELECT
    HexIdent,
    FlightID,
    Callsign,
    TIMESTAMP(DATETIME(Date_MSG_Logged, Time_MSG_Logged), "UTC") AS ts,
    Latitude, Longitude, Altitude,
    GroundSpeed, Track, VerticalRate, IsOnGround
  FROM `{TABLE}`
  WHERE Latitude IS NOT NULL AND Longitude IS NOT NULL AND Altitude IS NOT NULL
)
SELECT *
FROM base
WHERE HexIdent = '{HEXIDENT}'
ORDER BY ts
"""

client = bigquery.Client(project=PROJECT)
df = client.query(query).to_dataframe()

# -----------------------
# 2) 전처리 (논문 취지 반영: 이상치 제거/결측 보간/스무딩)
#    - BigQuery 데이터는 보통 불규칙 샘플링이라 "리샘플링"이 매우 중요
# -----------------------
df["ts"] = pd.to_datetime(df["ts"], utc=True)
df = df.drop_duplicates(subset=["ts"]).sort_values("ts").set_index("ts")

# (a) 기본 이상치 제거: 위경도 범위 + 고도 양수
df = df[(df["Latitude"].between(-90, 90)) &
        (df["Longitude"].between(-180, 180)) &
        (df["Altitude"] > 0)]

# (b) 리샘플링: 1초 or 5초 등 고정 간격으로 맞추기 (데이터 밀도에 따라 조절)
#     - 너무 촘촘하면 결측이 많아짐, 너무 성기면 정보 손실
RESAMPLE = "5S"
df = df.resample(RESAMPLE).mean(numeric_only=True)

# (c) 결측 보간: 시간 기반 보간
df[["Latitude", "Longitude", "Altitude"]] = df[["Latitude", "Longitude", "Altitude"]].interpolate(method="time")

# (d) 노이즈 스무딩: 이동평균 (윈도우는 데이터에 맞게)
SMOOTH_W = 3
for c in ["Latitude", "Longitude", "Altitude"]:
    df[c] = df[c].rolling(SMOOTH_W, center=True).mean()

df = df.dropna(subset=["Latitude", "Longitude", "Altitude"]).reset_index()

# -----------------------
# 3) 학습 데이터 구성: (lat, lon, alt) 시계열 → 다음 스텝 예측
#    논문은 70% 학습 / 30% 실패 이후 예측 구간으로 시뮬레이션  [oai_citation:2‡s41598-023-46914-2 (1).pdf](sediment://file_00000000efc8722f9343b4ed00194f30)
# -----------------------
coords = df[["Latitude", "Longitude", "Altitude"]].to_numpy(dtype=np.float32)

scaler = MinMaxScaler()
coords_scaled = scaler.fit_transform(coords).astype(np.float32)

WINDOW = 10  # 과거 10 step -> 다음 step (RESAMPLE=5S면 50초 히스토리)
X, y = [], []
for i in range(len(coords_scaled) - WINDOW):
    X.append(coords_scaled[i:i+WINDOW])
    y.append(coords_scaled[i+WINDOW])

X = np.array(X, dtype=np.float32)   # (N, WINDOW, 3)
y = np.array(y, dtype=np.float32)   # (N, 3)

split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=256, shuffle=True)
test_loader  = DataLoader(SeqDataset(X_test, y_test), batch_size=256, shuffle=False)

# -----------------------
# 4) Bi-LSTM 모델 (논문 구조 취지: Bi-LSTM + MSE + Adam, epoch ~ 30)  [oai_citation:3‡s41598-023-46914-2 (1).pdf](sediment://file_00000000efc8722f9343b4ed00194f30)
# -----------------------
class BiLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=80, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, 3)

    def forward(self, x):
        out, _ = self.lstm(x)      # out: (B, T, 2H)
        out = out[:, -1, :]        # last step
        return self.fc(out)        # (B, 3)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BiLSTM(hidden_size=80).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 30
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(train_loader.dataset)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d} | train MSE: {train_loss:.6f}")

# -----------------------
# 5) 예측 및 시각화 (True vs Pred)
# -----------------------
model.eval()
with torch.no_grad():
    preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

pred_coords = scaler.inverse_transform(preds)
true_coords = scaler.inverse_transform(y_test)

# 2D 플롯 (lat-lon)
plt.figure()
plt.plot(true_coords[:, 1], true_coords[:, 0], label="True")       # x=lon, y=lat
plt.plot(pred_coords[:, 1], pred_coords[:, 0], label="Predicted")
plt.legend()
plt.title("Trajectory (Lat/Lon) - True vs Predicted")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# 고도 비교
plt.figure()
plt.plot(true_coords[:, 2], label="True Alt")
plt.plot(pred_coords[:, 2], label="Pred Alt")
plt.legend()
plt.title("Altitude - True vs Predicted")
plt.xlabel("Step")
plt.ylabel("Altitude")
plt.show()