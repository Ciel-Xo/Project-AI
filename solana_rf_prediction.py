"""
============================================================
  Implementasi Random Forest untuk Prediksi Harga Solana
  Tools : Python, Pandas, NumPy, Matplotlib, Scikit-learn,
          yfinance
  Target: Prediksi harga Close SOL-USD (1 hari ke depan)
============================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

# ──────────────────────────────────────────────
# 1. PENGAMBILAN DATA
# ──────────────────────────────────────────────

def ambil_data(ticker: str = "SOL-USD",
               start: str = "2022-01-01",
               end:   str = "2023-12-31") -> pd.DataFrame:
    """Mengambil data historis dari Yahoo Finance."""
    print(f"[INFO] Mengambil data {ticker} dari {start} s.d. {end} ...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    print(f"[INFO] Total data: {len(df)} baris")
    print(f"[INFO] Kolom     : {list(df.columns)}\n")
    print(df.head())
    print()
    return df


# ──────────────────────────────────────────────
# 2. PREPROCESSING & FEATURE ENGINEERING
# ──────────────────────────────────────────────

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan data dan menambahkan fitur teknikal.
    Fitur yang dibuat:
      - MA7, MA30          : Moving Average 7 & 30 hari
      - Volatility         : High - Low harian
      - Price_Change       : Perubahan Close harian (%)
      - Volume_Change      : Perubahan Volume harian (%)
      - High_Low_Ratio     : High / Low
      - Close_Open_Ratio   : Close / Open
      - Target             : Close hari berikutnya (label)
    """
    print("[INFO] Memulai preprocessing ...")

    
    mv = df.isnull().sum()
    print(f"[INFO] Missing values sebelum dropna:\n{mv}\n")
    df.dropna(inplace=True)

    # ── Fitur teknikal ──
    df = df.copy()
    df['MA7']            = df['Close'].rolling(window=7).mean()
    df['MA30']           = df['Close'].rolling(window=30).mean()
    df['Volatility']     = df['High'] - df['Low']
    df['Price_Change']   = df['Close'].pct_change() * 100
    df['Volume_Change']  = df['Volume'].pct_change() * 100
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']

    # ── Target: harga Close 1 hari ke depan ──
    df['Target'] = df['Close'].shift(-1)

    
    df.dropna(inplace=True)

    print(f"[INFO] Total data setelah preprocessing: {len(df)} baris")
    print(f"[INFO] Fitur:\n{df[['Close','MA7','MA30','Volatility','Price_Change','Target']].tail(3)}\n")
    return df


# ──────────────────────────────────────────────
# 3. PEMBAGIAN DATA TRAINING & TESTING
# ──────────────────────────────────────────────

FITUR = ['Open', 'High', 'Low', 'Close', 'Volume',
         'MA7', 'MA30', 'Volatility', 'Price_Change',
         'Volume_Change', 'High_Low_Ratio', 'Close_Open_Ratio']

def split_data(df: pd.DataFrame, rasio_train: float = 0.80):
    """
    Membagi data secara sekuensial (menjaga urutan waktu).
    Tidak menggunakan train_test_split acak agar validitas
    time-series terjaga.
    """
    X = df[FITUR].values
    y = df['Target'].values
    dates = df.index

    split = int(len(X) * rasio_train)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test       = dates[split:]

    print(f"[INFO] Data training : {len(X_train)} sampel")
    print(f"[INFO] Data testing  : {len(X_test)} sampel\n")
    return X_train, X_test, y_train, y_test, dates_test


# ──────────────────────────────────────────────
# 4. TRAINING MODEL RANDOM FOREST
# ──────────────────────────────────────────────

def train_model(X_train, y_train) -> RandomForestRegressor:
    """
    Melatih model RandomForestRegressor.
    Hyperparameter:
      n_estimators  = 200   (jumlah pohon)
      max_depth     = 10    (kedalaman max tiap pohon)
      min_samples_split = 5 (min sampel untuk split)
      random_state  = 42    (reproducible)
    """
    print("[INFO] Melatih model Random Forest ...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,          
        random_state=42
    )
    model.fit(X_train, y_train)
    print("[INFO] Training selesai.\n")
    return model


# ──────────────────────────────────────────────
# 5. EVALUASI MODEL
# ──────────────────────────────────────────────

def evaluasi(y_test, y_pred) -> dict:
    """Menghitung MAE, MSE, RMSE, dan MAPE."""
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mask = y_test != 0
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

    print("=" * 45)
    print("  HASIL EVALUASI MODEL RANDOM FOREST")
    print("=" * 45)
    print(f"  MAE  : {mae:.4f}  USD")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f} USD")
    print(f"  MAPE : {mape:.2f} %")
    print("=" * 45)
    print()
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}


# ──────────────────────────────────────────────
# 6. VISUALISASI
# ──────────────────────────────────────────────

def visualisasi(dates_test, y_test, y_pred,
                model: RandomForestRegressor,
                metrics: dict,
                simpan: str = "hasil_prediksi_solana.png"):
    """
    Membuat 3 grafik:
      (A) Harga aktual vs prediksi
      (B) Feature importance
      (C) Scatter plot aktual vs prediksi
    """
    # ── Warna & style ──
    WARNA_AKTUAL  = "#2E86AB"
    WARNA_PREDIKSI = "#E84855"
    WARNA_BG      = "#0F1923"
    WARNA_PANEL   = "#162130"
    WARNA_TEKS    = "#D1E8FF"
    WARNA_GRID    = "#1E3044"

    plt.rcParams.update({
        'figure.facecolor': WARNA_BG,
        'axes.facecolor'  : WARNA_PANEL,
        'axes.edgecolor'  : WARNA_GRID,
        'axes.labelcolor' : WARNA_TEKS,
        'xtick.color'     : WARNA_TEKS,
        'ytick.color'     : WARNA_TEKS,
        'text.color'      : WARNA_TEKS,
        'grid.color'      : WARNA_GRID,
        'grid.linestyle'  : '--',
        'grid.alpha'      : 0.5,
        'font.family'     : 'monospace',
    })

    fig = plt.figure(figsize=(18, 13), facecolor=WARNA_BG)
    fig.suptitle(
        "Implementasi Random Forest — Prediksi Harga Solana (SOL-USD)",
        fontsize=16, fontweight='bold', color=WARNA_TEKS, y=0.98
    )

    gs = GridSpec(2, 2, figure=fig,
                  hspace=0.40, wspace=0.30,
                  top=0.93, bottom=0.07,
                  left=0.07, right=0.97)

    
    ax1 = fig.add_subplot(gs[0, :])  
    ax1.plot(dates_test, y_test, label="Harga Aktual",
             color=WARNA_AKTUAL, linewidth=1.8, alpha=0.9)
    ax1.plot(dates_test, y_pred, label="Harga Prediksi (RF)",
             color=WARNA_PREDIKSI, linewidth=1.5, linestyle="--", alpha=0.85)
    ax1.fill_between(dates_test, y_test, y_pred,
                     alpha=0.08, color=WARNA_PREDIKSI)
    ax1.set_title("Harga Aktual vs Prediksi (Data Testing)", fontsize=12,
                  color=WARNA_TEKS, pad=10)
    ax1.set_xlabel("Tanggal")
    ax1.set_ylabel("Harga (USD)")
    ax1.legend(facecolor=WARNA_PANEL, edgecolor=WARNA_GRID,
               labelcolor=WARNA_TEKS, fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax1.grid(True)

    
    metrik_teks = (f"MAE={metrics['MAE']:.2f}  |  "
                   f"RMSE={metrics['RMSE']:.2f}  |  "
                   f"MAPE={metrics['MAPE']:.1f}%")
    ax1.annotate(metrik_teks, xy=(0.01, 0.04),
                 xycoords='axes fraction',
                 fontsize=9, color="#A8D8EA",
                 bbox=dict(boxstyle='round,pad=0.3',
                           facecolor=WARNA_BG, alpha=0.7,
                           edgecolor=WARNA_GRID))

    
    ax2 = fig.add_subplot(gs[1, 0])
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)
    colors_bar  = plt.cm.cool(np.linspace(0.2, 0.9, len(FITUR)))

    bars = ax2.barh(
        [FITUR[i] for i in sorted_idx],
        importances[sorted_idx],
        color=[colors_bar[i] for i in range(len(sorted_idx))],
        edgecolor=WARNA_PANEL
    )
    ax2.set_title("Feature Importance", fontsize=12, color=WARNA_TEKS, pad=10)
    ax2.set_xlabel("Importance Score")
    ax2.grid(True, axis='x')
    
    for bar in bars:
        w = bar.get_width()
        ax2.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                 f'{w:.3f}', va='center', fontsize=8, color=WARNA_TEKS)

    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(y_test, y_pred, alpha=0.45, s=18,
                color=WARNA_AKTUAL, edgecolors='none')

   
    lim_min = min(y_test.min(), y_pred.min()) * 0.95
    lim_max = max(y_test.max(), y_pred.max()) * 1.05
    ax3.plot([lim_min, lim_max], [lim_min, lim_max],
             color=WARNA_PREDIKSI, linestyle='--', linewidth=1.5,
             label="Prediksi Sempurna")
    ax3.set_xlim(lim_min, lim_max)
    ax3.set_ylim(lim_min, lim_max)
    ax3.set_title("Scatter: Aktual vs Prediksi", fontsize=12, color=WARNA_TEKS, pad=10)
    ax3.set_xlabel("Harga Aktual (USD)")
    ax3.set_ylabel("Harga Prediksi (USD)")
    ax3.legend(facecolor=WARNA_PANEL, edgecolor=WARNA_GRID,
               labelcolor=WARNA_TEKS, fontsize=9)
    ax3.grid(True)

    plt.savefig(simpan, dpi=150, bbox_inches='tight',
                facecolor=WARNA_BG)
    print(f"[INFO] Grafik disimpan: {simpan}")
    plt.show()


# ──────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":

    # Langkah 1 — Ambil data
    df_raw = ambil_data(ticker="SOL-USD",
                        start="2022-01-01",
                        end="2023-12-31")

    # Langkah 2 — Preprocessing & feature engineering
    df = preprocessing(df_raw)

    # Langkah 3 — Bagi data
    X_train, X_test, y_train, y_test, dates_test = split_data(df, rasio_train=0.80)

    # Langkah 4 — Training
    model = train_model(X_train, y_train)

    # Langkah 5 — Prediksi
    y_pred = model.predict(X_test)

    # Langkah 6 — Evaluasi
    metrics = evaluasi(y_test, y_pred)

    # ── Prediksi 1 hari ke depan (ditampilkan setelah evaluasi) ──
    fitur_terakhir = df[FITUR].iloc[[-1]].values
    prediksi_besok = model.predict(fitur_terakhir)[0]
    tanggal_terakhir = df.index[-1].strftime("%Y-%m-%d")
    print(f"[PREDIKSI] Berdasarkan data terakhir ({tanggal_terakhir}),")
    print(f"           estimasi harga Solana 1 hari ke depan: ${prediksi_besok:.2f} USD")
    print()

    # Langkah 7 — Visualisasi
    visualisasi(dates_test, y_test, y_pred, model, metrics,
                simpan="hasil_prediksi_solana.png")
