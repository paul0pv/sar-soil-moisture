# validation/sentinel1_validation.py

"""
Validación del modelo con datos reales de Sentinel-1 IW

Requiere:
1. Descargar imágenes Sentinel-1 de ESA Copernicus Hub
2. Procesamiento con SNAP (calibración radiométrica, filtrado speckle)
3. Datos in situ sincronizados de campañas de campo

Sitios sugeridos (con datos públicos disponibles):
- ISMN (International Soil Moisture Network): >60 redes globales
- SMAP Validation Sites
- ESA CCI Soil Moisture validation sites
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import IEM_Model
from inversion.neural_network import load_trained_model


def validate_with_sentinel1(
    sentinel1_data_path="sentinel1_processed.csv",
    insitu_data_path="ground_truth.csv",
    trained_model_path="trained_model_rms_known.pth",
):
    """
    Compara estimaciones del modelo vs. mediciones in situ
    """
    # Cargar datos
    s1_df = pd.read_csv(sentinel1_data_path)
    gt_df = pd.read_csv(insitu_data_path)

    # Merge por fecha/ubicación
    merged = pd.merge(s1_df, gt_df, on=["date", "field_id"])

    # Cargar modelo de inversión
    model = load_trained_model(trained_model_path)

    # Estimar humedad
    mv_estimated = model.predict(
        merged["sigma0_VV"].values,
        merged["sigma0_VH"].values,
        merged["incidence_angle"].values,
        merged["rms_measured"].values,  # Si disponible
    )

    # Métricas
    mv_measured = merged["mv_insitu"].values
    rmse = np.sqrt(np.mean((mv_estimated - mv_measured) ** 2))
    mae = np.mean(np.abs(mv_estimated - mv_measured))
    bias = np.mean(mv_estimated - mv_measured)
    r2 = np.corrcoef(mv_estimated, mv_measured)[0, 1] ** 2

    print(f"Validación con datos reales:")
    print(f"  N samples: {len(merged)}")
    print(f"  RMSE: {rmse:.3f}%")
    print(f"  MAE: {mae:.3f}%")
    print(f"  Bias: {bias:.3f}%")
    print(f"  R²: {r2:.3f}")

    # Scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(mv_measured, mv_estimated, alpha=0.5)
    plt.plot([0, 50], [0, 50], "r--", lw=2, label="1:1 line")
    plt.xlabel("Humedad in situ (%)")
    plt.ylabel("Humedad estimada SAR (%)")
    plt.title(f"Validación Sentinel-1\nRMSE={rmse:.2f}%, R²={r2:.3f}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("sentinel1_validation.png", dpi=150)
    plt.close()

    return {"rmse": rmse, "mae": mae, "bias": bias, "r2": r2, "n_samples": len(merged)}
