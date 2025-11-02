# analysis/response_surfaces.py

import numpy as np
import matplotlib.pyplot as plt
from models import IEM_Model


def analyze_response_surface(
    mv_range=(5, 45, 100),  # (min, max, n_points)
    rms_range=(0.5, 3.0, 80),
    theta_values=[30, 35, 40, 45],
    polarization="VV",
):
    """
    Genera y analiza superficies σ⁰(mv, rms) para diferentes ángulos
    """
    model = IEM_Model()

    mv_grid = np.linspace(*mv_range)
    rms_grid = np.linspace(*rms_range)
    MV, RMS = np.meshgrid(mv_grid, rms_grid)

    results = {}

    for theta in theta_values:
        print(f"\nProcessing θ={theta}°...")

        # Calcular superficie completa
        SIGMA = np.zeros_like(MV)
        for i in range(len(rms_grid)):
            SIGMA[i, :] = model.compute_backscatter(
                mv_grid, rms_grid[i], theta, polarization
            )

        # Análisis de sensibilidad
        dsigma_dmv = np.gradient(SIGMA, mv_grid, axis=1)
        dsigma_drms = np.gradient(SIGMA, rms_grid, axis=0)

        # Ratio de sensibilidades (invertibilidad)
        sensitivity_ratio = np.abs(dsigma_dmv) / (np.abs(dsigma_drms) + 1e-6)

        # Umbral de invertibilidad: ratio > 2.0
        invertible_mask = sensitivity_ratio > 2.0

        results[theta] = {
            "sigma_surface": SIGMA,
            "sensitivity_mv": dsigma_dmv,
            "sensitivity_rms": dsigma_drms,
            "invertible_fraction": np.mean(invertible_mask),
            "optimal_rms_range": identify_optimal_range(invertible_mask, rms_grid),
        }

        # Visualización
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Panel 1: Superficie σ⁰
        im1 = axes[0, 0].contourf(MV, RMS, SIGMA, levels=20, cmap="viridis")
        axes[0, 0].set_title(f"σ⁰(mv, rms) @ θ={theta}°")
        axes[0, 0].set_xlabel("Humedad (%)")
        axes[0, 0].set_ylabel("Rugosidad (cm)")
        plt.colorbar(im1, ax=axes[0, 0], label="σ⁰ (dB)")

        # Panel 2: Sensibilidad a humedad
        im2 = axes[0, 1].contourf(MV, RMS, dsigma_dmv, levels=20, cmap="RdYlGn")
        axes[0, 1].set_title("∂σ⁰/∂mv (sensibilidad a humedad)")
        plt.colorbar(im2, ax=axes[0, 1], label="dB/%")

        # Panel 3: Ratio de sensibilidades
        im3 = axes[1, 0].contourf(
            MV, RMS, sensitivity_ratio, levels=[0, 1, 2, 5, 10, 20], cmap="Spectral_r"
        )
        axes[1, 0].contour(
            MV, RMS, invertible_mask, levels=[0.5], colors="black", linewidths=2
        )
        axes[1, 0].set_title("Ratio sensibilidad (|∂σ⁰/∂mv| / |∂σ⁰/∂rms|)")
        plt.colorbar(im3, ax=axes[1, 0])

        # Panel 4: Máscara de invertibilidad
        axes[1, 1].contourf(
            MV,
            RMS,
            invertible_mask.astype(float),
            levels=[0, 0.5, 1],
            colors=["red", "green"],
            alpha=0.5,
        )
        axes[1, 1].set_title(
            f"Región invertible ({results[theta]['invertible_fraction'] * 100:.1f}% del espacio)"
        )
        axes[1, 1].set_xlabel("Humedad (%)")
        axes[1, 1].set_ylabel("Rugosidad (cm)")

        plt.tight_layout()
        plt.savefig(f"response_surface_theta{theta}.png", dpi=150)
        plt.close()

    # Reporte consolidado
    print("\n" + "=" * 60)
    print("ANÁLISIS DE INVERTIBILIDAD")
    print("=" * 60)
    for theta, data in results.items():
        print(f"θ = {theta}°:")
        print(f"  Fracción invertible: {data['invertible_fraction'] * 100:.1f}%")
        print(
            f"  Rms óptimo: {data['optimal_rms_range'][0]:.2f}-{data['optimal_rms_range'][1]:.2f} cm"
        )

    return results


def identify_optimal_range(mask, rms_grid, threshold=0.8):
    """
    Identifica rango de rms donde >80% del espacio mv es invertible
    """
    invertible_fraction_per_rms = np.mean(mask, axis=1)
    good_indices = np.where(invertible_fraction_per_rms > threshold)[0]

    if len(good_indices) == 0:
        return (np.nan, np.nan)

    return (rms_grid[good_indices[0]], rms_grid[good_indices[-1]])
