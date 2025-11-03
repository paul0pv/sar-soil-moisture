"""
Análisis de Sensibilidad del Modelo IEM-B

Implementa Objetivo Específico 2 y Diferenciación D2:
- Calcula superficies de respuesta σ⁰(mv, rms, θ)
- Computa Jacobiano de sensibilidades
- Identifica umbrales de invertibilidad
- Mapea regiones de ambigüedad

INNOVACIÓN (no en literatura previa):
- Caracterización exhaustiva de límites de inversión
- Mapeo explícito de número de condición
- Criterios cuantitativos de calidad pixel-específicos

Referencias:
- Baghdadi et al. (2006): Sensibilidad angular
- Verhoest et al. (2008): Problema de rugosidad
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models import IEM_Model
from core.constants import VALID_DOMAIN, QUALITY, PLOT
from core.utils import (
    compute_numerical_derivative,
    compute_gradient,
    classify_quality_from_sensitivity_ratio,
    validate_parameters,
)


class SensitivityAnalyzer:
    """
    Analizador de sensibilidad del modelo IEM-B.

    Calcula:
    - Derivadas parciales ∂σ⁰/∂mv, ∂σ⁰/∂rms, ∂σ⁰/∂θ
    - Matriz Jacobiana y número de condición
    - Superficies de respuesta multidimensionales
    - Mapas de invertibilidad
    """

    def __init__(self, model: IEM_Model = None):
        """
        Args:
            model: Instancia de IEM_Model (usa default si None)
        """
        self.model = model or IEM_Model()

    def compute_local_sensitivity(
        self,
        mv: float,
        rms: float,
        theta: float,
        polarization: str = "VV",
        delta: float = 1e-5,
    ) -> dict:
        """
        Calcula sensibilidades locales en un punto del espacio paramétrico.

        Args:
            mv: Humedad volumétrica (%)
            rms: Rugosidad RMS (cm)
            theta: Ángulo de incidencia (°)
            polarization: Polarización
            delta: Paso para diferencias finitas

        Returns:
            Dict con derivadas parciales y métricas de invertibilidad
        """
        # Validar parámetros
        validate_parameters(mv=mv, rms=rms, theta=theta, strict=False)

        # Función auxiliar para calcular σ⁰
        def sigma_func(params):
            """params = [mv, rms, theta]"""
            return self.model.compute_backscatter(
                params[0], params[1], params[2], polarization
            )

        # Punto central
        x0 = np.array([mv, rms, theta])
        sigma0_center = sigma_func(x0)

        # Derivadas parciales (diferencias centrales)
        dsigma_dmv = (
            self.model.compute_backscatter(mv + delta, rms, theta, polarization)
            - self.model.compute_backscatter(mv - delta, rms, theta, polarization)
        ) / (2 * delta)

        dsigma_drms = (
            self.model.compute_backscatter(mv, rms + delta, theta, polarization)
            - self.model.compute_backscatter(mv, rms - delta, theta, polarization)
        ) / (2 * delta)

        dsigma_dtheta = (
            self.model.compute_backscatter(mv, rms, theta + delta, polarization)
            - self.model.compute_backscatter(mv, rms, theta - delta, polarization)
        ) / (2 * delta)

        # Matriz Jacobiana J = [∂σ⁰/∂mv, ∂σ⁰/∂rms, ∂σ⁰/∂θ]
        jacobian = np.array([dsigma_dmv, dsigma_drms, dsigma_dtheta])

        # Ratio de sensibilidades (criterio de invertibilidad)
        sensitivity_ratio = abs(dsigma_dmv) / (abs(dsigma_drms) + 1e-10)

        # Clasificar invertibilidad
        invertibility = classify_quality_from_sensitivity_ratio(sensitivity_ratio)

        # Número de condición (para inversión de mv y rms simultáneos)
        # Matriz 2x2: [[∂σ⁰_VV/∂mv, ∂σ⁰_VV/∂rms],
        #              [∂σ⁰_HV/∂mv, ∂σ⁰_HV/∂rms]]
        # (requeriría polarimetría completa, aquí usamos aproximación)
        condition_number = (
            sensitivity_ratio if sensitivity_ratio > 1 else 1 / sensitivity_ratio
        )

        # ¿Es invertible?
        is_invertible = (
            abs(dsigma_dmv) > QUALITY.MIN_SENSITIVITY_MV
            and sensitivity_ratio > QUALITY.MIN_SENSITIVITY_RATIO
        )

        return {
            "sigma0": float(sigma0_center),
            "jacobian": {
                "dsigma_dmv": float(dsigma_dmv),
                "dsigma_drms": float(dsigma_drms),
                "dsigma_dtheta": float(dsigma_dtheta),
            },
            "sensitivity_ratio": float(sensitivity_ratio),
            "condition_number": float(condition_number),
            "invertibility": invertibility,
            "is_invertible": bool(is_invertible),
            "parameters": {"mv": mv, "rms": rms, "theta": theta},
        }

    def compute_response_surface(
        self,
        mv_range: tuple = (5, 45, 100),
        rms_range: tuple = (0.5, 2.5, 50),
        theta: float = 35.0,
        polarization: str = "VV",
    ) -> dict:
        """
        Calcula superficie de respuesta σ⁰(mv, rms) para θ fijo.

        Args:
            mv_range: (min, max, n_points)
            rms_range: (min, max, n_points)
            theta: Ángulo fijo (°)
            polarization: Polarización

        Returns:
            Dict con grids de mv, rms, sigma0, y sensibilidades
        """
        mv_grid = np.linspace(*mv_range)
        rms_grid = np.linspace(*rms_range)
        MV, RMS = np.meshgrid(mv_grid, rms_grid)

        # Calcular σ⁰ en toda la grilla
        SIGMA = np.zeros_like(MV)
        for i in range(len(rms_grid)):
            SIGMA[i, :] = self.model.compute_backscatter(
                mv_grid, rms_grid[i], theta, polarization
            )

        # Calcular sensibilidades (gradientes)
        dsigma_dmv = compute_gradient(SIGMA, mv_grid[1] - mv_grid[0], None, axis=1)
        dsigma_drms = compute_gradient(SIGMA, None, rms_grid[1] - rms_grid[0], axis=0)

        # Ratio de sensibilidades
        sensitivity_ratio = np.abs(dsigma_dmv) / (np.abs(dsigma_drms) + 1e-10)

        # Máscara de invertibilidad
        invertible_mask = (np.abs(dsigma_dmv) > QUALITY.MIN_SENSITIVITY_MV) & (
            sensitivity_ratio > QUALITY.MIN_SENSITIVITY_RATIO
        )

        return {
            "mv_grid": mv_grid,
            "rms_grid": rms_grid,
            "MV": MV,
            "RMS": RMS,
            "SIGMA": SIGMA,
            "dsigma_dmv": dsigma_dmv,
            "dsigma_drms": dsigma_drms,
            "sensitivity_ratio": sensitivity_ratio,
            "invertible_mask": invertible_mask,
            "invertible_fraction": float(np.mean(invertible_mask)),
            "theta": theta,
            "polarization": polarization,
        }

    def analyze_angular_dependence(
        self,
        mv: float = 25.0,
        rms: float = 1.5,
        theta_range: tuple = (20, 50, 31),
        polarizations: list = ["VV", "HV"],
    ) -> dict:
        """
        Analiza dependencia angular de σ⁰ y sensibilidades.

        Args:
            mv: Humedad fija (%)
            rms: Rugosidad fija (cm)
            theta_range: (min, max, n_points) para ángulo
            polarizations: Lista de polarizaciones

        Returns:
            Dict con resultados por polarización
        """
        theta_grid = np.linspace(*theta_range)
        results = {}

        for pol in polarizations:
            sigma0 = np.zeros_like(theta_grid)
            sensitivities_mv = np.zeros_like(theta_grid)
            sensitivities_rms = np.zeros_like(theta_grid)

            for i, theta in enumerate(theta_grid):
                # σ⁰
                sigma0[i] = self.model.compute_backscatter(mv, rms, theta, pol)

                # Sensibilidades
                sens = self.compute_local_sensitivity(mv, rms, theta, pol)
                sensitivities_mv[i] = sens["jacobian"]["dsigma_dmv"]
                sensitivities_rms[i] = abs(sens["jacobian"]["dsigma_drms"])

            # Ángulo óptimo = máxima sensibilidad a mv
            optimal_idx = np.argmax(sensitivities_mv)
            optimal_theta = theta_grid[optimal_idx]

            results[pol] = {
                "theta_grid": theta_grid,
                "sigma0": sigma0,
                "dsigma_dmv": sensitivities_mv,
                "dsigma_drms": sensitivities_rms,
                "sensitivity_ratio": sensitivities_mv / (sensitivities_rms + 1e-10),
                "optimal_theta": float(optimal_theta),
                "optimal_sensitivity": float(sensitivities_mv[optimal_idx]),
            }

        return results

    def map_invertibility_domain(
        self,
        mv_range: tuple = (5, 45, 50),
        rms_range: tuple = (0.5, 3.0, 40),
        theta_values: list = [30, 35, 40, 45],
        polarization: str = "VV",
        output_dir: str = "analysis/outputs",
    ) -> dict:
        """
        Mapea dominio de invertibilidad completo.

        INNOVACIÓN: Caracterización exhaustiva no presente en literatura.

        Args:
            mv_range: Rango de humedad
            rms_range: Rango de rugosidad
            theta_values: Lista de ángulos a analizar
            polarization: Polarización
            output_dir: Directorio para figuras

        Returns:
            Dict con mapas de invertibilidad por ángulo
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        results = {}

        print("\n" + "=" * 70)
        print("MAPEO DE DOMINIO DE INVERTIBILIDAD")
        print("=" * 70)

        for theta in theta_values:
            print(f"\nAnalizando θ = {theta}°...")

            # Calcular superficie de respuesta
            surface = self.compute_response_surface(
                mv_range=mv_range,
                rms_range=rms_range,
                theta=theta,
                polarization=polarization,
            )

            # Identificar rango óptimo de rugosidad
            # (donde >80% del espacio mv es invertible)
            invertible_fraction_per_rms = np.mean(surface["invertible_mask"], axis=1)
            good_rms_indices = np.where(invertible_fraction_per_rms > 0.8)[0]

            if len(good_rms_indices) > 0:
                optimal_rms_range = (
                    surface["rms_grid"][good_rms_indices[0]],
                    surface["rms_grid"][good_rms_indices[-1]],
                )
            else:
                optimal_rms_range = (np.nan, np.nan)

            results[theta] = {
                "surface": surface,
                "invertible_fraction": surface["invertible_fraction"],
                "optimal_rms_range": optimal_rms_range,
            }

            # Visualización
            self._plot_invertibility_analysis(surface, theta, polarization, output_dir)

            print(f"  Fracción invertible: {surface['invertible_fraction'] * 100:.1f}%")
            if not np.isnan(optimal_rms_range[0]):
                print(
                    f"  Rms óptimo: {optimal_rms_range[0]:.2f} - {optimal_rms_range[1]:.2f} cm"
                )
            else:
                print(f"  ⚠️  No se encontró rango óptimo (invertibilidad <80%)")

        # Reporte consolidado
        self._generate_invertibility_report(results, output_dir)

        return results

    def _plot_invertibility_analysis(
        self, surface: dict, theta: float, polarization: str, output_dir: str
    ):
        """Genera figura de 4 paneles con análisis de invertibilidad"""
        fig, axes = plt.subplots(2, 2, figsize=PLOT.FIGSIZE_QUAD)

        MV = surface["MV"]
        RMS = surface["RMS"]

        # Panel 1: Superficie σ⁰
        im1 = axes[0, 0].contourf(
            MV, RMS, surface["SIGMA"], levels=20, cmap=PLOT.CMAP_SIGMA
        )
        axes[0, 0].set_title(f"σ⁰({polarization}) @ θ={theta}°")
        axes[0, 0].set_xlabel("Humedad volumétrica (%)")
        axes[0, 0].set_ylabel("Rugosidad RMS (cm)")
        plt.colorbar(im1, ax=axes[0, 0], label="σ⁰ (dB)")

        # Panel 2: Sensibilidad a humedad
        im2 = axes[0, 1].contourf(
            MV, RMS, surface["dsigma_dmv"], levels=20, cmap=PLOT.CMAP_SENSITIVITY
        )
        axes[0, 1].set_title("Sensibilidad a humedad (∂σ⁰/∂mv)")
        axes[0, 1].set_xlabel("Humedad volumétrica (%)")
        axes[0, 1].set_ylabel("Rugosidad RMS (cm)")
        cbar2 = plt.colorbar(im2, ax=axes[0, 1], label="dB/%")

        # Línea de sensibilidad mínima
        axes[0, 1].contour(
            MV,
            RMS,
            np.abs(surface["dsigma_dmv"]) - QUALITY.MIN_SENSITIVITY_MV,
            levels=[0],
            colors="red",
            linewidths=2,
            linestyles="--",
        )

        # Panel 3: Ratio de sensibilidades
        ratio_levels = [0, 1, 2, 5, 10, 20, 50]
        im3 = axes[1, 0].contourf(
            MV,
            RMS,
            surface["sensitivity_ratio"],
            levels=ratio_levels,
            cmap="Spectral_r",
            extend="max",
        )
        axes[1, 0].contour(
            MV,
            RMS,
            surface["invertible_mask"].astype(float),
            levels=[0.5],
            colors="black",
            linewidths=2,
            label="Límite invertibilidad",
        )
        axes[1, 0].set_title("Ratio sensibilidad (|∂σ⁰/∂mv| / |∂σ⁰/∂rms|)")
        axes[1, 0].set_xlabel("Humedad volumétrica (%)")
        axes[1, 0].set_ylabel("Rugosidad RMS (cm)")
        cbar3 = plt.colorbar(im3, ax=axes[1, 0])
        cbar3.set_label("Ratio")

        # Panel 4: Máscara de invertibilidad
        colors_mask = ["darkred", "lightgreen"]
        im4 = axes[1, 1].contourf(
            MV,
            RMS,
            surface["invertible_mask"].astype(float),
            levels=[-0.5, 0.5, 1.5],
            colors=colors_mask,
            alpha=0.6,
        )
        axes[1, 1].set_title(
            f"Región invertible ({surface['invertible_fraction'] * 100:.1f}%)"
        )
        axes[1, 1].set_xlabel("Humedad volumétrica (%)")
        axes[1, 1].set_ylabel("Rugosidad RMS (cm)")

        # Leyenda personalizada
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="lightgreen", alpha=0.6, label="Invertible"),
            Patch(facecolor="darkred", alpha=0.6, label="No invertible"),
        ]
        axes[1, 1].legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/invertibility_analysis_theta{theta}.png",
            dpi=PLOT.DPI,
            bbox_inches="tight",
        )
        plt.close()

    def _generate_invertibility_report(self, results: dict, output_dir: str):
        """Genera reporte de texto con resultados consolidados"""
        report_path = f"{output_dir}/invertibility_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("REPORTE DE ANÁLISIS DE INVERTIBILIDAD\n")
            f.write("=" * 70 + "\n\n")

            f.write("OBJETIVO: Identificar umbrales de rugosidad donde la inversión\n")
            f.write("          σ⁰ → mv es factible con incertidumbre aceptable.\n\n")

            f.write("CRITERIOS DE INVERTIBILIDAD:\n")
            f.write(f"  • |∂σ⁰/∂mv| > {QUALITY.MIN_SENSITIVITY_MV} dB/%\n")
            f.write(f"  • |∂σ⁰/∂mv| / |∂σ⁰/∂rms| > {QUALITY.MIN_SENSITIVITY_RATIO}\n\n")

            f.write("-" * 70 + "\n")
            f.write("RESULTADOS POR ÁNGULO DE INCIDENCIA\n")
            f.write("-" * 70 + "\n\n")

            for theta, data in sorted(results.items()):
                f.write(f"θ = {theta}°:\n")
                f.write(
                    f"  Fracción del espacio (mv, rms) invertible: {data['invertible_fraction'] * 100:.1f}%\n"
                )

                if not np.isnan(data["optimal_rms_range"][0]):
                    f.write(f"  Rango óptimo de rugosidad (>80% invertible):\n")
                    f.write(
                        f"    {data['optimal_rms_range'][0]:.2f} cm ≤ rms ≤ {data['optimal_rms_range'][1]:.2f} cm\n"
                    )
                else:
                    f.write(
                        f"  ⚠️  No existe rango óptimo (invertibilidad <80% en todo rms)\n"
                    )

                f.write("\n")

            f.write("-" * 70 + "\n")
            f.write("RECOMENDACIONES OPERACIONALES\n")
            f.write("-" * 70 + "\n\n")

            # Encontrar mejor ángulo
            best_theta = max(
                results.keys(), key=lambda t: results[t]["invertible_fraction"]
            )
            best_frac = results[best_theta]["invertible_fraction"]

            f.write(f"1. Ángulo de incidencia óptimo: θ = {best_theta}°\n")
            f.write(f"   (Maximiza fracción invertible: {best_frac * 100:.1f}%)\n\n")

            # Consolidar rangos óptimos
            all_rms_ranges = [
                data["optimal_rms_range"]
                for data in results.values()
                if not np.isnan(data["optimal_rms_range"][0])
            ]

            if all_rms_ranges:
                rms_min = min(r[0] for r in all_rms_ranges)
                rms_max = max(r[1] for r in all_rms_ranges)
                f.write(
                    f"2. Rango de rugosidad recomendado para inversión confiable:\n"
                )
                f.write(f"   {rms_min:.2f} cm ≤ rms ≤ {rms_max:.2f} cm\n\n")
            else:
                f.write(
                    f"2. ⚠️  ADVERTENCIA: Inversión no confiable en ningún rango de rms\n\n"
                )

            f.write(f"3. Para rms > {VALID_DOMAIN.RMS_HIGH_QUALITY_MAX} cm:\n")
            f.write(f"   La sensibilidad a humedad colapsa. RMSE esperado >8%.\n")
            f.write(f"   Considerar información a priori o métodos alternativos.\n\n")

        print(f"\n✅ Reporte guardado: {report_path}")


def run_comprehensive_sensitivity_analysis(output_dir: str = "analysis/outputs"):
    """
    Pipeline completo de análisis de sensibilidad.

    Ejecuta:
    1. Mapeo de invertibilidad para múltiples ángulos
    2. Análisis de dependencia angular
    3. Identificación de configuraciones óptimas
    """
    print("\n" + "█" * 70)
    print("ANÁLISIS EXHAUSTIVO DE SENSIBILIDAD DEL MODELO IEM-B")
    print("█" * 70)

    # Inicializar analizador
    model = IEM_Model(sand_pct=40, clay_pct=30)  # Suelo franco típico
    analyzer = SensitivityAnalyzer(model)

    # 1. Mapeo de dominio de invertibilidad
    print("\n[1/2] Mapeando dominio de invertibilidad...")
    invertibility_results = analyzer.map_invertibility_domain(
        mv_range=(5, 45, 80),
        rms_range=(0.5, 3.0, 60),
        theta_values=[25, 30, 35, 40, 45],
        polarization="VV",
        output_dir=output_dir,
    )

    # 2. Análisis de dependencia angular
    print("\n[2/2] Analizando dependencia angular...")
    angular_results = analyzer.analyze_angular_dependence(
        mv=25.0, rms=1.5, theta_range=(20, 50, 31), polarizations=["VV", "HV"]
    )

    # Visualizar dependencia angular
    fig, axes = plt.subplots(2, 2, figsize=PLOT.FIGSIZE_QUAD)

    for idx, pol in enumerate(["VV", "HV"]):
        data = angular_results[pol]

        # Panel 1: σ⁰(θ)
        axes[0, idx].plot(data["theta_grid"], data["sigma0"], "b-", lw=2)
        axes[0, idx].axvline(
            data["optimal_theta"],
            color="r",
            ls="--",
            label=f"Óptimo: {data['optimal_theta']:.1f}°",
        )
        axes[0, idx].set_xlabel("Ángulo de incidencia (°)")
        axes[0, idx].set_ylabel("σ⁰ (dB)")
        axes[0, idx].set_title(f"Retrodispersión {pol}")
        axes[0, idx].grid(True, alpha=0.3)
        axes[0, idx].legend()

        # Panel 2: Sensibilidades
        ax2 = axes[1, idx]
        ax2.plot(data["theta_grid"], data["dsigma_dmv"], "g-", lw=2, label="∂σ⁰/∂mv")
        ax2.plot(
            data["theta_grid"], data["dsigma_drms"], "r-", lw=2, label="|∂σ⁰/∂rms|"
        )
        ax2.axhline(
            QUALITY.MIN_SENSITIVITY_MV,
            color="g",
            ls=":",
            label=f"Umbral mv ({QUALITY.MIN_SENSITIVITY_MV} dB/%)",
        )
        ax2.set_xlabel("Ángulo de incidencia (°)")
        ax2.set_ylabel("Sensibilidad (dB/% o dB/cm)")
        ax2.set_title(f"Sensibilidades {pol}")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/angular_dependence.png", dpi=PLOT.DPI, bbox_inches="tight"
    )
    plt.close()

    print("\n" + "█" * 70)
    print("ANÁLISIS COMPLETADO")
    print("█" * 70)
    print(f"Figuras guardadas en: {output_dir}/")
    print(f"  • invertibility_analysis_theta{{25,30,35,40,45}}.png")
    print(f"  • angular_dependence.png")
    print(f"  • invertibility_report.txt")

    return invertibility_results, angular_results


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Análisis de sensibilidad IEM-B")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis/outputs",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--theta_values",
        type=float,
        nargs="+",
        default=[25, 30, 35, 40, 45],
        help="Ángulos de incidencia a analizar",
    )

    args = parser.parse_args()

    # Ejecutar análisis completo
    results = run_comprehensive_sensitivity_analysis(output_dir=args.output_dir)

    print("\n✅ ANÁLISIS DE SENSIBILIDAD COMPLETADO\n")
