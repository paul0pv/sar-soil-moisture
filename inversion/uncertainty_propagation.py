"""
Propagación de Incertidumbres en Estimación de Humedad del Suelo

Implementa Objetivo Específico 4 y Diferenciaciones D2, D3, O2:
- Propagación analítica de errores (Taylor de primer orden)
- Análisis Monte Carlo
- Intervalos de confianza probabilísticos
- Predicción de calidad SIN validación in situ (INNOVACIÓN O2)

Referencias:
- Doubková et al. (2012): Propagación de errores ASAR
- Baghdadi et al. (2012): Validación con ruido sintético
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models import IEM_Model
from core.constants import VALID_DOMAIN, QUALITY, ANALYSIS, PLOT
from core.utils import (
    compute_numerical_derivative,
    classify_quality_from_uncertainty,
    rmse,
    mae,
    bias,
)


class UncertaintyPropagator:
    """
    Propagador de incertidumbres para estimación de humedad del suelo.

    Implementa dos métodos:
    1. Analítico: Expansión de Taylor de primer orden
    2. Monte Carlo: Simulación estocástica

    INNOVACIÓN (O2): Predice incertidumbre ANTES de tener datos de validación,
    usando solo análisis de sensibilidad del modelo físico.
    """

    def __init__(self, model: IEM_Model = None):
        """
        Args:
            model: Instancia de IEM_Model
        """
        self.model = model or IEM_Model()

    def compute_jacobian(
        self,
        mv: float,
        rms: float,
        theta: float,
        polarization: str = "VV",
        delta: float = 1e-5,
    ) -> dict:
        """
        Calcula matriz Jacobiana de sensibilidades.

        J = [∂σ⁰/∂mv, ∂σ⁰/∂rms, ∂σ⁰/∂θ]

        Args:
            mv: Humedad volumétrica (%)
            rms: Rugosidad RMS (cm)
            theta: Ángulo de incidencia (°)
            polarization: Polarización
            delta: Paso para diferencias finitas

        Returns:
            Dict con componentes del Jacobiano
        """
        # Punto central
        sigma0_center = self.model.compute_backscatter(mv, rms, theta, polarization)

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

        return {
            "sigma0": float(sigma0_center),
            "dsigma_dmv": float(dsigma_dmv),
            "dsigma_drms": float(dsigma_drms),
            "dsigma_dtheta": float(dsigma_dtheta),
            "jacobian_array": np.array([dsigma_dmv, dsigma_drms, dsigma_dtheta]),
        }

    def propagate_error_analytical(
        self,
        mv_true: float,
        rms_true: float,
        theta_true: float,
        polarization: str = "VV",
        sigma_sensor: float = 1.0,  # dB (error radiométrico Sentinel-1)
        sigma_rms: float = 0.3,  # cm (error medición rugosidad)
        sigma_theta: float = 0.5,  # ° (error geolocalización)
        rms_known: bool = False,
    ) -> dict:
        """
        Propagación analítica usando expansión de Taylor de primer orden.

        Fórmula (asumiendo variables independientes):
        σ²_mv = (∂mv/∂σ⁰)² σ²_sensor + (∂mv/∂rms)² σ²_rms + (∂mv/∂θ)² σ²_theta

        Args:
            mv_true: Humedad verdadera (%)
            rms_true: Rugosidad verdadera (cm)
            theta_true: Ángulo verdadero (°)
            polarization: Polarización
            sigma_sensor: Incertidumbre del sensor (dB)
            sigma_rms: Incertidumbre en rugosidad (cm)
            sigma_theta: Incertidumbre en ángulo (°)
            rms_known: Si rugosidad es conocida exactamente (σ_rms = 0)

        Returns:
            Dict con incertidumbres propagadas y diagnósticos
        """
        # Calcular Jacobiano
        J = self.compute_jacobian(mv_true, rms_true, theta_true, polarization)

        # Verificar singularidad
        if abs(J["dsigma_dmv"]) < 1e-6:
            return {
                "status": "singular",
                "reason": "Sensibilidad a mv es nula (∂σ⁰/∂mv ≈ 0)",
                "mv_uncertainty": np.inf,
                "quality_flag": 0,
            }

        # Derivadas inversas (regla de la cadena)
        # Si σ⁰ = f(mv, rms, θ), entonces mv = g(σ⁰, rms, θ)
        # ∂mv/∂σ⁰ = 1 / (∂σ⁰/∂mv)  [manteniendo rms, θ constantes]
        dmv_dsigma = 1.0 / J["dsigma_dmv"]
        dmv_drms = -J["dsigma_drms"] / J["dsigma_dmv"]
        dmv_dtheta = -J["dsigma_dtheta"] / J["dsigma_dmv"]

        # Propagación cuadrática
        if rms_known:
            sigma_rms = 0.0  # Rugosidad conocida exactamente

        var_mv = (
            (dmv_dsigma * sigma_sensor) ** 2
            + (dmv_drms * sigma_rms) ** 2
            + (dmv_dtheta * sigma_theta) ** 2
        )

        sigma_mv = np.sqrt(var_mv)

        # Intervalos de confianza
        ci_68 = (mv_true - sigma_mv, mv_true + sigma_mv)  # 1σ
        ci_95 = (mv_true - 1.96 * sigma_mv, mv_true + 1.96 * sigma_mv)  # 2σ
        ci_99 = (mv_true - 2.58 * sigma_mv, mv_true + 2.58 * sigma_mv)  # 3σ

        # Desglose de contribuciones
        total_var = var_mv
        contrib_sensor = (
            (dmv_dsigma * sigma_sensor) ** 2 / total_var * 100 if total_var > 0 else 0
        )
        contrib_rms = (
            (dmv_drms * sigma_rms) ** 2 / total_var * 100 if total_var > 0 else 0
        )
        contrib_theta = (
            (dmv_dtheta * sigma_theta) ** 2 / total_var * 100 if total_var > 0 else 0
        )

        # Clasificar calidad
        quality = classify_quality_from_uncertainty(sigma_mv)

        return {
            "status": "success",
            "mv_true": float(mv_true),
            "mv_uncertainty": float(sigma_mv),
            "relative_uncertainty_pct": float(sigma_mv / mv_true * 100)
            if mv_true > 0
            else np.inf,
            "confidence_intervals": {
                "ci_68": ci_68,  # 68% (1σ)
                "ci_95": ci_95,  # 95% (2σ)
                "ci_99": ci_99,  # 99% (3σ)
            },
            "contributions_pct": {
                "sensor_noise": float(contrib_sensor),
                "roughness_error": float(contrib_rms),
                "geometry_error": float(contrib_theta),
            },
            "jacobian": {
                "dsigma_dmv": J["dsigma_dmv"],
                "dsigma_drms": J["dsigma_drms"],
                "dsigma_dtheta": J["dsigma_dtheta"],
            },
            "inverse_jacobian": {
                "dmv_dsigma": float(dmv_dsigma),
                "dmv_drms": float(dmv_drms),
                "dmv_dtheta": float(dmv_dtheta),
            },
            "quality_flag": quality["flag"],
            "quality_label": quality["label"],
            "parameters": {
                "mv": mv_true,
                "rms": rms_true,
                "theta": theta_true,
                "sigma_sensor": sigma_sensor,
                "sigma_rms": sigma_rms,
                "sigma_theta": sigma_theta,
            },
        }

    def monte_carlo_uncertainty(
        self,
        mv_true: float,
        rms_true: float,
        theta_true: float,
        polarization: str = "VV",
        sigma_sensor: float = 1.0,
        sigma_rms: float = 0.3,
        sigma_theta: float = 0.5,
        n_samples: int = ANALYSIS.N_MC_SAMPLES,
        inversion_method: str = "numerical",
    ) -> dict:
        """
        Análisis de incertidumbre mediante simulación Monte Carlo.

        Algoritmo:
        1. Generar N muestras de (σ⁰_obs, rms_obs, θ_obs) con ruido
        2. Para cada muestra, invertir σ⁰_obs → mv_estimated
        3. Calcular estadísticas de la distribución de mv_estimated

        Args:
            mv_true: Humedad verdadera (%)
            rms_true: Rugosidad verdadera (cm)
            theta_true: Ángulo verdadero (°)
            polarization: Polarización
            sigma_sensor: Desviación estándar del ruido del sensor (dB)
            sigma_rms: Desviación estándar del error de rugosidad (cm)
            sigma_theta: Desviación estándar del error de ángulo (°)
            n_samples: Número de muestras Monte Carlo
            inversion_method: 'numerical' (optimización) o 'lut' (lookup table)

        Returns:
            Dict con estadísticas de la distribución de mv_estimated
        """
        # Generar σ⁰ verdadero
        sigma0_true = self.model.compute_backscatter(
            mv_true, rms_true, theta_true, polarization
        )

        # Generar perturbaciones
        sigma0_samples = sigma0_true + np.random.normal(0, sigma_sensor, n_samples)
        rms_samples = rms_true + np.random.normal(0, sigma_rms, n_samples)
        theta_samples = theta_true + np.random.normal(0, sigma_theta, n_samples)

        # Clip a rangos físicos
        rms_samples = np.clip(
            rms_samples, VALID_DOMAIN.RMS_SAFE_MIN, VALID_DOMAIN.RMS_SAFE_MAX
        )
        theta_samples = np.clip(
            theta_samples, VALID_DOMAIN.THETA_MIN, VALID_DOMAIN.THETA_MAX
        )

        # Invertir cada muestra
        mv_samples = np.zeros(n_samples)

        print(f"\nEjecutando Monte Carlo ({n_samples} muestras)...")

        for i in range(n_samples):
            if (i + 1) % (n_samples // 10) == 0:
                print(f"  Progreso: {(i + 1) / n_samples * 100:.0f}%")

            mv_samples[i] = self._invert_sigma0(
                sigma0_obs=sigma0_samples[i],
                rms=rms_samples[i],
                theta=theta_samples[i],
                polarization=polarization,
                method=inversion_method,
            )

        # Filtrar inversiones fallidas (NaN)
        valid_mask = ~np.isnan(mv_samples)
        mv_valid = mv_samples[valid_mask]
        n_valid = len(mv_valid)
        n_failed = n_samples - n_valid

        if n_valid == 0:
            return {
                "status": "failed",
                "reason": "Todas las inversiones fallaron",
                "n_samples": n_samples,
                "n_failed": n_samples,
            }

        # Estadísticas de la distribución
        mv_mean = np.mean(mv_valid)
        mv_std = np.std(mv_valid)
        mv_median = np.median(mv_valid)
        mv_p05 = np.percentile(mv_valid, 2.5)
        mv_p95 = np.percentile(mv_valid, 97.5)
        mv_p25 = np.percentile(mv_valid, 25)
        mv_p75 = np.percentile(mv_valid, 75)

        # Métricas de error
        bias_mc = mv_mean - mv_true
        rmse_mc = np.sqrt(np.mean((mv_valid - mv_true) ** 2))
        mae_mc = np.mean(np.abs(mv_valid - mv_true))

        return {
            "status": "success",
            "mv_true": float(mv_true),
            "mv_mean": float(mv_mean),
            "mv_std": float(mv_std),
            "mv_median": float(mv_median),
            "confidence_intervals": {
                "ci_50_iqr": (float(mv_p25), float(mv_p75)),  # Rango intercuartil
                "ci_95": (float(mv_p05), float(mv_p95)),  # 95% (percentiles)
            },
            "bias": float(bias_mc),
            "rmse": float(rmse_mc),
            "mae": float(mae_mc),
            "samples": mv_valid,
            "n_samples": n_samples,
            "n_valid": n_valid,
            "n_failed": n_failed,
            "success_rate": float(n_valid / n_samples),
        }

    def _invert_sigma0(
        self,
        sigma0_obs: float,
        rms: float,
        theta: float,
        polarization: str = "VV",
        method: str = "numerical",
        mv_range: tuple = (VALID_DOMAIN.MV_MIN, VALID_DOMAIN.MV_MAX),
    ) -> float:
        """
        Invierte σ⁰ observado para encontrar mv.

        Asume rms y θ conocidos, busca mv que minimiza |σ⁰_sim - σ⁰_obs|.

        Args:
            sigma0_obs: σ⁰ observado (dB)
            rms: Rugosidad (cm)
            theta: Ángulo (°)
            polarization: Polarización
            method: 'numerical' (optimización) o 'lut' (lookup table)
            mv_range: Rango de búsqueda para mv

        Returns:
            mv estimado (%) o NaN si falla
        """
        if method == "numerical":
            # Optimización numérica
            def objective(mv):
                sigma0_sim = self.model.compute_backscatter(
                    mv, rms, theta, polarization
                )
                return (sigma0_sim - sigma0_obs) ** 2

            result = minimize_scalar(
                objective, bounds=mv_range, method="bounded", options={"xatol": 1e-3}
            )

            if result.success:
                # Verificar que el error residual sea razonable
                residual = np.sqrt(result.fun)
                if residual < 5.0:  # Error < 5 dB (tolerancia amplia)
                    return result.x

            return np.nan

        elif method == "lut":
            # Lookup table (más rápido pero menos preciso)
            mv_grid = np.linspace(*mv_range, 100)
            sigma0_grid = self.model.compute_backscatter(
                mv_grid, rms, theta, polarization
            )

            # Encontrar mv más cercano
            idx = np.argmin(np.abs(sigma0_grid - sigma0_obs))
            return mv_grid[idx]

        else:
            raise ValueError(f"Método '{method}' no reconocido")

    def predict_quality_map(
        self,
        mv_range: tuple = (5, 45, 50),
        rms_range: tuple = (0.5, 2.5, 40),
        theta: float = 35.0,
        polarization: str = "VV",
        sigma_sensor: float = 1.0,
        output_path: str = "analysis/outputs/quality_map_predicted.png",
    ) -> dict:
        """
        Predice mapa de calidad esperada SIN necesidad de validación.

        INNOVACIÓN (O2): Usa solo análisis de sensibilidad del modelo físico
        para estimar incertidumbre esperada en cada píxel.

        Args:
            mv_range: Rango de humedad
            rms_range: Rango de rugosidad
            theta: Ángulo de incidencia
            polarization: Polarización
            sigma_sensor: Ruido del sensor (dB)
            output_path: Ruta para guardar figura

        Returns:
            Dict con mapas de incertidumbre y calidad
        """
        mv_grid = np.linspace(*mv_range)
        rms_grid = np.linspace(*rms_range)
        MV, RMS = np.meshgrid(mv_grid, rms_grid)

        # Calcular incertidumbre en cada punto
        UNCERTAINTY = np.zeros_like(MV)
        QUALITY_FLAG = np.zeros_like(MV, dtype=int)

        print(f"\nCalculando mapa de calidad predicha para θ={theta}°...")

        total_points = len(rms_grid) * len(mv_grid)
        computed = 0

        for i in range(len(rms_grid)):
            for j in range(len(mv_grid)):
                result = self.propagate_error_analytical(
                    mv_true=mv_grid[j],
                    rms_true=rms_grid[i],
                    theta_true=theta,
                    polarization=polarization,
                    sigma_sensor=sigma_sensor,
                    sigma_rms=0.3,  # Error típico de medición
                    sigma_theta=0.5,
                )

                if result["status"] == "success":
                    UNCERTAINTY[i, j] = result["mv_uncertainty"]
                    QUALITY_FLAG[i, j] = result["quality_flag"]
                else:
                    UNCERTAINTY[i, j] = np.nan
                    QUALITY_FLAG[i, j] = 0

                computed += 1
                if computed % (total_points // 20) == 0:
                    print(f"  Progreso: {computed / total_points * 100:.0f}%")

        # Estadísticas
        valid_mask = ~np.isnan(UNCERTAINTY)
        high_quality_fraction = np.sum(QUALITY_FLAG == 3) / valid_mask.sum()
        medium_quality_fraction = np.sum(QUALITY_FLAG == 2) / valid_mask.sum()
        low_quality_fraction = np.sum(QUALITY_FLAG == 1) / valid_mask.sum()
        poor_quality_fraction = np.sum(QUALITY_FLAG == 0) / valid_mask.sum()

        # Visualización
        fig, axes = plt.subplots(1, 2, figsize=PLOT.FIGSIZE_DOUBLE)

        # Panel 1: Mapa de incertidumbre
        levels_unc = [0, 3, 5, 8, 12, 20]
        im1 = axes[0].contourf(
            MV,
            RMS,
            UNCERTAINTY,
            levels=levels_unc,
            cmap=PLOT.CMAP_UNCERTAINTY,
            extend="max",
        )
        axes[0].set_title(f"Incertidumbre esperada σ_mv @ θ={theta}°\n(Sin validación)")
        axes[0].set_xlabel("Humedad volumétrica (%)")
        axes[0].set_ylabel("Rugosidad RMS (cm)")
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label("σ_mv (%)")

        # Líneas de calidad
        axes[0].contour(
            MV,
            RMS,
            UNCERTAINTY,
            levels=[
                QUALITY.UNCERTAINTY_HIGH_QUALITY,
                QUALITY.UNCERTAINTY_MEDIUM_QUALITY,
                QUALITY.UNCERTAINTY_LOW_QUALITY,
            ],
            colors=["green", "yellow", "orange"],
            linewidths=2,
            linestyles="--",
        )

        # Panel 2: Mapa de flags de calidad
        colors_quality = ["darkred", "orange", "yellow", "green"]
        im2 = axes[1].contourf(
            MV,
            RMS,
            QUALITY_FLAG,
            levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
            colors=colors_quality,
            alpha=0.7,
        )
        axes[1].set_title(f"Flags de calidad predichos @ θ={theta}°")
        axes[1].set_xlabel("Humedad volumétrica (%)")
        axes[1].set_ylabel("Rugosidad RMS (cm)")

        # Leyenda personalizada
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(
                facecolor="green",
                alpha=0.7,
                label=f"Alta ({high_quality_fraction * 100:.0f}%)",
            ),
            Patch(
                facecolor="yellow",
                alpha=0.7,
                label=f"Media ({medium_quality_fraction * 100:.0f}%)",
            ),
            Patch(
                facecolor="orange",
                alpha=0.7,
                label=f"Baja ({low_quality_fraction * 100:.0f}%)",
            ),
            Patch(
                facecolor="darkred",
                alpha=0.7,
                label=f"No conf. ({poor_quality_fraction * 100:.0f}%)",
            ),
        ]
        axes[1].legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=PLOT.DPI, bbox_inches="tight")
        plt.close()

        print(f"\n✅ Mapa de calidad guardado: {output_path}")
        print(f"\nDistribución de calidad:")
        print(f"  Alta:       {high_quality_fraction * 100:.1f}%")
        print(f"  Media:      {medium_quality_fraction * 100:.1f}%")
        print(f"  Baja:       {low_quality_fraction * 100:.1f}%")
        print(f"  No confiable: {poor_quality_fraction * 100:.1f}%")

        return {
            "mv_grid": mv_grid,
            "rms_grid": rms_grid,
            "uncertainty_map": UNCERTAINTY,
            "quality_map": QUALITY_FLAG,
            "statistics": {
                "high_quality_fraction": float(high_quality_fraction),
                "medium_quality_fraction": float(medium_quality_fraction),
                "low_quality_fraction": float(low_quality_fraction),
                "poor_quality_fraction": float(poor_quality_fraction),
                "mean_uncertainty": float(np.nanmean(UNCERTAINTY)),
                "median_uncertainty": float(np.nanmedian(UNCERTAINTY)),
            },
            "theta": theta,
            "polarization": polarization,
        }


def run_comprehensive_uncertainty_analysis(output_dir: str = "analysis/outputs"):
    """
    Pipeline completo de análisis de incertidumbres.

    Ejecuta:
    1. Comparación propagación analítica vs. Monte Carlo
    2. Mapas de calidad predicha (sin validación)
    3. Análisis de contribuciones de error
    """
    print("\n" + "█" * 70)
    print("ANÁLISIS EXHAUSTIVO DE PROPAGACIÓN DE INCERTIDUMBRES")
    print("█" * 70)

    # Inicializar
    model = IEM_Model(sand_pct=40, clay_pct=30)
    propagator = UncertaintyPropagator(model)

    # Caso de estudio
    mv_test = 25.0  # %
    rms_test = 1.5  # cm
    theta_test = 35.0  # °

    print(f"\n[1/3] Caso de estudio: mv={mv_test}%, rms={rms_test}cm, θ={theta_test}°")

    # 1. Propagación analítica
    print("\n  a) Propagación analítica...")
    analytical = propagator.propagate_error_analytical(
        mv_true=mv_test,
        rms_true=rms_test,
        theta_true=theta_test,
        polarization="VV",
        sigma_sensor=1.0,
        sigma_rms=0.3,
        sigma_theta=0.5,
    )

    print(f"\n  Resultados analíticos:")
    print(
        f"    σ_mv = {analytical['mv_uncertainty']:.3f}% ({analytical['relative_uncertainty_pct']:.1f}% relativo)"
    )
    print(
        f"    IC 95%: [{analytical['confidence_intervals']['ci_95'][0]:.2f}, {analytical['confidence_intervals']['ci_95'][1]:.2f}]%"
    )
    print(f"    Calidad: {analytical['quality_label']}")
    print(f"    Contribuciones:")
    print(
        f"      Ruido sensor:     {analytical['contributions_pct']['sensor_noise']:.1f}%"
    )
    print(
        f"      Error rugosidad:  {analytical['contributions_pct']['roughness_error']:.1f}%"
    )
    print(
        f"      Error geometría:  {analytical['contributions_pct']['geometry_error']:.1f}%"
    )

    # 2. Monte Carlo
    print("\n  b) Simulación Monte Carlo...")
    mc = propagator.monte_carlo_uncertainty(
        mv_true=mv_test,
        rms_true=rms_test,
        theta_true=theta_test,
        polarization="VV",
        sigma_sensor=1.0,
        sigma_rms=0.3,
        sigma_theta=0.5,
        n_samples=5000,
        inversion_method="numerical",
    )

    if mc["status"] == "success":
        print(f"\n  Resultados Monte Carlo:")
        print(f"    mv_estimado = {mc['mv_mean']:.3f} ± {mc['mv_std']:.3f}%")
        print(
            f"    IC 95%: [{mc['confidence_intervals']['ci_95'][0]:.2f}, {mc['confidence_intervals']['ci_95'][1]:.2f}]%"
        )
        print(f"    Sesgo: {mc['bias']:.3f}%")
        print(f"    RMSE: {mc['rmse']:.3f}%")
        print(f"    Tasa de éxito: {mc['success_rate'] * 100:.1f}%")

        # Comparación
        diff_std = abs(analytical["mv_uncertainty"] - mc["mv_std"])
        agreement = diff_std / mc["mv_std"] * 100

        print(f"\n  Acuerdo analítico vs. MC:")
        print(f"    Diferencia en σ_mv: {diff_std:.3f}% ({agreement:.1f}% relativo)")
        if agreement < 10:
            print(f"    ✅ Excelente acuerdo (<10%)")
        elif agreement < 20:
            print(f"    ✓ Buen acuerdo (<20%)")
        else:
            print(f"    ⚠️  Discrepancia significativa (>{agreement:.0f}%)")

    # 3. Mapas de calidad predicha
    print("\n[2/3] Generando mapas de calidad predicha...")
    quality_map = propagator.predict_quality_map(
        mv_range=(5, 45, 60),
        rms_range=(0.5, 2.5, 50),
        theta=35.0,
        polarization="VV",
        sigma_sensor=1.0,
        output_path=f"{output_dir}/quality_map_predicted_theta35.png",
    )

    # 4. Análisis multi-angular
    print("\n[3/3] Analizando dependencia angular de incertidumbres...")

    theta_values = [25, 30, 35, 40, 45]
    uncertainties_by_theta = []

    for theta in theta_values:
        result = propagator.propagate_error_analytical(
            mv_true=mv_test,
            rms_true=rms_test,
            theta_true=theta,
            polarization="VV",
            sigma_sensor=1.0,
            sigma_rms=0.3,
            sigma_theta=0.5,
        )
        uncertainties_by_theta.append(result["mv_uncertainty"])

    # Visualizar
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(theta_values, uncertainties_by_theta, "bo-", lw=2, markersize=8)
    ax.axhline(
        QUALITY.UNCERTAINTY_HIGH_QUALITY,
        color="g",
        ls="--",
        label=f"Umbral alta calidad ({QUALITY.UNCERTAINTY_HIGH_QUALITY}%)",
    )
    ax.axhline(
        QUALITY.UNCERTAINTY_MEDIUM_QUALITY,
        color="orange",
        ls="--",
        label=f"Umbral media calidad ({QUALITY.UNCERTAINTY_MEDIUM_QUALITY}%)",
    )
    ax.axhline(
        QUALITY.UNCERTAINTY_LOW_QUALITY,
        color="red",
        ls="--",
        label=f"Umbral baja calidad ({QUALITY.UNCERTAINTY_LOW_QUALITY}%)",
    )
    ax.set_xlabel("Ángulo de incidencia (°)", fontsize=12)
    ax.set_ylabel("Incertidumbre σ_mv (%)", fontsize=12)
    ax.set_title(
        f"Incertidumbre vs. ángulo\nmv={mv_test}%, rms={rms_test}cm", fontsize=14
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/uncertainty_vs_angle.png", dpi=PLOT.DPI, bbox_inches="tight"
    )
    plt.close()

    # Ángulo óptimo
    optimal_idx = np.argmin(uncertainties_by_theta)
    optimal_theta = theta_values[optimal_idx]
    min_uncertainty = uncertainties_by_theta[optimal_idx]

    print(f"\n  Ángulo óptimo: θ = {optimal_theta}°")
    print(f"  Mínima incertidumbre: σ_mv = {min_uncertainty:.3f}%")

    print("\n" + "█" * 70)
    print("ANÁLISIS DE INCERTIDUMBRES COMPLETADO")
    print("█" * 70)
    print(f"\nResultados en: {output_dir}/")
    print("  • quality_map_predicted_theta35.png")
    print("  • uncertainty_vs_angle.png")
    print("\n📊 INNOVACIÓN CLAVE (O2):")
    print("   Predicción de calidad SIN validación in situ mediante")
    print("   análisis de sensibilidad del modelo físico solamente.\n")

    return {
        "analytical": analytical,
        "monte_carlo": mc,
        "quality_map": quality_map,
        "angular_dependence": {
            "theta_values": theta_values,
            "uncertainties": uncertainties_by_theta,
            "optimal_theta": optimal_theta,
            "min_uncertainty": min_uncertainty,
        },
    }


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Propagación de incertidumbres IEM-B (INNOVACIÓN O2)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis/outputs",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["analytical", "monte_carlo", "quality_map", "full"],
        help="Modo de análisis",
    )
    parser.add_argument(
        "--n_mc", type=int, default=10000, help="Número de muestras Monte Carlo"
    )

    args = parser.parse_args()

    if args.mode == "full":
        # Análisis completo
        results = run_comprehensive_uncertainty_analysis(output_dir=args.output_dir)

    elif args.mode == "analytical":
        # Solo propagación analítica
        propagator = UncertaintyPropagator()
        result = propagator.propagate_error_analytical(
            mv_true=25.0, rms_true=1.5, theta_true=35.0
        )
        print("\nResultado analítico:")
        print(f"  σ_mv = {result['mv_uncertainty']:.3f}%")
        print(f"  Calidad: {result['quality_label']}")

    elif args.mode == "monte_carlo":
        # Solo Monte Carlo
        propagator = UncertaintyPropagator()
        result = propagator.monte_carlo_uncertainty(
            mv_true=25.0, rms_true=1.5, theta_true=35.0, n_samples=args.n_mc
        )
        if result["status"] == "success":
            print("\nResultado Monte Carlo:")
            print(f"  mv = {result['mv_mean']:.3f} ± {result['mv_std']:.3f}%")
            print(f"  RMSE = {result['rmse']:.3f}%")

    elif args.mode == "quality_map":
        # Solo mapa de calidad
        propagator = UncertaintyPropagator()
        quality_map = propagator.predict_quality_map(
            output_path=f"{args.output_dir}/quality_map.png"
        )
        print(
            f"\nMapa guardado. Alta calidad: {quality_map['statistics']['high_quality_fraction'] * 100:.1f}%"
        )

    print("\n✅ ANÁLISIS COMPLETADO\n")
