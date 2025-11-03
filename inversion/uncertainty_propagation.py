import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from models import IEM_Model


class UncertaintyAnalyzer:
    """
    Cuantifica incertidumbres en estimación de humedad mediante:
    1. Propagación analítica de errores
    2. Análisis Monte Carlo
    3. Intervalos de confianza
    """

    def __init__(self, model):
        self.model = model

    def compute_jacobian(self, mv, rms, theta, delta=1e-5):
        """
        Calcula matriz Jacobiana de sensibilidades
        J = [∂σ⁰/∂mv, ∂σ⁰/∂rms, ∂σ⁰/∂θ]
        """
        sigma0_center = self.model.compute_backscatter(mv, rms, theta, "VV")

        # Derivadas numéricas
        dsigma_dmv = (
            self.model.compute_backscatter(mv + delta, rms, theta, "VV")
            - self.model.compute_backscatter(mv - delta, rms, theta, "VV")
        ) / (2 * delta)

        dsigma_drms = (
            self.model.compute_backscatter(mv, rms + delta, theta, "VV")
            - self.model.compute_backscatter(mv, rms - delta, theta, "VV")
        ) / (2 * delta)

        dsigma_dtheta = (
            self.model.compute_backscatter(mv, rms, theta + delta, "VV")
            - self.model.compute_backscatter(mv, rms, theta - delta, "VV")
        ) / (2 * delta)

        return np.array([dsigma_dmv, dsigma_drms, dsigma_dtheta])

    def propagate_error_analytical(
        self,
        mv,
        rms,
        theta,
        sigma_sensor=1.0,  # dB (error radiométrico)
        sigma_rms=0.3,  # cm (error medición rugosidad)
        sigma_theta=0.5,  # ° (error geolocalización)
    ):
        """
        Propagación analítica usando fórmula de Taylor de primer orden:
        σ²_mv = (∂mv/∂σ⁰)² σ²_sensor + (∂mv/∂rms)² σ²_rms + (∂mv/∂θ)² σ²_theta
        """
        J = self.compute_jacobian(mv, rms, theta)

        # Invertir Jacobiano (asumiendo rms conocido)
        # dmv/dsigma = 1 / (∂σ⁰/∂mv)
        if abs(J[0]) < 1e-6:
            return {
                "mv_uncertainty": np.inf,
                "status": "singular",
                "reason": "Sensibilidad a mv es nula",
            }

        dmv_dsigma = 1.0 / J[0]
        dmv_drms = -J[1] / J[0]  # Regla de cadena
        dmv_dtheta = -J[2] / J[0]

        # Propagación cuadrática
        var_mv = (
            (dmv_dsigma * sigma_sensor) ** 2
            + (dmv_drms * sigma_rms) ** 2
            + (dmv_dtheta * sigma_theta) ** 2
        )

        sigma_mv = np.sqrt(var_mv)

        # Intervalos de confianza (95%)
        ci_95 = (mv - 1.96 * sigma_mv, mv + 1.96 * sigma_mv)

        # Desglose de contribuciones
        contrib_sensor = (dmv_dsigma * sigma_sensor) ** 2 / var_mv * 100
        contrib_rms = (dmv_drms * sigma_rms) ** 2 / var_mv * 100
        contrib_theta = (dmv_dtheta * sigma_theta) ** 2 / var_mv * 100

        return {
            "mv_uncertainty": float(sigma_mv),
            "ci_95": ci_95,
            "relative_uncertainty": float(sigma_mv / mv * 100),
            "contributions": {
                "sensor_noise": contrib_sensor,
                "roughness_error": contrib_rms,
                "geometry_error": contrib_theta,
            },
            "jacobian": {
                "dsigma_dmv": J[0],
                "dsigma_drms": J[1],
                "dsigma_dtheta": J[2],
            },
            "status": "success",
        }

    def monte_carlo_uncertainty(
        self,
        mv_true,
        rms_true,
        theta_true,
        sigma_sensor=1.0,
        sigma_rms=0.3,
        sigma_theta=0.5,
        n_samples=10000,
        inversion_model=None,  # Red neuronal entrenada
    ):
        """
        Análisis Monte Carlo: Perturba entradas y evalúa distribución de salidas
        """
        # Generar perturbaciones
        sigma0_true = self.model.compute_backscatter(
            mv_true, rms_true, theta_true, "VV"
        )

        sigma0_samples = sigma0_true + np.random.normal(0, sigma_sensor, n_samples)
        rms_samples = rms_true + np.random.normal(0, sigma_rms, n_samples)
        theta_samples = theta_true + np.random.normal(0, sigma_theta, n_samples)

        # Caso 1: Si tenemos modelo de inversión, usarlo
        if inversion_model is not None:
            # (Implementar inversión con red neuronal)
            mv_samples = inversion_model.predict(
                sigma0_samples, theta_samples, rms_samples
            )
        else:
            # Caso 2: Inversión directa (solo si rms conocido exactamente)
            # Esto es un placeholder - requiere solver numérico
            mv_samples = np.zeros(n_samples)
            for i in range(n_samples):
                # Buscar mv que produce sigma0_samples[i]
                mv_samples[i] = self._invert_sigma0(
                    sigma0_samples[i], rms_samples[i], theta_samples[i]
                )

        # Estadísticas de la distribución
        mv_mean = np.mean(mv_samples)
        mv_std = np.std(mv_samples)
        mv_median = np.median(mv_samples)
        mv_p05 = np.percentile(mv_samples, 2.5)
        mv_p95 = np.percentile(mv_samples, 97.5)

        return {
            "mv_mean": float(mv_mean),
            "mv_std": float(mv_std),
            "mv_median": float(mv_median),
            "ci_95_mc": (float(mv_p05), float(mv_p95)),
            "samples": mv_samples,
            "bias_mc": float(mv_mean - mv_true),
            "rmse_mc": float(np.sqrt(np.mean((mv_samples - mv_true) ** 2))),
        }

    def _invert_sigma0(self, sigma0_obs, rms, theta, mv_range=(5, 45)):
        """
        Inversión numérica: Encuentra mv que minimiza |σ⁰_sim - σ⁰_obs|
        """
        from scipy.optimize import minimize_scalar

        def objective(mv):
            sigma0_sim = self.model.compute_backscatter(mv, rms, theta, "VV")
            return (sigma0_sim - sigma0_obs) ** 2

        result = minimize_scalar(objective, bounds=mv_range, method="bounded")
        return result.x if result.success else np.nan

    def characterize_uncertainty_map(
        self,
        mv_range=(5, 45, 50),
        rms_range=(0.5, 2.5, 40),
        theta_values=[30, 35, 40, 45],
    ):
        """
        Genera mapas de incertidumbre esperada para todo el espacio paramétrico
        """
        mv_grid = np.linspace(*mv_range)
        rms_grid = np.linspace(*rms_range)
        MV, RMS = np.meshgrid(mv_grid, rms_grid)

        results = {}

        for theta in theta_values:
            print(f"Procesando θ={theta}°...")

            # Calcular incertidumbre en cada punto
            UNCERTAINTY = np.zeros_like(MV)
            QUALITY_FLAG = np.zeros_like(MV, dtype=int)

            for i in range(len(rms_grid)):
                for j in range(len(mv_grid)):
                    unc_data = self.propagate_error_analytical(
                        mv_grid[j], rms_grid[i], theta
                    )

                    if unc_data["status"] == "success":
                        UNCERTAINTY[i, j] = unc_data["mv_uncertainty"]

                        # Clasificar calidad
                        if unc_data["mv_uncertainty"] < 3.0:
                            QUALITY_FLAG[i, j] = 3  # Alta
                        elif unc_data["mv_uncertainty"] < 5.0:
                            QUALITY_FLAG[i, j] = 2  # Media
                        elif unc_data["mv_uncertainty"] < 8.0:
                            QUALITY_FLAG[i, j] = 1  # Baja
                        else:
                            QUALITY_FLAG[i, j] = 0  # No confiable
                    else:
                        UNCERTAINTY[i, j] = np.nan
                        QUALITY_FLAG[i, j] = 0

            results[theta] = {
                "uncertainty_map": UNCERTAINTY,
                "quality_map": QUALITY_FLAG,
                "high_quality_fraction": np.sum(QUALITY_FLAG == 3) / QUALITY_FLAG.size,
            }

            # Visualización
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Panel 1: Mapa de incertidumbre
            im1 = axes[0].contourf(
                MV, RMS, UNCERTAINTY, levels=[0, 3, 5, 8, 12, 20], cmap="RdYlGn_r"
            )
            axes[0].set_title(f"Incertidumbre esperada σ_mv @ θ={theta}°")
            axes[0].set_xlabel("Humedad (%)")
            axes[0].set_ylabel("Rugosidad (cm)")
            plt.colorbar(im1, ax=axes[0], label="σ_mv (%)")

            # Panel 2: Mapa de calidad
            im2 = axes[1].contourf(
                MV,
                RMS,
                QUALITY_FLAG,
                levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
                colors=["darkred", "orange", "yellow", "green"],
            )
            axes[1].set_title(f"Flags de calidad @ θ={theta}°")
            axes[1].set_xlabel("Humedad (%)")
            axes[1].set_ylabel("Rugosidad (cm)")
            cbar = plt.colorbar(im2, ax=axes[1], ticks=[0, 1, 2, 3])
            cbar.ax.set_yticklabels(["No conf.", "Baja", "Media", "Alta"])

            plt.tight_layout()
            plt.savefig(f"uncertainty_map_theta{theta}.png", dpi=150)
            plt.close()

            print(
                f"  Fracción alta calidad: {results[theta]['high_quality_fraction'] * 100:.1f}%"
            )

        return results


# Ejemplo de uso
if __name__ == "__main__":
    model = IEM_Model()
    analyzer = UncertaintyAnalyzer(model)

    # Caso de estudio
    mv_test = 25.0  # %
    rms_test = 1.5  # cm
    theta_test = 35.0  # °

    print("=" * 60)
    print("ANÁLISIS DE INCERTIDUMBRE - CASO DE ESTUDIO")
    print("=" * 60)
    print(f"Condiciones: mv={mv_test}%, rms={rms_test}cm, θ={theta_test}°")

    # Propagación analítica
    result = analyzer.propagate_error_analytical(mv_test, rms_test, theta_test)

    print(f"\nPropagación analítica:")
    print(
        f"  σ_mv = {result['mv_uncertainty']:.3f}% ({result['relative_uncertainty']:.1f}% relativo)"
    )
    print(f"  IC 95%: [{result['ci_95'][0]:.2f}, {result['ci_95'][1]:.2f}]%")
    print(f"  Contribuciones:")
    print(f"    Ruido sensor: {result['contributions']['sensor_noise']:.1f}%")
    print(f"    Error rugosidad: {result['contributions']['roughness_error']:.1f}%")
    print(f"    Error geometría: {result['contributions']['geometry_error']:.1f}%")

    # Monte Carlo
    print("\nEjecutando simulación Monte Carlo (10,000 muestras)...")
    mc_result = analyzer.monte_carlo_uncertainty(mv_test, rms_test, theta_test)

    print(f"  mv_estimado = {mc_result['mv_mean']:.3f} ± {mc_result['mv_std']:.3f}%")
    print(
        f"  IC 95% (MC): [{mc_result['ci_95_mc'][0]:.2f}, {mc_result['ci_95_mc'][1]:.2f}]%"
    )
    print(f"  Sesgo: {mc_result['bias_mc']:.3f}%")
    print(f"  RMSE: {mc_result['rmse_mc']:.3f}%")

    # Mapas completos
    print("\nGenerando mapas de incertidumbre...")
    maps = analyzer.characterize_uncertainty_map()
