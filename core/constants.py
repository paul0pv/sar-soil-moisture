"""
Constantes globales y configuraciones del proyecto IEM-B

Centraliza parámetros físicos, rangos de validez, y configuraciones
para facilitar mantenimiento y consistencia.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

# ============================================================================
# PARÁMETROS FÍSICOS FUNDAMENTALES
# ============================================================================


@dataclass(frozen=True)
class PhysicalConstants:
    """Constantes físicas inmutables"""

    SPEED_OF_LIGHT: float = 2.99792458e8  # m/s
    SENTINEL1_FREQUENCY: float = 5.405e9  # Hz (Banda C)
    ALOS2_FREQUENCY: float = 1.27e9  # Hz (Banda L)
    TERRASAR_FREQUENCY: float = 9.65e9  # Hz (Banda X)


PHYSICS = PhysicalConstants()

# ============================================================================
# RANGOS DE VALIDEZ DEL MODELO IEM-B
# ============================================================================


@dataclass
class ValidationDomain:
    """
    Dominios de validez según literatura:
    - Baghdadi et al. (2011): Calibración Lopt
    - Fung et al. (1992): Validez del IEM
    - Hallikainen et al. (1985): Modelo dieléctrico
    """

    # Humedad volumétrica (%)
    MV_MIN: float = 5.0
    MV_MAX: float = 45.0
    MV_SAFE_MIN: float = 0.5  # Límite absoluto IEM
    MV_SAFE_MAX: float = 50.0

    # Rugosidad superficial (cm)
    RMS_MIN: float = 0.5
    RMS_MAX: float = 2.5  # ⚠️ Reducido desde 3.0 (sensibilidad colapsa)
    RMS_SAFE_MIN: float = 0.3  # Límite absoluto IEM
    RMS_SAFE_MAX: float = 5.0
    RMS_HIGH_QUALITY_MAX: float = 2.0  # Umbral para inversión confiable

    # Ángulo de incidencia (grados)
    THETA_MIN: float = 25.0
    THETA_MAX: float = 45.0
    THETA_SENTINEL1_MIN: float = 29.0  # IW mode
    THETA_SENTINEL1_MAX: float = 46.0

    # Textura del suelo (%)
    SAND_MIN: float = 0.0
    SAND_MAX: float = 100.0
    CLAY_MIN: float = 0.0
    CLAY_MAX: float = 100.0

    # Frecuencia (Hz)
    FREQ_MIN: float = 1.4e9  # Hallikainen validado
    FREQ_MAX: float = 18.0e9

    def is_valid(
        self, mv=None, rms=None, theta=None, frequency=None
    ) -> Dict[str, bool]:
        """Valida si parámetros están dentro del dominio operacional"""
        checks = {}

        if mv is not None:
            checks["mv_valid"] = self.MV_MIN <= mv <= self.MV_MAX
            checks["mv_safe"] = self.MV_SAFE_MIN <= mv <= self.MV_SAFE_MAX

        if rms is not None:
            checks["rms_valid"] = self.RMS_MIN <= rms <= self.RMS_MAX
            checks["rms_safe"] = self.RMS_SAFE_MIN <= rms <= self.RMS_SAFE_MAX
            checks["rms_high_quality"] = rms <= self.RMS_HIGH_QUALITY_MAX

        if theta is not None:
            checks["theta_valid"] = self.THETA_MIN <= theta <= self.THETA_MAX
            checks["theta_sentinel1"] = (
                self.THETA_SENTINEL1_MIN <= theta <= self.THETA_SENTINEL1_MAX
            )

        if frequency is not None:
            checks["freq_valid"] = self.FREQ_MIN <= frequency <= self.FREQ_MAX

        return checks


VALID_DOMAIN = ValidationDomain()

# ============================================================================
# UMBRALES DE CALIDAD PARA INVERSIÓN
# ============================================================================


@dataclass
class QualityThresholds:
    """
    Umbrales para clasificar calidad de estimaciones
    Basado en análisis de sensibilidad (Objetivo 2)
    """

    # Incertidumbre en humedad (%)
    UNCERTAINTY_HIGH_QUALITY: float = 3.0  # σ_mv < 3% → Alta confianza
    UNCERTAINTY_MEDIUM_QUALITY: float = 5.0  # 3% < σ_mv < 5% → Media
    UNCERTAINTY_LOW_QUALITY: float = 8.0  # 5% < σ_mv < 8% → Baja
    # σ_mv > 8% → No confiable

    # Sensibilidad mínima para inversión (dB/%)
    MIN_SENSITIVITY_MV: float = 0.3  # |∂σ⁰/∂mv| > 0.3 dB/%

    # Ratio de sensibilidades (invertibilidad)
    MIN_SENSITIVITY_RATIO: float = 2.0  # |∂σ⁰/∂mv| / |∂σ⁰/∂rms| > 2

    # Número de condición máximo (matriz Jacobiana)
    MAX_CONDITION_NUMBER: float = 100.0

    # RMSE aceptable según literatura
    RMSE_EXCELLENT: float = 3.5  # Ettalbi et al. (2023)
    RMSE_GOOD: float = 5.0  # Yu et al. (2025)
    RMSE_ACCEPTABLE: float = 6.5  # Baghdadi et al. (2012)
    RMSE_POOR: float = 10.0  # Umbral operacional


QUALITY = QualityThresholds()

# ============================================================================
# CONFIGURACIÓN DE ANÁLISIS PARAMÉTRICO
# ============================================================================


@dataclass
class AnalysisConfig:
    """Configuración para generación de dataset y análisis"""

    # Dataset sintético (Objetivo 1)
    N_SAMPLES_TRAINING: int = 100000
    N_SAMPLES_VALIDATION: int = 20000

    # Grillas de análisis (Objetivo 2)
    MV_GRID_POINTS: int = 100
    RMS_GRID_POINTS: int = 50
    THETA_GRID_POINTS: int = 20

    # Ruido del sensor (dB)
    SENSOR_NOISE_STD: float = 1.0  # Sentinel-1 típico
    SENSOR_NOISE_MIN: float = 0.5
    SENSOR_NOISE_MAX: float = 1.5

    # Distribución de texturas (realista)
    TEXTURE_LOAM: Tuple[float, float, float] = (40, 30, 0.60)  # sand, clay, prob
    TEXTURE_CLAY: Tuple[float, float, float] = (20, 45, 0.25)
    TEXTURE_SANDY_LOAM: Tuple[float, float, float] = (60, 20, 0.15)

    # Monte Carlo
    N_MC_SAMPLES: int = 10000

    # Red neuronal
    NN_HIDDEN_SIZE: int = 20  # Baghdadi et al. (2012)
    NN_N_LAYERS: int = 2
    NN_LEARNING_RATE: float = 0.001
    NN_EPOCHS: int = 100
    NN_BATCH_SIZE: int = 256


ANALYSIS = AnalysisConfig()

# ============================================================================
# INFORMACIÓN A PRIORI (ESCENARIOS)
# ============================================================================


@dataclass
class PriorScenarios:
    """
    Escenarios de información a priori según Baghdadi et al. (2012)
    """

    NO_PRIOR = {
        "name": "no_prior",
        "description": "Sin información previa sobre mv o rms",
        "mv_range": (VALID_DOMAIN.MV_MIN, VALID_DOMAIN.MV_MAX),
        "rms_range": (VALID_DOMAIN.RMS_MIN, VALID_DOMAIN.RMS_MAX),
    }

    MV_RANGE_DRY = {
        "name": "mv_dry_wet",
        "description": "Suelos secos a húmedos (mv < 30%)",
        "mv_range": (5.0, 35.0),  # Overlap de 5%
        "rms_range": (VALID_DOMAIN.RMS_MIN, VALID_DOMAIN.RMS_MAX),
    }

    MV_RANGE_WET = {
        "name": "mv_very_wet",
        "description": "Suelos muy húmedos (mv > 25%)",
        "mv_range": (25.0, 45.0),
        "rms_range": (VALID_DOMAIN.RMS_MIN, VALID_DOMAIN.RMS_MAX),
    }

    RMS_KNOWN = {
        "name": "rms_known",
        "description": "Rugosidad conocida (medida o asumida)",
        "mv_range": (VALID_DOMAIN.MV_MIN, VALID_DOMAIN.MV_MAX),
        "rms_known": True,
    }

    RMS_RANGE_SMOOTH = {
        "name": "rms_smooth",
        "description": "Superficie lisa a moderada (rms < 1.5 cm)",
        "mv_range": (VALID_DOMAIN.MV_MIN, VALID_DOMAIN.MV_MAX),
        "rms_range": (0.5, 2.0),  # Overlap de 0.5 cm
    }

    RMS_RANGE_ROUGH = {
        "name": "rms_rough",
        "description": "Superficie moderada a rugosa (rms > 1.0 cm)",
        "mv_range": (VALID_DOMAIN.MV_MIN, VALID_DOMAIN.MV_MAX),
        "rms_range": (1.0, 2.5),
    }


PRIORS = PriorScenarios()

# ============================================================================
# MENSAJES DE WARNING
# ============================================================================


class WarningMessages:
    """Mensajes de advertencia estandarizados"""

    @staticmethod
    def out_of_range(param_name, value, valid_range):
        return (
            f"⚠️  {param_name}={value:.2f} fuera del rango validado "
            f"[{valid_range[0]:.2f}, {valid_range[1]:.2f}]. "
            f"Resultados pueden ser imprecisos."
        )

    @staticmethod
    def low_sensitivity(sensitivity_value):
        return (
            f"⚠️  Baja sensibilidad a humedad (∂σ⁰/∂mv = {sensitivity_value:.3f} dB/%). "
            f"Inversión puede ser ambigua."
        )

    @staticmethod
    def high_roughness(rms_value):
        return (
            f"⚠️  Rugosidad elevada (rms={rms_value:.2f} cm > {VALID_DOMAIN.RMS_HIGH_QUALITY_MAX} cm). "
            f"Sensibilidad a humedad degradada. RMSE esperado > 8%."
        )

    @staticmethod
    def singular_jacobian():
        return (
            "❌ Matriz Jacobiana singular o mal condicionada. "
            "Inversión no factible en este punto."
        )


WARNINGS = WarningMessages()

# ============================================================================
# POLARIZACIONES SOPORTADAS
# ============================================================================

SUPPORTED_POLARIZATIONS = ["VV", "HV", "VH", "HH"]
POLARIZATION_OFFSETS = {
    "VV": 0.0,
    "HV": -12.0,  # Aproximado VV - 12 dB
    "VH": -12.0,
    "HH": -1.5,  # Aproximado VV - 1.5 dB
}

# ============================================================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# ============================================================================


@dataclass
class PlotConfig:
    """Configuración estándar para figuras"""

    DPI: int = 150
    FIGSIZE_SINGLE: Tuple[float, float] = (8, 6)
    FIGSIZE_DOUBLE: Tuple[float, float] = (14, 6)
    FIGSIZE_QUAD: Tuple[float, float] = (14, 12)

    CMAP_SIGMA: str = "viridis"
    CMAP_SENSITIVITY: str = "RdYlGn"
    CMAP_QUALITY: str = "RdYlGn"
    CMAP_UNCERTAINTY: str = "RdYlGn_r"


PLOT = PlotConfig()

# ============================================================================
# METADATA DEL PROYECTO
# ============================================================================

PROJECT_METADATA = {
    "name": "IEM-B Soil Moisture Inversion",
    "version": "2.0.0",
    "description": "Simulación y análisis de estimación de humedad del suelo mediante SAR Banda C",
    "authors": ["Research Team"],
    "references": [
        "Fung et al. (1992) - IEEE TGRS - IEM fundamental",
        "Hallikainen et al. (1985) - IEEE TGRS - Modelo dieléctrico",
        "Baghdadi et al. (2011) - IEEE GRSL - Calibración IEM_B",
        "Baghdadi et al. (2012) - HESS - Inversión con redes neuronales",
    ],
    "keywords": [
        "SAR",
        "soil moisture",
        "IEM",
        "C-band",
        "Sentinel-1",
        "neural networks",
    ],
}
