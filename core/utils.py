"""
Utilidades comunes para el proyecto IEM-B

Funciones auxiliares para:
- Validación de inputs
- Conversiones de unidades
- Cálculos numéricos recurrentes
- Logging y warnings
"""

import numpy as np
import warnings
from typing import Union, Tuple, Dict, Optional
from core.constants import VALID_DOMAIN, WARNINGS, QUALITY

# ============================================================================
# VALIDACIÓN DE INPUTS
# ============================================================================


def validate_parameters(
    mv: Optional[Union[float, np.ndarray]] = None,
    rms: Optional[Union[float, np.ndarray]] = None,
    theta: Optional[Union[float, np.ndarray]] = None,
    frequency: Optional[float] = None,
    strict: bool = False,
) -> Dict[str, bool]:
    """
    Valida parámetros de entrada contra dominios de validez.

    Args:
        mv: Humedad volumétrica (%)
        rms: Rugosidad RMS (cm)
        theta: Ángulo de incidencia (grados)
        frequency: Frecuencia (Hz)
        strict: Si True, lanza excepciones; si False, solo warnings

    Returns:
        Dict con status de validación para cada parámetro

    Raises:
        ValueError: Si strict=True y parámetros fuera de rango
    """
    validation_results = {}
    issues = []

    # Validar humedad
    if mv is not None:
        mv_arr = np.atleast_1d(mv)
        checks = VALID_DOMAIN.is_valid(mv=np.mean(mv_arr))
        validation_results["mv"] = checks

        if not checks.get("mv_valid", True):
            msg = WARNINGS.out_of_range(
                "mv", np.mean(mv_arr), (VALID_DOMAIN.MV_MIN, VALID_DOMAIN.MV_MAX)
            )
            issues.append(msg)
            if not checks.get("mv_safe", True):
                issues.append(f"❌ mv fuera de límites absolutos del IEM")

    # Validar rugosidad
    if rms is not None:
        rms_arr = np.atleast_1d(rms)
        checks = VALID_DOMAIN.is_valid(rms=np.mean(rms_arr))
        validation_results["rms"] = checks

        if not checks.get("rms_high_quality", True):
            msg = WARNINGS.high_roughness(np.mean(rms_arr))
            issues.append(msg)

        if not checks.get("rms_valid", True):
            msg = WARNINGS.out_of_range(
                "rms", np.mean(rms_arr), (VALID_DOMAIN.RMS_MIN, VALID_DOMAIN.RMS_MAX)
            )
            issues.append(msg)

    # Validar ángulo
    if theta is not None:
        theta_arr = np.atleast_1d(theta)
        checks = VALID_DOMAIN.is_valid(theta=np.mean(theta_arr))
        validation_results["theta"] = checks

        if not checks.get("theta_valid", True):
            msg = WARNINGS.out_of_range(
                "theta",
                np.mean(theta_arr),
                (VALID_DOMAIN.THETA_MIN, VALID_DOMAIN.THETA_MAX),
            )
            issues.append(msg)

    # Validar frecuencia
    if frequency is not None:
        checks = VALID_DOMAIN.is_valid(frequency=frequency)
        validation_results["frequency"] = checks

        if not checks.get("freq_valid", True):
            msg = (
                f"⚠️  Frecuencia {frequency / 1e9:.2f} GHz fuera del rango validado "
                f"de Hallikainen (1.4-18 GHz). Modelo dieléctrico puede ser impreciso."
            )
            issues.append(msg)

    # Manejo de issues
    if issues:
        full_message = "\n".join(issues)
        if strict:
            raise ValueError(f"Parámetros inválidos:\n{full_message}")
        else:
            warnings.warn(full_message, UserWarning)

    validation_results["overall_valid"] = len(issues) == 0
    validation_results["n_issues"] = len(issues)

    return validation_results


# ============================================================================
# CONVERSIONES DE UNIDADES
# ============================================================================


def mv_percent_to_fraction(
    mv_percent: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Convierte humedad de % a fracción volumétrica (0-1)"""
    mv_percent = np.asarray(mv_percent)
    return np.where(mv_percent > 1.0, mv_percent / 100.0, mv_percent)


def mv_fraction_to_percent(
    mv_frac: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Convierte humedad de fracción (0-1) a %"""
    mv_frac = np.asarray(mv_frac)
    return np.where(mv_frac <= 1.0, mv_frac * 100.0, mv_frac)


def cm_to_m(length_cm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convierte centímetros a metros"""
    return np.asarray(length_cm) / 100.0


def m_to_cm(length_m: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convierte metros a centímetros"""
    return np.asarray(length_m) * 100.0


def linear_to_dB(
    sigma_linear: Union[float, np.ndarray], floor: float = 1e-20
) -> Union[float, np.ndarray]:
    """
    Convierte coeficiente de retrodispersión de lineal a dB.

    Args:
        sigma_linear: σ⁰ en unidades lineales
        floor: Valor mínimo para evitar log(0)

    Returns:
        σ⁰ en dB
    """
    sigma_linear = np.asarray(sigma_linear)
    sigma_clipped = np.where(sigma_linear <= floor, floor, sigma_linear)
    return 10.0 * np.log10(sigma_clipped)


def dB_to_linear(sigma_dB: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convierte σ⁰ de dB a unidades lineales"""
    return 10.0 ** (np.asarray(sigma_dB) / 10.0)


# ============================================================================
# CÁLCULOS NUMÉRICOS
# ============================================================================


def compute_numerical_derivative(
    func: callable, x: float, delta: float = 1e-5, method: str = "central"
) -> float:
    """
    Calcula derivada numérica de una función.

    Args:
        func: Función a derivar (debe aceptar escalar)
        x: Punto donde evaluar derivada
        delta: Paso para diferencias finitas
        method: 'forward', 'backward', o 'central'

    Returns:
        Aproximación numérica de f'(x)
    """
    if method == "central":
        return (func(x + delta) - func(x - delta)) / (2 * delta)
    elif method == "forward":
        return (func(x + delta) - func(x)) / delta
    elif method == "backward":
        return (func(x) - func(x - delta)) / delta
    else:
        raise ValueError(
            f"Método '{method}' no reconocido. Use 'forward', 'backward', o 'central'."
        )


def compute_gradient(
    field_2d: np.ndarray,
    dx: Union[float, np.ndarray],
    dy: Union[float, np.ndarray],
    axis: int,
) -> np.ndarray:
    """
    Calcula gradiente de un campo 2D usando diferencias centrales.

    Args:
        field_2d: Campo 2D (ej. superficie σ⁰(mv, rms))
        dx: Espaciado en dirección x (o array de coordenadas)
        dy: Espaciado en dirección y
        axis: 0 para ∂/∂y, 1 para ∂/∂x

    Returns:
        Gradiente del campo
    """
    return np.gradient(field_2d, dx if axis == 1 else dy, axis=axis)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula Root Mean Square Error"""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula Mean Absolute Error"""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula sesgo (diferencia promedio)"""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula coeficiente de determinación R²"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


# ============================================================================
# CLASIFICACIÓN DE CALIDAD
# ============================================================================


def classify_quality_from_uncertainty(uncertainty: float) -> Dict[str, Union[int, str]]:
    """
    Clasifica calidad de estimación basada en incertidumbre σ_mv.

    Args:
        uncertainty: Incertidumbre en humedad (%)

    Returns:
        Dict con flag numérico (0-3) y etiqueta descriptiva
    """
    if uncertainty < QUALITY.UNCERTAINTY_HIGH_QUALITY:
        return {"flag": 3, "label": "Alta", "color": "green"}
    elif uncertainty < QUALITY.UNCERTAINTY_MEDIUM_QUALITY:
        return {"flag": 2, "label": "Media", "color": "yellow"}
    elif uncertainty < QUALITY.UNCERTAINTY_LOW_QUALITY:
        return {"flag": 1, "label": "Baja", "color": "orange"}
    else:
        return {"flag": 0, "label": "No confiable", "color": "red"}


def classify_quality_from_sensitivity_ratio(ratio: float) -> Dict[str, Union[int, str]]:
    """
    Clasifica calidad basada en ratio de sensibilidades |∂σ⁰/∂mv| / |∂σ⁰/∂rms|.

    Args:
        ratio: Ratio de sensibilidades

    Returns:
        Dict con flag de calidad
    """
    if ratio > 5.0:
        return {"flag": 3, "label": "Excelente invertibilidad"}
    elif ratio > QUALITY.MIN_SENSITIVITY_RATIO:
        return {"flag": 2, "label": "Buena invertibilidad"}
    elif ratio > 1.0:
        return {"flag": 1, "label": "Invertibilidad marginal"}
    else:
        return {"flag": 0, "label": "No invertible"}


# ============================================================================
# LOGGING Y DIAGNÓSTICO
# ============================================================================


def print_parameter_summary(mv=None, rms=None, theta=None, sigma0=None):
    """Imprime resumen formateado de parámetros"""
    print("=" * 60)
    if mv is not None:
        mv_stats = f"mv: {np.mean(mv):.2f}%" + (
            f" (±{np.std(mv):.2f})" if np.size(mv) > 1 else ""
        )
        print(mv_stats)
    if rms is not None:
        rms_stats = f"rms: {np.mean(rms):.2f} cm" + (
            f" (±{np.std(rms):.2f})" if np.size(rms) > 1 else ""
        )
        print(rms_stats)
    if theta is not None:
        theta_stats = f"θ: {np.mean(theta):.2f}°" + (
            f" (±{np.std(theta):.2f})" if np.size(theta) > 1 else ""
        )
        print(theta_stats)
    if sigma0 is not None:
        sigma_stats = f"σ⁰: {np.mean(sigma0):.2f} dB" + (
            f" (±{np.std(sigma0):.2f})" if np.size(sigma0) > 1 else ""
        )
        print(sigma_stats)
    print("=" * 60)


def format_metrics_table(metrics_dict: Dict[str, Dict[str, float]]) -> str:
    """
    Formatea diccionario de métricas en tabla legible.

    Args:
        metrics_dict: {scenario_name: {metric_name: value}}

    Returns:
        String con tabla formateada
    """
    scenarios = list(metrics_dict.keys())
    if not scenarios:
        return "No hay métricas para mostrar"

    metric_names = list(metrics_dict[scenarios[0]].keys())

    # Header
    header = f"{'Escenario':<25}"
    for metric in metric_names:
        header += f"{metric.upper():<12}"

    # Rows
    rows = [header, "-" * len(header)]
    for scenario in scenarios:
        row = f"{scenario:<25}"
        for metric in metric_names:
            value = metrics_dict[scenario].get(metric, np.nan)
            row += f"{value:<12.3f}"
        rows.append(row)

    return "\n".join(rows)
