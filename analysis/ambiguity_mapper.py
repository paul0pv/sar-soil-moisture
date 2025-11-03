"""
Mapeador de Ambigüedad en Inversión SAR

Caracteriza explícitamente las regiones del espacio paramétrico donde
múltiples combinaciones (mv, rms) producen el mismo σ⁰ observado.

Esto NO ha sido hecho sistemáticamente en la literatura. Los estudios previos
(Baghdadi 2012, Ettalbi 2023) entrenan redes sin analizar formalmente
dónde y por qué la inversión falla.

Métodos:
- Análisis de iso-contornos de σ⁰
- Conteo de soluciones múltiples
- Mapeo de regiones de unicidad vs. ambigüedad
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import fsolve
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models import IEM_Model
from core.constants import VALID_DOMAIN, QUALITY, PLOT
from core.utils import linear_to_dB, dB_to_linear


class AmbiguityMapper:
    """
    Mapea ambigüedad en la inversión σ⁰ → (mv, rms).

    Para un σ⁰ observado dado, identifica:
    1. Número de soluciones (mv, rms) posibles
    2. Separación entre soluciones
    3. Incertidumbre esperada según nivel de ambigüedad
    """

    def __init__(self, model: IEM_Model = None):
        self.model = model or IEM_Model()
        self.response_surfaces = {}  # Cache de superficies calculadas

    def build_response_surface(
        self,
        theta: float,
        polarization: str = "VV",
        mv_range: tuple = (5, 45, 100),
        rms_range: tuple = (0.5, 3.0, 80),
    ) -> dict:
        """
        Construye superficie de respuesta σ⁰(mv, rms) para θ fijo.

        Returns:
            Dict con grids y función interpoladora
        """
        key = (theta, polarization)

        if key in self.response_surfaces:
            return self.response_surfaces[key]

        mv_grid = np.linspace(*mv_range)
        rms_grid = np.linspace(*rms_range)
        MV, RMS = np.meshgrid(mv_grid, rms_grid)

        # Calcular σ⁰ en grilla
        SIGMA = np.zeros_like(MV)
        for i in range(len(rms_grid)):
            SIGMA[i, :] = self.model.compute_backscatter(
                mv_grid, rms_grid[i], theta, polarization
            )

        # Crear interpolador para inversión
        interpolator = RegularGridInterpolator(
            (rms_grid, mv_grid),
            SIGMA,
            method="cubic",
            bounds_error=False,
            fill_value=None,
        )

        surface = {
            "mv_grid": mv_grid,
            "rms_grid": rms_grid,
            "MV": MV,
            "RMS": RMS,
            "SIGMA": SIGMA,
            "interpolator": interpolator,
            "theta": theta,
            "polarization": polarization,
        }

        self.response_surfaces[key] = surface
        return surface

    def count_solutions(
        self,
        sigma0_target: float,
        theta: float,
        polarization: str = "VV",
        tolerance_dB: float = 0.5,
    ) -> dict:
        """
        Cuenta número de soluciones (mv, rms) que producen σ⁰ objetivo.

        Args:
            sigma0_target: σ⁰ observado (dB)
            theta: Ángulo de incidencia (°)
            polarization: Polarización
            tolerance_dB: Tolerancia para considerar "igual" (±0.5 dB típico de sensor)

        Returns:
            Dict con soluciones encontradas y métricas de ambigüedad
        """
        surface = self.build_response_surface(theta, polarization)

        # Encontrar puntos donde |σ⁰_sim - σ⁰_target| < tolerance
        diff = np.abs(surface["SIGMA"] - sigma0_target)
        solutions_mask = diff < tolerance_dB

        # Extraer coordenadas de soluciones
        solutions_idx = np.where(solutions_mask)
        solutions = []

        for i, j in zip(solutions_idx[0], solutions_idx[1]):
            rms_sol = surface["rms_grid"][i]
            mv_sol = surface["mv_grid"][j]
            sigma_sol = surface["SIGMA"][i, j]
            error = abs(sigma_sol - sigma0_target)

            solutions.append(
                {
                    "mv": float(mv_sol),
                    "rms": float(rms_sol),
                    "sigma0": float(sigma_sol),
                    "error_dB": float(error),
                }
            )

        n_solutions = len(solutions)

        # Calcular dispersión de soluciones (si hay múltiples)
        if n_solutions > 1:
            mv_values = np.array([s["mv"] for s in solutions])
            rms_values = np.array([s["rms"] for s in solutions])

            mv_spread = np.std(mv_values)
            rms_spread = np.std(rms_values)

            # Distancia máxima entre soluciones
            from scipy.spatial.distance import pdist

            coords = np.column_stack([mv_values, rms_values])
            max_distance = (
                np.max(pdist(coords, metric="euclidean")) if n_solutions > 1 else 0.0
            )
        else:
            mv_spread = 0.0
            rms_spread = 0.0
            max_distance = 0.0

        # Clasificar nivel de ambigüedad
        if n_solutions == 0:
            ambiguity_level = "no_solution"
        elif n_solutions == 1:
            ambiguity_level = "unique"
        elif n_solutions <= 5 and mv_spread < 5.0:
            ambiguity_level = "low"
        elif n_solutions <= 10 and mv_spread < 10.0:
            ambiguity_level = "medium"
        else:
            ambiguity_level = "high"

        return {
            "sigma0_target": sigma0_target,
            "n_solutions": n_solutions,
            "solutions": solutions,
            "mv_spread": float(mv_spread),
            "rms_spread": float(rms_spread),
            "max_distance": float(max_distance),
            "ambiguity_level": ambiguity_level,
            "tolerance_dB": tolerance_dB,
        }

    def map_ambiguity_field(
        self,
        theta: float,
        polarization: str = "VV",
        sigma0_range: tuple = (-25, -10, 30),
        tolerance_dB: float = 0.5,
    ) -> dict:
        """
        Mapea nivel de ambigüedad para rango de σ⁰ observados.

        Args:
            theta: Ángulo de incidencia
            polarization: Polarización
            sigma0_range: (min, max, n_points) para σ⁰
            tolerance_dB: Tolerancia del sensor

        Returns:
            Dict con mapa de ambigüedad
        """
        surface = self.build_response_surface(theta, polarization)
        sigma0_test_values = np.linspace(*sigma0_range)

        n_solutions_array = np.zeros_like(sigma0_test_values, dtype=int)
        mv_spread_array = np.zeros_like(sigma0_test_values)
        ambiguity_labels = []

        print(f"\nMapeando ambigüedad para θ={theta}°, {polarization}...")

        for i, sigma0_target in enumerate(sigma0_test_values):
            result = self.count_solutions(
                sigma0_target, theta, polarization, tolerance_dB
            )
            n_solutions_array[i] = result["n_solutions"]
            mv_spread_array[i] = result["mv_spread"]
            ambiguity_labels.append(result["ambiguity_level"])

        # Estadísticas
        unique_fraction = np.sum(n_solutions_array == 1) / len(n_solutions_array)
        ambiguous_fraction = np.sum(n_solutions_array > 1) / len(n_solutions_array)
        no_solution_fraction = np.sum(n_solutions_array == 0) / len(n_solutions_array)

        return {
            "sigma0_values": sigma0_test_values,
            "n_solutions": n_solutions_array,
            "mv_spread": mv_spread_array,
            "ambiguity_labels": ambiguity_labels,
            "statistics": {
                "unique_fraction": float(unique_fraction),
                "ambiguous_fraction": float(ambiguous_fraction),
                "no_solution_fraction": float(no_solution_fraction),
                "mean_n_solutions": float(
                    np.mean(n_solutions_array[n_solutions_array > 0])
                ),
            },
            "theta": theta,
            "polarization": polarization,
        }

    def visualize_ambiguity_regions(
        self,
        theta_values: list = [30, 35, 40, 45],
        polarization: str = "VV",
        output_dir: str = "analysis/outputs",
    ):
        """
        Genera visualizaciones de regiones de ambigüedad.

        Crea:
        1. Mapa de iso-contornos con multiplicidad de soluciones
        2. Gráfico de ambigüedad vs. σ⁰ observado
        3. Reporte de zonas problemáticas
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("VISUALIZACIÓN DE REGIONES DE AMBIGÜEDAD")
        print("=" * 70)

        for theta in theta_values:
            print(f"\nProcesando θ={theta}°...")

            # Construir superficie
            surface = self.build_response_surface(theta, polarization)

            # Mapear ambigüedad
            ambiguity_map = self.map_ambiguity_field(theta, polarization)

            # Figura de 3 paneles
            fig = plt.figure(figsize=(18, 5))

            # Panel 1: Superficie con iso-contornos
            ax1 = plt.subplot(1, 3, 1)
            contour_levels = np.linspace(
                np.min(surface["SIGMA"]), np.max(surface["SIGMA"]), 15
            )
            cs = ax1.contour(
                surface["MV"],
                surface["RMS"],
                surface["SIGMA"],
                levels=contour_levels,
                cmap="viridis",
                linewidths=1.5,
            )
            ax1.clabel(cs, inline=True, fontsize=8, fmt="%1.1f dB")

            # Resaltar regiones de alta ambigüedad (líneas muy juntas)
            # (espaciado entre contornos < umbral indica alta curvatura)
            ax1.set_xlabel("Humedad volumétrica (%)")
            ax1.set_ylabel("Rugosidad RMS (cm)")
            ax1.set_title(
                f"Iso-contornos σ⁰ @ θ={theta}°\n(Líneas juntas = alta ambigüedad)"
            )
            ax1.grid(True, alpha=0.3)

            # Panel 2: Número de soluciones vs. σ⁰
            ax2 = plt.subplot(1, 3, 2)
            ax2.plot(
                ambiguity_map["sigma0_values"], ambiguity_map["n_solutions"], "b-", lw=2
            )
            ax2.axhline(1, color="g", ls="--", label="Solución única")
            ax2.fill_between(
                ambiguity_map["sigma0_values"],
                0,
                ambiguity_map["n_solutions"],
                where=(ambiguity_map["n_solutions"] > 1),
                color="red",
                alpha=0.3,
                label="Ambiguo",
            )
            ax2.set_xlabel("σ⁰ observado (dB)")
            ax2.set_ylabel("Número de soluciones (mv, rms)")
            ax2.set_title(
                f"Multiplicidad de soluciones\n(tolerancia ±{ambiguity_map['statistics']['unique_fraction'] * 100:.0f}% únicos)"
            )
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Panel 3: Dispersión de mv vs. σ⁰
            ax3 = plt.subplot(1, 3, 3)
            ax3.plot(
                ambiguity_map["sigma0_values"], ambiguity_map["mv_spread"], "r-", lw=2
            )
            ax3.axhline
