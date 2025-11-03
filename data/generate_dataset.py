"""
Generación de Dataset Sintético para Entrenamiento de Red Neuronal

- Genera 100,000+ escenarios sintéticos
- Cubre espacio paramétrico completo (mv, rms, θ)
- Añade ruido realista del sensor
- Incluye variación textural del suelo
- Exporta a formato CSV para análisis posterior

Referencias:
- Baghdadi et al. (2012): Dataset de 268,110 muestras
- Ettalbi et al. (2023): Dataset multi-resolución
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models import IEM_Model, DielectricModel
from core.constants import VALID_DOMAIN, ANALYSIS, PRIORS
from core.utils import validate_parameters, print_parameter_summary


class SyntheticDatasetGenerator:
    """
    Generador de dataset sintético para entrenamiento de modelos de inversión.

    Características:
    - Muestreo estratificado del espacio paramétrico
    - Variación textural realista del suelo
    - Ruido gaussiano del sensor
    - Múltiples polarizaciones (VV, HV)
    - Validación automática de parámetros
    """

    def __init__(
        self,
        frequency: float = 5.405e9,
        polarizations: list = ["VV", "HV"],
        texture_distribution: str = "realistic",
    ):
        """
        Inicializa generador.

        Args:
            frequency: Frecuencia del radar (Hz)
            polarizations: Lista de polarizaciones a simular
            texture_distribution: 'realistic', 'uniform', o 'loam_only'
        """
        self.frequency = frequency
        self.polarizations = polarizations
        self.texture_distribution = texture_distribution

        # Definir distribución de texturas
        if texture_distribution == "realistic":
            # Distribución representativa de suelos agrícolas
            self.textures = [
                {"sand": 40, "clay": 30, "name": "loam", "prob": 0.60},
                {"sand": 20, "clay": 45, "name": "clay", "prob": 0.25},
                {"sand": 60, "clay": 20, "name": "sandy_loam", "prob": 0.15},
            ]
        elif texture_distribution == "uniform":
            # Distribución uniforme de texturas
            self.textures = [
                {"sand": 20, "clay": 40, "name": "clay", "prob": 0.33},
                {"sand": 40, "clay": 30, "name": "loam", "prob": 0.34},
                {"sand": 60, "clay": 20, "name": "sandy_loam", "prob": 0.33},
            ]
        elif texture_distribution == "loam_only":
            # Solo suelo franco (simplificado)
            self.textures = [{"sand": 40, "clay": 30, "name": "loam", "prob": 1.0}]
        else:
            raise ValueError(f"Distribución '{texture_distribution}' no reconocida")

        print(f"Generador inicializado:")
        print(f"  Frecuencia: {frequency / 1e9:.3f} GHz")
        print(f"  Polarizaciones: {polarizations}")
        print(f"  Texturas: {[t['name'] for t in self.textures]}")

    def generate_random_samples(
        self,
        n_samples: int,
        mv_range: tuple = None,
        rms_range: tuple = None,
        theta_range: tuple = None,
        sampling_strategy: str = "stratified",
    ) -> pd.DataFrame:
        """
        Genera muestras aleatorias del espacio paramétrico.

        Args:
            n_samples: Número de muestras a generar
            mv_range: (min, max) para humedad (%)
            rms_range: (min, max) para rugosidad (cm)
            theta_range: (min, max) para ángulo (°)
            sampling_strategy: 'uniform', 'stratified', o 'latin_hypercube'

        Returns:
            DataFrame con parámetros muestreados
        """
        # Usar rangos por defecto si no se especifican
        mv_range = mv_range or (VALID_DOMAIN.MV_MIN, VALID_DOMAIN.MV_MAX)
        rms_range = rms_range or (VALID_DOMAIN.RMS_MIN, VALID_DOMAIN.RMS_MAX)
        theta_range = theta_range or (VALID_DOMAIN.THETA_MIN, VALID_DOMAIN.THETA_MAX)

        if sampling_strategy == "uniform":
            # Muestreo uniforme aleatorio
            mv_samples = np.random.uniform(*mv_range, n_samples)
            rms_samples = np.random.uniform(*rms_range, n_samples)
            theta_samples = np.random.uniform(*theta_range, n_samples)

        elif sampling_strategy == "stratified":
            # Muestreo estratificado (más representativo)
            # Dividir espacio en estratos y muestrear uniformemente en cada uno
            n_strata = int(np.sqrt(n_samples))

            mv_bins = np.linspace(*mv_range, n_strata + 1)
            rms_bins = np.linspace(*rms_range, n_strata + 1)
            theta_bins = np.linspace(*theta_range, n_strata + 1)

            mv_samples = []
            rms_samples = []
            theta_samples = []

            samples_per_stratum = max(1, n_samples // (n_strata**3))

            for i in range(n_strata):
                for j in range(n_strata):
                    for k in range(n_strata):
                        mv_samples.extend(
                            np.random.uniform(
                                mv_bins[i], mv_bins[i + 1], samples_per_stratum
                            )
                        )
                        rms_samples.extend(
                            np.random.uniform(
                                rms_bins[j], rms_bins[j + 1], samples_per_stratum
                            )
                        )
                        theta_samples.extend(
                            np.random.uniform(
                                theta_bins[k], theta_bins[k + 1], samples_per_stratum
                            )
                        )

            # Truncar al tamaño deseado
            mv_samples = np.array(mv_samples[:n_samples])
            rms_samples = np.array(rms_samples[:n_samples])
            theta_samples = np.array(theta_samples[:n_samples])

        elif sampling_strategy == "latin_hypercube":
            # Latin Hypercube Sampling (mejor cobertura del espacio)
            from scipy.stats import qmc

            sampler = qmc.LatinHypercube(d=3, seed=42)
            samples = sampler.random(n=n_samples)

            mv_samples = samples[:, 0] * (mv_range[1] - mv_range[0]) + mv_range[0]
            rms_samples = samples[:, 1] * (rms_range[1] - rms_range[0]) + rms_range[0]
            theta_samples = (
                samples[:, 2] * (theta_range[1] - theta_range[0]) + theta_range[0]
            )

        else:
            raise ValueError(f"Estrategia '{sampling_strategy}' no reconocida")

        # Muestrear texturas según probabilidades
        texture_probs = [t["prob"] for t in self.textures]
        texture_indices = np.random.choice(
            len(self.textures), size=len(mv_samples), p=texture_probs
        )

        sand_samples = np.array([self.textures[i]["sand"] for i in texture_indices])
        clay_samples = np.array([self.textures[i]["clay"] for i in texture_indices])
        texture_names = np.array([self.textures[i]["name"] for i in texture_indices])

        # Crear DataFrame
        df = pd.DataFrame(
            {
                "mv": mv_samples,
                "rms": rms_samples,
                "theta": theta_samples,
                "sand": sand_samples,
                "clay": clay_samples,
                "texture": texture_names,
            }
        )

        return df

    def simulate_backscatter(
        self,
        params_df: pd.DataFrame,
        add_noise: bool = True,
        noise_std_dB: float = ANALYSIS.SENSOR_NOISE_STD,
    ) -> pd.DataFrame:
        """
        Simula coeficientes de retrodispersión para parámetros dados.

        Args:
            params_df: DataFrame con columnas [mv, rms, theta, sand, clay]
            add_noise: Si añadir ruido gaussiano del sensor
            noise_std_dB: Desviación estándar del ruido (dB)

        Returns:
            DataFrame expandido con columnas sigma0 por polarización
        """
        results = []

        # Agrupar por textura para reutilizar modelo
        for texture_name in params_df["texture"].unique():
            mask = params_df["texture"] == texture_name
            subset = params_df[mask]

            # Inicializar modelo con textura correspondiente
            sand = subset["sand"].iloc[0]
            clay = subset["clay"].iloc[0]
            model = IEM_Model(frequency=self.frequency, sand_pct=sand, clay_pct=clay)

            # Simular cada polarización
            for pol in self.polarizations:
                for idx, row in tqdm(
                    subset.iterrows(),
                    total=len(subset),
                    desc=f"Simulando {texture_name} - {pol}",
                    leave=False,
                ):
                    # Calcular σ⁰
                    sigma0 = model.compute_backscatter(
                        mv=row["mv"],
                        rms_cm=row["rms"],
                        theta_deg=row["theta"],
                        polarization=pol,
                    )

                    # Añadir ruido
                    if add_noise:
                        sigma0 += np.random.normal(0, noise_std_dB)

                    # Guardar resultado
                    results.append(
                        {
                            "sample_id": idx,
                            "mv": row["mv"],
                            "rms": row["rms"],
                            "theta": row["theta"],
                            "sand": row["sand"],
                            "clay": row["clay"],
                            "texture": row["texture"],
                            "polarization": pol,
                            "sigma0_dB": sigma0,
                            "noise_added": add_noise,
                            "noise_std": noise_std_dB if add_noise else 0.0,
                        }
                    )

        return pd.DataFrame(results)

    def generate_complete_dataset(
        self,
        n_samples: int = ANALYSIS.N_SAMPLES_TRAINING,
        output_path: str = "synthetic_dataset.csv",
        sampling_strategy: str = "stratified",
        add_noise: bool = True,
        save_summary: bool = True,
    ) -> pd.DataFrame:
        """
        Pipeline completo: muestreo + simulación + guardado.

        Args:
            n_samples: Número de configuraciones únicas a muestrear
            output_path: Ruta del archivo CSV de salida
            sampling_strategy: Estrategia de muestreo
            add_noise: Si añadir ruido del sensor
            save_summary: Si guardar archivo de resumen estadístico

        Returns:
            DataFrame completo con todas las simulaciones
        """
        print("\n" + "=" * 70)
        print("GENERACIÓN DE DATASET SINTÉTICO")
        print("=" * 70)

        # Paso 1: Muestreo de parámetros
        print(f"\n[1/3] Muestreando {n_samples} configuraciones...")
        params_df = self.generate_random_samples(
            n_samples=n_samples, sampling_strategy=sampling_strategy
        )

        print(f"  Rangos muestreados:")
        print(f"    mv: [{params_df['mv'].min():.1f}, {params_df['mv'].max():.1f}] %")
        print(
            f"    rms: [{params_df['rms'].min():.2f}, {params_df['rms'].max():.2f}] cm"
        )
        print(
            f"    θ: [{params_df['theta'].min():.1f}, {params_df['theta'].max():.1f}] °"
        )

        # Paso 2: Simulación de retrodispersión
        print(
            f"\n[2/3] Simulando retrodispersión ({len(self.polarizations)} polarizaciones)..."
        )
        dataset = self.simulate_backscatter(params_df=params_df, add_noise=add_noise)

        print(f"  Total de muestras generadas: {len(dataset)}")
        print(f"  Tamaño estimado: {len(dataset) * 100 / 1e6:.1f} MB")

        # Paso 3: Guardado
        print(f"\n[3/3] Guardando dataset en {output_path}...")
        dataset.to_csv(output_path, index=False)

        # Resumen estadístico
        if save_summary:
            summary_path = output_path.replace(".csv", "_summary.txt")
            with open(summary_path, "w") as f:
                f.write("=" * 70 + "\n")
                f.write("RESUMEN DEL DATASET SINTÉTICO\n")
                f.write("=" * 70 + "\n\n")

                f.write(f"Archivo: {output_path}\n")
                f.write(f"Número de muestras: {len(dataset)}\n")
                f.write(f"Polarizaciones: {self.polarizations}\n")
                f.write(
                    f"Ruido añadido: {add_noise} (σ={ANALYSIS.SENSOR_NOISE_STD} dB)\n\n"
                )

                f.write("Estadísticas por variable:\n")
                f.write("-" * 70 + "\n")
                for col in ["mv", "rms", "theta", "sigma0_dB"]:
                    if col in dataset.columns:
                        f.write(f"{col}:\n")
                        f.write(f"  Min: {dataset[col].min():.3f}\n")
                        f.write(f"  Max: {dataset[col].max():.3f}\n")
                        f.write(f"  Mean: {dataset[col].mean():.3f}\n")
                        f.write(f"  Std: {dataset[col].std():.3f}\n\n")

                f.write("Distribución de texturas:\n")
                f.write("-" * 70 + "\n")
                texture_counts = dataset["texture"].value_counts()
                for texture, count in texture_counts.items():
                    f.write(
                        f"  {texture}: {count} ({count / len(dataset) * 100:.1f}%)\n"
                    )

            print(f"  Resumen guardado en {summary_path}")

        print("\n" + "=" * 70)
        print("✅ GENERACIÓN COMPLETADA")
        print("=" * 70)

        return dataset


def generate_training_and_validation_sets(
    n_training: int = ANALYSIS.N_SAMPLES_TRAINING,
    n_validation: int = ANALYSIS.N_SAMPLES_VALIDATION,
    output_dir: str = "data",
    seed: int = 42,
):
    """
    Genera datasets de entrenamiento y validación separados.

    Args:
        n_training: Muestras para entrenamiento
        n_validation: Muestras para validación
        output_dir: Directorio de salida
        seed: Semilla aleatoria para reproducibilidad
    """
    np.random.seed(seed)

    # Crear directorio si no existe
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generador
    generator = SyntheticDatasetGenerator(
        frequency=5.405e9, polarizations=["VV", "HV"], texture_distribution="realistic"
    )

    # Training set
    print("\n" + "█" * 70)
    print("GENERANDO DATASET DE ENTRENAMIENTO")
    print("█" * 70)
    train_df = generator.generate_complete_dataset(
        n_samples=n_training,
        output_path=str(output_path / "training_dataset.csv"),
        sampling_strategy="stratified",
        add_noise=True,
    )

    # Validation set (sin ruido para métricas limpias)
    print("\n" + "█" * 70)
    print("GENERANDO DATASET DE VALIDACIÓN")
    print("█" * 70)
    validation_df = generator.generate_complete_dataset(
        n_samples=n_validation,
        output_path=str(output_path / "validation_dataset.csv"),
        sampling_strategy="latin_hypercube",
        add_noise=False,  # Sin ruido para validación limpia
    )

    print("\n" + "█" * 70)
    print("DATASETS GENERADOS EXITOSAMENTE")
    print("█" * 70)
    print(
        f"Training:   {len(train_df)} muestras → {output_path / 'training_dataset.csv'}"
    )
    print(
        f"Validation: {len(validation_df)} muestras → {output_path / 'validation_dataset.csv'}"
    )

    return train_df, validation_df


def generate_scenario_specific_datasets():
    """
    Genera datasets para diferentes escenarios de información a priori.

    Implementa los 4 casos de Baghdadi et al. (2012):
    1. Sin información previa
    2. Rango de humedad conocido (seco vs húmedo)
    3. Rango de rugosidad conocido
    4. Ambos rangos conocidos
    """
    output_dir = Path("data/scenarios")
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = SyntheticDatasetGenerator()

    scenarios = {
        "no_prior": PRIORS.NO_PRIOR,
        "mv_dry_wet": PRIORS.MV_RANGE_DRY,
        "mv_very_wet": PRIORS.MV_RANGE_WET,
        "rms_smooth": PRIORS.RMS_RANGE_SMOOTH,
        "rms_rough": PRIORS.RMS_RANGE_ROUGH,
    }

    datasets = {}

    for scenario_name, config in scenarios.items():
        print(f"\n{'=' * 70}")
        print(f"Generando escenario: {scenario_name}")
        print(f"Descripción: {config['description']}")
        print(f"{'=' * 70}")

        # Generar dataset específico para este escenario
        params_df = generator.generate_random_samples(
            n_samples=20000,
            mv_range=config["mv_range"],
            rms_range=config.get(
                "rms_range", (VALID_DOMAIN.RMS_MIN, VALID_DOMAIN.RMS_MAX)
            ),
            sampling_strategy="stratified",
        )

        dataset = generator.simulate_backscatter(params_df, add_noise=True)

        # Guardar
        output_path = output_dir / f"{scenario_name}_dataset.csv"
        dataset.to_csv(output_path, index=False)

        datasets[scenario_name] = dataset

        print(f"✅ Guardado: {output_path} ({len(dataset)} muestras)")

    print("\n" + "█" * 70)
    print("TODOS LOS ESCENARIOS GENERADOS")
    print("█" * 70)

    return datasets


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generador de dataset sintético IEM-B")
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["standard", "scenarios", "both"],
        help="Modo de generación",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=100000,
        help="Número de muestras de entrenamiento",
    )
    parser.add_argument(
        "--n_val", type=int, default=20000, help="Número de muestras de validación"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data", help="Directorio de salida"
    )

    args = parser.parse_args()

    if args.mode in ["standard", "both"]:
        print("\n" + "█" * 70)
        print("MODO: GENERACIÓN ESTÁNDAR (Training + Validation)")
        print("█" * 70)
        generate_training_and_validation_sets(
            n_training=args.n_train, n_validation=args.n_val, output_dir=args.output_dir
        )

    if args.mode in ["scenarios", "both"]:
        print("\n" + "█" * 70)
        print("MODO: GENERACIÓN POR ESCENARIOS (A Priori Information)")
        print("█" * 70)
        generate_scenario_specific_datasets()

    print("\n✅ PROCESO COMPLETADO\n")
