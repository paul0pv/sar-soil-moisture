# generate_dataset.py

import numpy as np
import pandas as pd
from models import IEM_Model
from tqdm import tqdm


def generate_training_data(
    n_samples=100000,
    mv_range=(5, 45),
    rms_range=(0.5, 2.5),  # Rango invertible
    theta_range=(25, 45),
    polarizations=["VV", "HV"],
    add_noise=True,
    noise_std_dB=1.0,
):
    """
    Genera dataset sintético para entrenamiento de red neuronal
    """
    # Muestreo aleatorio estratificado
    mv_samples = np.random.uniform(*mv_range, n_samples)
    rms_samples = np.random.uniform(*rms_range, n_samples)
    theta_samples = np.random.uniform(*theta_range, n_samples)

    # Variación textural (3 clases)
    textures = [
        (40, 30, "loam"),  # 60%
        (20, 45, "clay"),  # 30%
        (60, 20, "sandy_loam"),  # 10%
    ]

    data = []
    model = IEM_Model()

    for i in tqdm(range(n_samples), desc="Generating samples"):
        # Seleccionar textura aleatoriamente
        sand, clay, tex_name = textures[np.random.choice(3, p=[0.6, 0.3, 0.1])]
        model.dielectric = DielectricModel(sand, clay)

        for pol in polarizations:
            sigma0 = model.compute_backscatter(
                mv_samples[i], rms_samples[i], theta_samples[i], pol
            )

            # Añadir ruido realista
            if add_noise:
                sigma0 += np.random.normal(0, noise_std_dB)

            data.append(
                {
                    "mv": mv_samples[i],
                    "rms": rms_samples[i],
                    "theta": theta_samples[i],
                    "sand": sand,
                    "clay": clay,
                    "texture": tex_name,
                    "polarization": pol,
                    "sigma0_dB": sigma0,
                }
            )

    df = pd.DataFrame(data)
    df.to_csv("synthetic_dataset_100k.csv", index=False)

    print(f"\nDataset generado:")
    print(f"  Total samples: {len(df)}")
    print(f"  Polarizations: {polarizations}")
    print(f"  Mv range: {df['mv'].min():.1f}-{df['mv'].max():.1f}%")
    print(f"  Rms range: {df['rms'].min():.2f}-{df['rms'].max():.2f} cm")

    return df
