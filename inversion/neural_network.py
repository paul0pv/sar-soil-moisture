import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim


class SoilMoistureInverter(nn.Module):
    """
    Red neuronal para inversión σ⁰ → mv

    Arquitectura basada en Baghdadi et al. (2012):
    - Entrada: [σ⁰_VV, σ⁰_HV, θ, (opcional: rms_prior)]
    - Hidden: 2 capas × 20 neuronas
    - Salida: [mv_estimado, (opcional: rms_estimado)]
    """

    def __init__(self, n_inputs=3, n_outputs=1, hidden_size=20):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_outputs),
        )

    def forward(self, x):
        return self.network(x)


def train_inversion_model(
    dataset_path="synthetic_dataset_100k.csv",
    scenarios={
        "no_prior": {"inputs": ["sigma0_VV", "sigma0_HV", "theta"], "outputs": ["mv"]},
        "rms_known": {
            "inputs": ["sigma0_VV", "sigma0_HV", "theta", "rms"],
            "outputs": ["mv"],
        },
        "mv_range": {
            "inputs": ["sigma0_VV", "sigma0_HV", "theta", "mv_class"],
            "outputs": ["mv"],
        },
    },
    test_size=0.2,
    epochs=100,
    learning_rate=0.001,
    seed: int | None = 42,
    early_stopping: bool = True,
    es_patience: int = 20,
):
    """
    Entrena modelos para diferentes escenarios de información a priori
    """
    # Reproducibilidad
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar dataset
    df = pd.read_csv(dataset_path)

    # Pivotar polarizaciones a columnas
    df_pivot = df.pivot_table(
        index=["mv", "rms", "theta", "sand", "clay"],
        columns="polarization",
        values="sigma0_dB",
        aggfunc="first",
    ).reset_index()
    df_pivot.columns.name = None
    df_pivot.columns = ["mv", "rms", "theta", "sand", "clay", "sigma0_HV", "sigma0_VV"]

    # Añadir clase de humedad (seco/húmedo, umbral 30%)
    df_pivot["mv_class"] = (df_pivot["mv"] > 30).astype(int)

    results = {}

    for scenario_name, config in scenarios.items():
        print(f"\n{'=' * 60}")
        print(f"Entrenando escenario: {scenario_name}")
        print(f"{'=' * 60}")

        # Preparar datos
        X = df_pivot[config["inputs"]].values
        y = df_pivot[config["outputs"]].values

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Normalizar
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train)

        # Convertir a tensores
        X_train_t = torch.FloatTensor(X_train_scaled).float().to(device)
        y_train_t = torch.FloatTensor(y_train_scaled).float().to(device)
        X_test_t = torch.FloatTensor(X_test_scaled).float().to(device)

        # Modelo
        model = SoilMoistureInverter(
            n_inputs=X.shape[1], n_outputs=y.shape[1], hidden_size=20
        ).to(device)

        # Entrenamiento
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_losses = []
        best_val = np.inf
        epochs_no_improve = 0
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # Early stopping simple basado en pérdida de entrenamiento (sin validación explícita)
            # Nota: si se desea, separar un validation set del train y usarlo aquí.
            if early_stopping:
                current = loss.item()
                if current + 1e-6 < best_val:
                    best_val = current
                    epochs_no_improve = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= es_patience:
                        print(f"Early stopping @ epoch {epoch+1}, best loss={best_val:.4f}")
                        model.load_state_dict(best_state)
                        break

            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        # Evaluación
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_test_t).detach().cpu().numpy()
            y_pred = scaler_y.inverse_transform(y_pred_scaled)

        # Métricas
        rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y_test)))
        bias = float(np.mean(y_pred - y_test))

        results[scenario_name] = {
            "model": model,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "rmse": float(rmse),
            "mae": float(mae),
            "bias": float(bias),
            "train_losses": train_losses,
            "y_true": y_test,
            "y_pred": y_pred,
        }

        print(f"\nResultados:")
        print(f"  RMSE: {rmse:.3f}%")
        print(f"  MAE:  {mae:.3f}%")
        print(f"  Bias: {bias:.3f}%")

        ## Guardar modelo
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
                "config": config,
                "metrics": {"rmse": rmse, "mae": mae, "bias": bias},
            },
            f"trained_model_{scenario_name}.pth",
        )

    # Comparación entre escenarios
    print("\n" + "=" * 60)
    print("COMPARACIÓN DE ESCENARIOS")
    print("=" * 60)
    print(f"{'Escenario':<20} {'RMSE (%)':<12} {'MAE (%)':<12} {'Bias (%)':<12}")
    print("-" * 60)
    for name, data in results.items():
        print(
            f"{name:<20} {data['rmse']:<12.3f} {data['mae']:<12.3f} {data['bias']:<12.3f}"
        )

    # Visualización comparativa
    fig, axes = plt.subplots(1, len(scenarios), figsize=(15, 5))
    for idx, (name, data) in enumerate(results.items()):
        axes[idx].scatter(data["y_true"], data["y_pred"], alpha=0.3, s=4)
        axes[idx].plot([0, 50], [0, 50], "r--", lw=2)
        axes[idx].set_xlabel("Humedad real (%)")
        axes[idx].set_ylabel("Humedad estimada (%)")
        axes[idx].set_title(f"{name}\nRMSE={data['rmse']:.2f}%")
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("inversion_comparison.png", dpi=150)
    plt.close()

    return results


def monte_carlo_through_nn(
    model_bundle_path: str,
    mv_true: float,
    rms_true: float,
    theta_true: float,
    sigma0_vv_true_db: float,
    sigma0_hv_true_db: float,
    *,
    use_rms: bool = False,
    use_mv_class: bool = False,
    sigma_sensor_db: float = 1.0,
    sigma_rms: float = 0.3,
    sigma_theta: float = 0.5,
    n_samples: int = 5000,
    seed: int | None = 123,
):
    """
    Propagación Monte Carlo a través de la RN inversora. Genera N vectores de entrada
    perturbados con ruido del sensor y de parámetros geométricos y obtiene la distribución de mv_hat.
    - Asume que el bundle .pth incluye scalers y config.
    - sigma0_* y sigma_sensor_db en dB, coherentes con dataset de entrenamiento.
    """
    rng = np.random.default_rng(seed)
    bundle = torch.load(model_bundle_path, map_location="cpu")
    config = bundle["config"]
    scaler_X = bundle["scaler_X"]
    scaler_y = bundle["scaler_y"]
    model = SoilMoistureInverter(n_inputs=len(config["inputs"]), n_outputs=len(config["outputs"]), hidden_size=20)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    # Perturbaciones
    sigma0_vv = sigma0_vv_true_db + rng.normal(0.0, sigma_sensor_db, n_samples)
    sigma0_hv = sigma0_hv_true_db + rng.normal(0.0, sigma_sensor_db, n_samples)
    theta     = theta_true + rng.normal(0.0, sigma_theta, n_samples)
    rms       = rms_true   + rng.normal(0.0, sigma_rms, n_samples)
    mv_class  = (mv_true > 30).astype(int) if isinstance(mv_true, np.ndarray) else int(mv_true > 30)

    # Construir matriz de entrada según config
    cols = []
    for name in config["inputs"]:
        if name == "sigma0_VV":
            cols.append(sigma0_vv)
        elif name == "sigma0_HV":
            cols.append(sigma0_hv)
        elif name == "theta":
            cols.append(theta)
        elif name == "rms":
            cols.append(rms if use_rms else np.full(n_samples, rms_true))
        elif name == "mv_class":
            cols.append(np.full(n_samples, mv_class if use_mv_class else 0))
        else:
            raise ValueError(f"Entrada desconocida: {name}")
    X = np.vstack(cols).T

    # Escalado y predicción
    Xs = scaler_X.transform(X)
    with torch.no_grad():
        y_pred_s = model(torch.from_numpy(Xs).float()).detach().cpu().numpy()
    y_pred = scaler_y.inverse_transform(y_pred_s).reshape(-1)

    # Resumen
    mean = float(np.mean(y_pred))
    std  = float(np.std(y_pred))
    p025, p975 = np.percentile(y_pred, [2.5, 97.5])
    return {
        "mv_true": float(mv_true),
        "mv_mean": mean,
        "mv_std": std,
        "ci_95": (float(p025), float(p975)),
        "samples": y_pred,
        "n": int(n_samples),
    }
