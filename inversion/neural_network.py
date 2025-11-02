# inversion/neural_network.py

import numpy as np
import matplotlib as plt
import pandas as pd
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
):
    """
    Entrena modelos para diferentes escenarios de información a priori
    """
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
        X_train_t = torch.FloatTensor(X_train_scaled)
        y_train_t = torch.FloatTensor(y_train_scaled)
        X_test_t = torch.FloatTensor(X_test_scaled)

        # Modelo
        model = SoilMoistureInverter(
            n_inputs=X.shape[1], n_outputs=y.shape[1], hidden_size=20
        )

        # Entrenamiento
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        # Evaluación
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_test_t).numpy()
            y_pred = scaler_y.inverse_transform(y_pred_scaled)

        # Métricas
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        mae = np.mean(np.abs(y_pred - y_test))
        bias = np.mean(y_pred - y_test)

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
        axes[idx].scatter(data["y_true"], data["y_pred"], alpha=0.3, s=1)
        axes[idx].plot([0, 50], [0, 50], "r--", lw=2)
        axes[idx].set_xlabel("Humedad real (%)")
        axes[idx].set_ylabel("Humedad estimada (%)")
        axes[idx].set_title(f"{name}\nRMSE={data['rmse']:.2f}%")
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("inversion_comparison.png", dpi=150)
    plt.close()

    return results
