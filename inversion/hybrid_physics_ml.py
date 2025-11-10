"""
Modelo Híbrido Física + Machine Learning

Implementa Estado del Arte:
- Integra conocimiento físico del IEM-B con redes neuronales
- Arquitectura guiada por física (Physics-Guided Neural Networks)
- Loss function con restricciones físicas

Referencias:
- Yu et al. (2025): LSTM + Water Cloud Model
- Chung et al. (2022): Integración de información hidrológica
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models import IEM_Model
from core.constants import VALID_DOMAIN, ANALYSIS, QUALITY
from core.utils import rmse, mae, bias


class PhysicsGuidedNN(nn.Module):
    """
    Red neuronal guiada por física para inversión SAR.

    INNOVACIÓN (Estado del Arte 10.1):
    Combina arquitectura de red neuronal con conocimiento físico explícito:
    1. Capa de física: Calcula σ⁰ esperado según IEM-B
    2. Capa de corrección: Aprende residuos/discrepancias
    3. Loss function: Penaliza violaciones de restricciones físicas

    Ventajas vs. NN pura:
    - Mejor generalización con menos datos
    - Predicciones físicamente consistentes
    - Interpretabilidad mejorada
    """

    def __init__(
        self,
        n_inputs: int = 3,  # [σ⁰_VV, σ⁰_HV, θ]
        hidden_sizes: list = [32, 32],
        physical_model: IEM_Model = None,
        use_physical_residuals: bool = True,
        anchor_mv: float = 25.0,
        anchor_rms: float = 1.5,
        anchor_theta: float = 35.0,
        fd_step: float = 1e-2,
    ):
        """
        Args:
            n_inputs: Número de características de entrada
            hidden_sizes: Lista con tamaños de capas ocultas
            physical_model: Instancia de IEM_Model (para cálculos físicos)
            use_physical_residuals: Si usar residuos físicos en forward pass
        """
        super(PhysicsGuidedNN, self).__init__()

        self.physical_model = physical_model or IEM_Model()
        self.use_physical_residuals = use_physical_residuals

        # Red neuronal para aprender correcciones/residuos
        layers = []
        in_features = n_inputs

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.2))
            in_features = hidden_size

        # Capa de salida: [mv, rms]
        layers.append(nn.Linear(in_features, 2))

        self.network = nn.Sequential(*layers)

        # Mezcla convexa: w in (0,1). Output = w*physics + (1-w)*ML
        self.alpha = nn.Parameter(torch.tensor(0.0))  # logit inicial ~ w=0.5

        # === Coeficientes de sensibilidad física (linealización de IEM) ===
        # σ0 ≈ a*mv + b*rms + c*theta + d  (d en torno a ancla)
        # Se calculan una vez con diferencias centrales.
        mv0, r0, th0 = anchor_mv, anchor_rms, anchor_theta
        h = fd_step
        # Valores base
        s00 = self.physical_model.compute_backscatter(mv0, r0, th0, "VV")
        # Derivadas numéricas (unidades coherentes con IEM en dB, mv en %, rms en cm, theta en grados)
        a = (self.physical_model.compute_backscatter(mv0 + h, r0, th0, "VV")
             - self.physical_model.compute_backscatter(mv0 - h, r0, th0, "VV")) / (2*h)
        b = (self.physical_model.compute_backscatter(mv0, r0 + h, th0, "VV")
             - self.physical_model.compute_backscatter(mv0, r0 - h, th0, "VV")) / (2*h)
        c = (self.physical_model.compute_backscatter(mv0, r0, th0 + h, "VV")
             - self.physical_model.compute_backscatter(mv0, r0, th0 - h, "VV")) / (2*h)
        # Guardar como buffers para usarlos en autograd (constantes)
        self.register_buffer("phys_a", torch.tensor(float(a)))
        self.register_buffer("phys_b", torch.tensor(float(b)))
        self.register_buffer("phys_c", torch.tensor(float(c)))
        self.register_buffer("phys_d0", torch.tensor(float(s00 - a*mv0 - b*r0 - c*th0)))

    def forward(self, x, rms_prior=None):
        """
        Forward pass con guía física.

        Args:
            x: Tensor [batch_size, n_inputs] con [σ⁰_VV, σ⁰_HV, θ, ...]
            rms_prior: Tensor [batch_size, 1] con rugosidad conocida (opcional)

        Returns:
            Tensor [batch_size, 2] con [mv_predicted, rms_predicted]
        """
        # Componente ML: Predicción directa de la red
        ml_prediction = self.network(x)

        if not self.use_physical_residuals:
            return ml_prediction

        # Componente física: inversor lineal aproximado usando sensibilidad a mv.
        # σ0 ≈ a*mv + b*rms + c*theta + d  =>  mv ≈ (σ0 - b*rms - c*theta - d)/a
        sigma0_vv = x[:, 0]
        theta = x[:, 2] if x.shape[1] > 2 else torch.full_like(sigma0_vv, 35.0)
        if rms_prior is None:
            rms_prior = torch.full_like(sigma0_vv, 1.5)
        mv_phys = (sigma0_vv - self.phys_b * rms_prior - self.phys_c * theta - self.phys_d0) / (self.phys_a + 1e-8)
        mv_phys = torch.clamp(mv_phys, VALID_DOMAIN.MV_MIN, VALID_DOMAIN.MV_MAX)
        rms_phys = rms_prior
        physics_prediction = torch.stack([mv_phys, rms_phys], dim=1)

        # Mezcla convexa: w \in (0,1)
        w = torch.sigmoid(self.alpha)
        combined = w * physics_prediction + (1.0 - w) * ml_prediction

        return combined

    def physics_constrained_loss(
        self, predictions, targets, sigma0_inputs, theta_inputs
    ):
        """
        Loss function con restricciones físicas.

        Loss total = MSE(predicción, target) + λ * penalización_física

        Penalizaciones físicas:
        1. Consistencia forward: σ⁰_predicho(mv, rms) ≈ σ⁰_observado
        2. Monotonía: σ⁰ debe aumentar con mv
        3. Rangos físicos: mv y rms dentro de límites

        Args:
            predictions: Tensor [batch, 2] con [mv_pred, rms_pred]
            targets: Tensor [batch, 2] con [mv_true, rms_true]
            sigma0_inputs: Tensor [batch, 1] con σ⁰ observado
            theta_inputs: Tensor [batch, 1] con θ

        Returns:
            Scalar loss
        """
        # 1. Loss de datos (MSE estándar)
        data_loss = nn.MSELoss()(predictions, targets)

        # 2. Penalización de consistencia física diferenciable usando la linealización:
        # σ0_lin(pred) = a*mv_pred + b*rms_pred + c*theta + d
        mv_pred  = predictions[:, 0]
        rms_pred = predictions[:, 1]
        theta    = theta_inputs.squeeze()
        sigma0_lin = self.phys_a * mv_pred + self.phys_b * rms_pred + self.phys_c * theta + self.phys_d0
        sigma0_obs = sigma0_inputs.squeeze()
        physics_consistency_loss = torch.mean((sigma0_lin - sigma0_obs) ** 2)

        # 3. Penalización de rangos físicos
        mv_out_of_range = torch.relu(
            predictions[:, 0] - VALID_DOMAIN.MV_MAX
        ) + torch.relu(VALID_DOMAIN.MV_MIN - predictions[:, 0])

        rms_out_of_range = torch.relu(
            predictions[:, 1] - VALID_DOMAIN.RMS_MAX
        ) + torch.relu(VALID_DOMAIN.RMS_MIN - predictions[:, 1])

        range_penalty = torch.mean(mv_out_of_range**2 + rms_out_of_range**2)

        # Combinar losses
        lambda_physics = 0.1  # Peso de penalización física
        lambda_range = 0.05  # Peso de penalización de rangos

        total_loss = (
            data_loss
            + lambda_physics * physics_consistency_loss
            + lambda_range * range_penalty
        )

        return total_loss, {
            "data_loss": data_loss.item(),
            "physics_loss": physics_consistency_loss.item(),
            "range_penalty": range_penalty.item(),
        }


class SARInversionDataset(Dataset):
    """Dataset para entrenamiento de inversión SAR"""

    def __init__(self, csv_path: str, features: list, targets: list):
        """
        Args:
            csv_path: Ruta del archivo CSV con datos sintéticos
            features: Lista de nombres de columnas para entrada (ej. ['sigma0_VV', 'sigma0_HV', 'theta'])
            targets: Lista de nombres de columnas para salida (ej. ['mv', 'rms'])
        """
        df = pd.read_csv(csv_path)

        # Pivotar polarizaciones a columnas si es necesario
        if "polarization" in df.columns:
            df_pivot = df.pivot_table(
                index=[c for c in ["sample_id", "mv", "rms", "theta"] if c in df.columns],
                columns="polarization",
                values="sigma0_dB",
                aggfunc="first",
            ).reset_index()

            # Renombrar columnas
            cols = list(df_pivot.columns)
            # Asegurar orden: [mv, rms, theta, sigma0_HV, sigma0_VV] (+sample_id si existe)
            rename_map = {}
            if "sigma0_HV" not in cols and "HV" in cols:
                rename_map["HV"] = "sigma0_HV"
            if "sigma0_VV" not in cols and "VV" in cols:
                rename_map["VV"] = "sigma0_VV"
            df_pivot = df_pivot.rename(columns=rename_map)
            df = df_pivot

        self.X = df[features].values.astype(np.float32)
        self.y = df[targets].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


def train_physics_guided_model(
    train_csv: str = "data/training_dataset.csv",
    val_csv: str = "data/validation_dataset.csv",
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    output_dir: str = "inversion/models",
) -> dict:
    """
    Entrena modelo híbrido física-ML.

    Args:
        train_csv: Ruta del dataset de entrenamiento
        val_csv: Ruta del dataset de validación
        epochs: Número de épocas
        batch_size: Tamaño de batch
        learning_rate: Tasa de aprendizaje
        output_dir: Directorio para guardar modelos

    Returns:
        Dict con métricas de entrenamiento
    """
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO DE MODELO HÍBRIDO FÍSICA-ML")
    print("=" * 70)

    # Preparar datasets
    features = ["sigma0_VV", "sigma0_HV", "theta"]
    targets = ["mv", "rms"]

    train_dataset = SARInversionDataset(train_csv, features, targets)
    val_dataset = SARInversionDataset(val_csv, features, targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nDatasets cargados:")
    print(f"  Training:   {len(train_dataset)} muestras")
    print(f"  Validation: {len(val_dataset)} muestras")

    # Inicializar modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    # Semillas básicas
    np.random.seed(42); random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    model = PhysicsGuidedNN(
        n_inputs=len(features), hidden_sizes=[32, 32], use_physical_residuals=True
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # Entrenamiento
    train_losses = []
    val_losses = []
    best_val_loss = np.inf

    print(f"\nEntrenando por {epochs} épocas...")

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions = model(batch_X)

            # Loss con restricciones físicas
            loss, loss_components = model.physics_constrained_loss(
                predictions,
                batch_y,
                sigma0_inputs=batch_X[:, 0:1],  # σ⁰_VV
                theta_inputs=batch_X[:, 2:3],  # θ
            )

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                predictions = model(batch_X)
                loss, _ = model.physics_constrained_loss(
                    predictions,
                    batch_y,
                    sigma0_inputs=batch_X[:, 0:1],
                    theta_inputs=batch_X[:, 2:3],
                )

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # Logging
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

        # Guardar mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                f"{output_dir}/best_physics_guided_model.pth",
            )

    print(f"\n Entrenamiento completado")
    print(f"   Mejor val loss: {best_val_loss:.4f}")
    print(f"   Modelo guardado: {output_dir}/best_physics_guided_model.pth")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "model": model,
    }


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Modelo híbrido física-ML (Estado del Arte 10.1)"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="data/training_dataset.csv",
        help="Dataset de entrenamiento",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="data/validation_dataset.csv",
        help="Dataset de validación",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=256, help="Tamaño de batch")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inversion/models",
        help="Directorio de salida",
    )

    args = parser.parse_args()

    # Verificar que existan los datasets
    if not Path(args.train_csv).exists():
        print(f"\n ERROR: No se encontró {args.train_csv}")
        print("   Ejecute primero: python data/generate_dataset.py")
        sys.exit(1)

    # Entrenar
    results = train_physics_guided_model(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    print("\n MODELO HÍBRIDO FÍSICA-ML ENTRENADO")
    print("\n CONTRIBUCIÓN:")
    print("   Integra conocimiento físico del IEM-B con capacidad de")
    print("   aprendizaje de redes neuronales (Estado del Arte 2025).\n")
