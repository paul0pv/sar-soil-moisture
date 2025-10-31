import numpy as np
import matplotlib.pyplot as plt
from models import IEM_Model


def run_cross_validation_test():
    """
    Test 5 (ADICIONAL): Validación Cruzada HV vs VV.

    OBJETIVO:
    =========
    Verificar que la relación entre polarización cruzada (HV) y
    co-polarizada (VV) es físicamente consistente.

    PRINCIPIO FÍSICO:
    =================

    1. Para TODAS las condiciones:
       σ⁰_HV < σ⁰_VV  (HV es siempre más débil)

    2. Diferencia típica (depolarization ratio):
       DR = σ⁰_VV - σ⁰_HV ≈ 8-15 dB (rango típico)

    3. DR aumenta con:
       - Menor rugosidad (superficies lisas despolarizan menos)
       - Menor humedad (menos contraste dieléctrico)
       - Mayor frecuencia (scattering más superficial)

    4. DR disminuye con:
       - Mayor rugosidad (más despolarización)
       - Mayor humedad (más contraste)

    NOTA:
    =====
    Este test NO puede ejecutarse sin implementación de VV,
    pero se proporciona como referencia para validación futura.
    """
    print("=" * 80)
    print("Test 5: Validación Cruzada - Ratio de Despolarización HV/VV")
    print("=" * 80)
    print("\n⚠ NOTA: Este test requiere implementación de polarización VV")
    print("          Se proporciona como referencia para validación futura")
    print("-" * 80)

    # Parámetros de prueba
    mv_range = np.linspace(5, 40, 35)
    rms_values = [0.5, 1.5, 3.0]
    theta = 30.0

    model_hv = IEM_Model(sand_pct=40, clay_pct=30)

    print(f"\nSi se implementara VV, este test verificaría:")
    print(f"\n1. Que σ⁰_HV < σ⁰_VV para TODOS los puntos")
    print(f"2. Que el ratio de despolarización DR esté en [8, 15] dB")
    print(f"3. Que DR disminuya con rugosidad creciente")
    print(f"4. Que DR sea estable con humedad")

    print(f"\nParámetros de prueba propuestos:")
    print(f"  Humedad: {mv_range.min():.1f}% a {mv_range.max():.1f}%")
    print(f"  Rugosidades: {rms_values} cm")
    print(f"  Ángulo fijo: {theta}°")

    # Calcular solo HV (VV no implementado)
    print(f"\n" + "-" * 80)
    print("Calculando curvas HV (solo para referencia)...")
    print("-" * 80)

    plt.figure(figsize=(14, 5))

    for i, rms in enumerate(rms_values):
        sigma_hv = model_hv.compute_backscatter(
            mv=mv_range, rms_cm=rms, theta_deg=theta, polarization="HV"
        )

        # Plot HV
        plt.subplot(1, 3, 1)
        plt.plot(mv_range, sigma_hv, label=f"HV, rms={rms}cm", linewidth=2)

        # Simular VV esperado (HV + 10-12 dB típicamente)
        sigma_vv_simulado = sigma_hv + (12 - i * 1.5)  # DR decrece con rugosidad

        plt.subplot(1, 3, 2)
        plt.plot(
            mv_range,
            sigma_vv_simulado,
            "--",
            label=f"VV estimado, rms={rms}cm",
            linewidth=2,
            alpha=0.7,
        )

        # Ratio de despolarización
        dr = sigma_vv_simulado - sigma_hv

        plt.subplot(1, 3, 3)
        plt.plot(mv_range, dr, label=f"DR, rms={rms}cm", linewidth=2)

    # Configurar subplots
    plt.subplot(1, 3, 1)
    plt.xlabel("Humedad Volumétrica (%)", fontsize=10)
    plt.ylabel("σ⁰_HV (dB)", fontsize=10)
    plt.title("Polarización HV\n(Implementado)", fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.xlabel("Humedad Volumétrica (%)", fontsize=10)
    plt.ylabel("σ⁰_VV (dB)", fontsize=10)
    plt.title("Polarización VV\n(Estimado para referencia)", fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.xlabel("Humedad Volumétrica (%)", fontsize=10)
    plt.ylabel("DR = σ⁰_VV - σ⁰_HV (dB)", fontsize=10)
    plt.title("Ratio de Despolarización\n(Debe estar en [8, 15] dB)", fontweight="bold")
    plt.axhspan(8, 15, alpha=0.2, color="green", label="Rango esperado")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("test5_cross_validation_reference.png", dpi=150, bbox_inches="tight")
    print("\nGráfico de referencia guardado: test5_cross_validation_reference.png")
    plt.show()

    print("\n" + "=" * 80)
    print("Test 5: NO EJECUTABLE (requiere implementación VV)")
    print("=" * 80)
    print("\nPara implementar este test en el futuro:")
    print("  1. Crear clase IEM_Model_VV con polarización co-polarizada")
    print("  2. Calcular ambas polarizaciones para mismos parámetros")
    print("  3. Verificar que HV < VV siempre")
    print("  4. Verificar que DR esté en rango [8, 15] dB")
    print("  5. Verificar tendencias de DR con rugosidad y humedad")
    print("=" * 80)


if __name__ == "__main__":
    run_cross_validation_test()
