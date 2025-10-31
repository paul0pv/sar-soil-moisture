import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from models import IEM_Model, SurfaceRoughness


def run_fung_test():
    """
    Test 1: Validación Física contra Fung (1992).

    OBJETIVO:
    =========
    Validar la implementación pura de las ecuaciones de scattering (Eq. 17-18)
    para polarización cruzada, bypasando modelos de suelo.

    IMPORTANTE - POLARIZACIÓN CRUZADA:
    ==================================
    Fung (1992) NO presenta una figura específica para HV en el paper principal.
    La Figura 2 es solo para VV. Sin embargo, el paper establece que:

    1. Cross-pol tiene SOLO scattering múltiple (no Kirchhoff)
    2. Cross-pol es típicamente 10-15 dB MENOR que co-pol
    3. Cross-pol debe seguir la misma tendencia angular pero desplazada

    Por lo tanto, este test valida:
    - Comportamiento físico correcto (decrecimiento con ángulo)
    - Magnitud esperada (~10-15 dB menor que VV)
    - Que NO haya término de Kirchhoff contribuyendo

    REFERENCIAS:
    ============
    - Fung et al., IEEE TGRS, Vol. 30, No. 2, March 1992
      * Ecuación (17): Serie de scattering
      * Ecuación (18): Términos I_pq
      * Página 357: "cross-polarized scattering contains only multiple
        scattering terms"
      * Ecuación (97): Reducción a segundo orden para cross-pol
    """
    print("=" * 70)
    print("Test 1: Validación Física - Polarización Cruzada HV")
    print("=" * 70)
    print("\nObjetivo: Verificar física de scattering sin calibraciones")
    print("Método: Parámetros fijos de 'juguete' (bypass Hallikainen/Baghdadi)")
    print("-" * 70)

    # --- 1. PARÁMETROS FIJOS DE REFERENCIA ---
    # Usamos los mismos que Fung Fig. 2, pero para HV
    eps_r = complex(15, 0)  # Dieléctrico fijo (mismo que VV para comparación)
    freq = 5.405e9  # Banda C

    model_shell = IEM_Model(frequency=freq)
    k = model_shell.k

    # Parámetros de rugosidad (mismos que Fig. 2 de Fung)
    ks = 1.0  # k*s = 1.0
    kl = 6.0  # k*L = 6.0

    s_m = ks / k  # rms en metros
    L_m = kl / k  # Longitud de correlación en metros

    print(f"\nParámetros de prueba:")
    print(f"  Constante dieléctrica: εr = {eps_r}")
    print(f"  Frecuencia: {freq / 1e9:.3f} GHz")
    print(f"  k*s = {ks} → s = {s_m * 100:.3f} cm")
    print(f"  k*L = {kl} → L = {L_m * 100:.3f} cm")

    # Rango de ángulos
    theta_deg = np.linspace(20, 70, 51)
    theta_rad = np.deg2rad(theta_deg)

    # Parámetros geométricos
    sint = np.sin(theta_rad)
    cost = np.cos(theta_rad)
    k_z = k * cost
    k_x = k * sint

    # --- 2. CÁLCULO MANUAL (WHITE-BOX TEST) ---
    print("\n" + "-" * 70)
    print("Calculando σ⁰_HV usando métodos internos del modelo...")
    print("-" * 70)

    roughness = SurfaceRoughness()

    # Coeficientes de Fresnel (ambos para cross-pol)
    R_h = model_shell._fresnel_h(eps_r, theta_rad)
    R_v = model_shell._fresnel_v(eps_r, theta_rad)

    # Términos de scattering cruzado
    f_hv = model_shell._f_hv(eps_r, sint, cost, R_h, R_v)
    F_hv = model_shell._F_hv(eps_r, sint, cost)

    print(f"\nVerificación de términos de scattering:")
    print(
        f"  f_hv promedio: {np.mean(np.abs(f_hv)):.6f} (debe ser ≈0 para backscatter)"
    )
    print(f"  F_hv promedio: {np.mean(np.abs(F_hv)):.6f} (debe ser >0)")

    # --- 3. SERIE DE SCATTERING (Eq. 17) ---
    k_z2_s2 = (k_z * s_m) ** 2
    exp_term = np.exp(-2 * k_z2_s2)

    series_sum = np.zeros_like(R_h, dtype=complex)

    for n in range(1, model_shell.N_TERMS + 1):
        n_fact = factorial(n)
        sqrt_n_fact = np.sqrt(n_fact)

        # Espectro de potencia
        Wn = roughness.get_spectrum(2 * k_x, L_m, n)

        # Términos de la serie I_hv (Eq. 18 para cross-pol)
        I_hv_n = ((2 * k_z * s_m) ** n / sqrt_n_fact) * f_hv + (
            (k_z * s_m) ** (2 * n) / (2 * sqrt_n_fact)
        ) * F_hv

        # Acumulación
        series_sum += (Wn / n_fact) * np.abs(I_hv_n) ** 2

    # Coeficiente final (SOLO complementario, no Kirchhoff)
    sigma0_C = (k**2 / 2) * exp_term * series_sum
    sigma0_linear = np.real(sigma0_C)
    sigma0_dB = 10 * np.log10(sigma0_linear)

    # --- 4. VALORES DE REFERENCIA ---
    # Para cross-pol, usamos valores teóricos esperados
    # Basados en la regla: HV ≈ VV - 12 dB (aproximado)
    ref_theta = [20, 30, 40, 50, 60, 70]
    ref_sigma_vv = [-13.0, -15.5, -18.0, -20.5, -23.5, -27.0]  # De Fung Fig 2
    ref_sigma_hv = [x - 12 for x in ref_sigma_vv]  # Offset típico

    print(f"\n" + "-" * 70)
    print("Comparación con valores de referencia:")
    print("-" * 70)
    print(f"{'Ángulo':<10} {'σ⁰_HV calc':<15} {'σ⁰_HV ref':<15} {'Diferencia':<15}")
    print("-" * 70)

    for i, angle in enumerate(ref_theta):
        idx = np.argmin(np.abs(theta_deg - angle))
        calc_val = sigma0_dB[idx]
        ref_val = ref_sigma_hv[i]
        diff = calc_val - ref_val
        print(
            f"{angle:>6.0f}°   {calc_val:>10.2f} dB   {ref_val:>10.2f} dB   {diff:>10.2f} dB"
        )

    # Calcular RMSE
    rmse = 0
    for i, angle in enumerate(ref_theta):
        idx = np.argmin(np.abs(theta_deg - angle))
        rmse += (sigma0_dB[idx] - ref_sigma_hv[i]) ** 2
    rmse = np.sqrt(rmse / len(ref_theta))

    print("-" * 70)
    print(f"RMSE: {rmse:.2f} dB")
    print("-" * 70)

    # --- 5. VALIDACIONES FÍSICAS ---
    print(f"\n" + "=" * 70)
    print("VALIDACIONES FÍSICAS")
    print("=" * 70)

    # Validación 1: Decrecimiento con ángulo
    is_decreasing = np.all(np.diff(sigma0_dB) < 1e-3)  # Tolerancia numérica
    print(f"\n1. Decrecimiento monotónico con ángulo:")
    print(f"   {'✓ PASS' if is_decreasing else '✗ FAIL'}")
    print(f"   σ⁰(20°) = {sigma0_dB[0]:.2f} dB")
    print(f"   σ⁰(70°) = {sigma0_dB[-1]:.2f} dB")
    print(f"   Δ = {sigma0_dB[-1] - sigma0_dB[0]:.2f} dB (debe ser negativo)")

    # Validación 2: Rango de valores esperado
    expected_range = (-35, -20)  # Para HV con estos parámetros
    in_range = (sigma0_dB.min() > expected_range[0]) and (
        sigma0_dB.max() < expected_range[1]
    )
    print(f"\n2. Rango de valores físicamente razonable:")
    print(f"   {'✓ PASS' if in_range else '✗ FAIL'}")
    print(f"   Rango calculado: [{sigma0_dB.min():.2f}, {sigma0_dB.max():.2f}] dB")
    print(f"   Rango esperado: {expected_range} dB")

    # Validación 3: RMSE aceptable
    rmse_pass = rmse < 3.0  # Más tolerante para cross-pol
    print(f"\n3. Precisión respecto a referencia:")
    print(f"   {'✓ PASS' if rmse_pass else '✗ FAIL'}")
    print(f"   RMSE = {rmse:.2f} dB (umbral: < 3.0 dB)")

    # Resultado final
    all_pass = is_decreasing and in_range and rmse_pass
    print("\n" + "=" * 70)
    if all_pass:
        print("RESULTADO FINAL: ✓ TEST SUPERADO")
        print("\nLa física de scattering para HV está correctamente implementada.")
    else:
        print("RESULTADO FINAL: ✗ TEST FALLIDO")
        print("\nRevisar implementación de términos cruzados f_hv y F_hv.")
    print("=" * 70)

    # --- 6. GRÁFICO ---
    print("\nGenerando gráfico comparativo...")

    plt.figure(figsize=(12, 8))

    # Plot principal
    plt.plot(theta_deg, sigma0_dB, "r-", linewidth=2, label="IEM-HV Implementado")
    plt.plot(
        ref_theta, ref_sigma_hv, "bo--", markersize=8, label="Referencia (VV - 12 dB)"
    )

    # Plot VV para comparación
    plt.plot(
        ref_theta,
        ref_sigma_vv,
        "gs--",
        markersize=6,
        alpha=0.5,
        label="Fung (1992) VV (referencia)",
    )

    plt.title(
        "Test 1: Validación Física - Polarización Cruzada HV\n"
        + "Comparación con comportamiento esperado de Fung (1992)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Ángulo de Incidencia θ (grados)", fontsize=12)
    plt.ylabel("σ⁰ (dB)", fontsize=12)
    plt.legend(fontsize=10, loc="lower left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.ylim(-35, -15)

    # Anotaciones
    plt.text(
        0.98,
        0.97,
        f"RMSE = {rmse:.2f} dB\nks = {ks}, kL = {kl}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig("test1_fung_hv_validation.png", dpi=150, bbox_inches="tight")
    print("Gráfico guardado: test1_fung_hv_validation.png")
    plt.show()


if __name__ == "__main__":
    run_fung_test()
