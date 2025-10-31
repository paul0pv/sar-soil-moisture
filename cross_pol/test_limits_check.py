import numpy as np
from models import IEM_Model


def run_limit_test():
    """
    Test 3: Chequeo de Límites - Comportamiento con Suelo Seco (HV).

    OBJETIVO:
    =========
    Verificar que el modelo produce valores físicamente razonables
    en el extremo de humedad muy baja (suelo seco).

    PRINCIPIO FÍSICO:
    =================
    Para polarización cruzada HV:

    1. El agua es el principal mecanismo de despolarización
    2. Sin agua (mv → 0):
       - εr → εr_dry ≈ 3-5 (bajo)
       - Poca diferencia entre R_h y R_v
       - Poca rugosidad efectiva relativa
       - σ⁰_HV debe ser MUY bajo (< -30 dB típicamente)

    3. HV es SIEMPRE más bajo que VV en magnitud
       - HV depende de scattering múltiple puro
       - VV tiene componente de Kirchhoff (más fuerte)

    CRITERIO DE ÉXITO:
    ==================
    Para suelo muy seco (mv = 0.1%):
    σ⁰_HV < -30 dB para todos los casos

    (Más estricto que VV porque HV es inherentemente más débil)
    """
    print("=" * 80)
    print("Test 3: Chequeo de Límites - Suelo Seco (Polarización HV)")
    print("=" * 80)
    print("\nObjetivo: Verificar comportamiento físico en extremo de humedad baja")
    print("Criterio: σ⁰_HV < -30 dB para mv ≈ 0")
    print("-" * 80)

    # Humedad muy baja (suelo seco)
    mv_seca = 0.1  # 0.1% de humedad volumétrica

    # Parámetros de prueba: matriz de rugosidad y ángulo
    scenarios = {
        "Liso, Ángulo Bajo": {
            "rms": 0.5,
            "theta": 25.0,
            "descripcion": "Superficie lisa, incidencia baja",
        },
        "Liso, Ángulo Alto": {
            "rms": 0.5,
            "theta": 45.0,
            "descripcion": "Superficie lisa, incidencia alta",
        },
        "Rugoso, Ángulo Bajo": {
            "rms": 3.0,
            "theta": 25.0,
            "descripcion": "Superficie rugosa, incidencia baja",
        },
        "Rugoso, Ángulo Alto": {
            "rms": 3.0,
            "theta": 45.0,
            "descripcion": "Superficie rugosa, incidencia alta",
        },
    }

    # Instanciar modelo (textura estándar)
    model = IEM_Model(sand_pct=40, clay_pct=30)

    print(f"\nParámetros de prueba:")
    print(f"  Humedad volumétrica: mv = {mv_seca}% (suelo muy seco)")
    print(f"  Textura del suelo: Arena 40%, Arcilla 30%, Limo 30%")
    print(f"  Polarización: HV (cruzada)")

    limite_esperado_dB = -30.0  # Umbral para HV
    tests_passed = 0
    tests_failed = 0

    print("\n" + "=" * 80)
    print("EJECUTANDO TESTS")
    print("=" * 80)

    resultados = []

    for nombre, params in scenarios.items():
        print(f"\n{'-' * 80}")
        print(f"Escenario: {nombre}")
        print(f"{'-' * 80}")
        print(f"Descripción: {params['descripcion']}")
        print(f"  Rugosidad RMS: rms = {params['rms']:.1f} cm")
        print(f"  Ángulo de incidencia: θ = {params['theta']:.1f}°")

        # Calcular σ⁰
        sigma_dB = model.compute_backscatter(
            mv=mv_seca,
            rms_cm=params["rms"],
            theta_deg=params["theta"],
            polarization="HV",
        )

        print(f"\nResultado:")
        print(f"  σ⁰_HV Calculado: {sigma_dB:.2f} dB")
        print(f"  Umbral esperado: < {limite_esperado_dB:.1f} dB")

        # Validación
        if sigma_dB < limite_esperado_dB:
            resultado = "✓ PASS"
            tests_passed += 1
            print(f"\n  {resultado} - Valor dentro del rango esperado")
        else:
            resultado = "✗ FAIL"
            tests_failed += 1
            diferencia = sigma_dB - limite_esperado_dB
            print(f"\n  {resultado} - Valor {diferencia:.2f} dB por encima del umbral")

        resultados.append(
            {"escenario": nombre, "sigma": sigma_dB, "resultado": resultado}
        )

    # --- ANÁLISIS ADICIONAL ---
    print("\n" + "=" * 80)
    print("ANÁLISIS DE TENDENCIAS")
    print("=" * 80)

    sigmas = [r["sigma"] for r in resultados]

    print(f"\nEstadísticas de σ⁰_HV para suelo seco:")
    print(f"  Mínimo: {min(sigmas):.2f} dB")
    print(f"  Máximo: {max(sigmas):.2f} dB")
    print(f"  Promedio: {np.mean(sigmas):.2f} dB")
    print(f"  Desviación estándar: {np.std(sigmas):.2f} dB")

    # Validaciones físicas adicionales
    print(f"\nValidaciones físicas:")

    # 1. Mayor rugosidad → mayor despolarización → σ⁰_HV ligeramente mayor
    # (aunque sigue siendo muy bajo)
    sigma_liso_bajo = resultados[0]["sigma"]
    sigma_rugoso_bajo = resultados[2]["sigma"]

    if sigma_rugoso_bajo > sigma_liso_bajo:
        print(f"  ✓ Rugosidad aumenta despolarización (como esperado)")
        print(
            f"    Liso: {sigma_liso_bajo:.2f} dB → Rugoso: {sigma_rugoso_bajo:.2f} dB"
        )
    else:
        print(f"  ⚠ Rugosidad no aumenta despolarización (inesperado para mv muy bajo)")

    # 2. Mayor ángulo → menor retorno
    sigma_liso_bajo_ang = resultados[0]["sigma"]
    sigma_liso_alto_ang = resultados[1]["sigma"]

    if sigma_liso_alto_ang < sigma_liso_bajo_ang:
        print(f"  ✓ Ángulo mayor reduce retorno (como esperado)")
        print(
            f"    25°: {sigma_liso_bajo_ang:.2f} dB → 45°: {sigma_liso_alto_ang:.2f} dB"
        )

    # --- RESUMEN FINAL ---
    print("\n" + "=" * 80)
    print("RESUMEN DE RESULTADOS")
    print("=" * 80)
    print(f"\n{'Escenario':<30} {'σ⁰_HV (dB)':<15} {'Resultado':<10}")
    print("-" * 80)

    for r in resultados:
        print(f"{r['escenario']:<30} {r['sigma']:>10.2f}     {r['resultado']:<10}")

    print("-" * 80)
    print(f"Tests superados: {tests_passed}/{len(scenarios)}")
    print(f"Tests fallidos: {tests_failed}/{len(scenarios)}")

    print("\n" + "=" * 80)
    if tests_failed == 0:
        print("RESULTADO FINAL: ✓ TODOS LOS TESTS SUPERADOS")
        print("\nEl modelo produce valores físicamente correctos para suelo seco.")
        print("La polarización cruzada HV muestra correctamente señales muy débiles")
        print("cuando hay poca humedad (principal fuente de despolarización).")
    else:
        print("RESULTADO FINAL: ✗ ALGUNOS TESTS FALLARON")
        print(f"\n{tests_failed} de {len(scenarios)} escenarios produjeron valores")
        print("más altos de lo esperado para suelo seco.")
        print("\nPosibles causas:")
        print("  1. Término F_hv demasiado fuerte para bajos ε_r")
        print("  2. Factor de normalización incorrecto en serie")
        print("  3. Espectro de rugosidad demasiado amplio")
    print("=" * 80)


if __name__ == "__main__":
    run_limit_test()
