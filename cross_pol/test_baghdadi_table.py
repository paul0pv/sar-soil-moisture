import numpy as np
from models import IEM_Model


def run_baghdadi_test():
    """
    Test 2: Validación contra Baghdadi (2011) - Polarización Cruzada HV.

    OBJETIVO:
    =========
    Validar la cadena completa del modelo IEM_B calibrado:
    Hallikainen → Baghdadi (Lopt) → IEM (Fung)

    FUENTE DE DATOS:
    ================
    Baghdadi et al. (2011), IEEE GRSL, Vol. 8, No. 1, January 2011
    "Semi-empirical Calibration of the Integral Equation Model for SAR Data
    in C-Band and Cross Polarization Using Radar Images and Field Measurements"

    Tabla 1, página 16: Valores de simulación σ⁰_HV

    NOTA IMPORTANTE - POLARIZACIÓN HV:
    ==================================
    El paper de Baghdadi (2011) presenta resultados para HH, VV Y HV.
    Los valores de referencia aquí son específicos para HV.

    CASOS DE PRUEBA:
    ================
    Se usan parcelas con mediciones in-situ de:
    - Humedad volumétrica (mv)
    - Rugosidad RMS (rms)
    - Ángulo de incidencia (theta)
    - Textura del suelo (arena, arcilla)

    CRITERIO DE ÉXITO:
    ==================
    Error absoluto < 1.0 dB (más tolerante que VV debido a menor SNR en HV)
    """
    print("=" * 80)
    print("Test 2: Validación Contra Baghdadi (2011) Tabla 1 - Polarización HV")
    print("=" * 80)
    print("\nObjetivo: Validar cadena completa (Hallikainen + Baghdadi + IEM)")
    print("Fuente: Baghdadi et al., IEEE GRSL, Vol. 8, No. 1, 2011")
    print("-" * 80)

    # --- CASOS DE PRUEBA DE BAGHDADI (2011) TABLA 1 - HV ---
    # NOTA: Los valores de σ⁰ esperados son para polarización HV
    # (diferentes de los valores VV en el paper)

    test_A11 = {
        "nombre": "Parcela A11",
        "mv": 16.9,  # Humedad volumétrica (%)
        "rms_cm": 1.01,  # Rugosidad RMS (cm)
        "theta_deg": 38.6,  # Ángulo de incidencia (grados)
        "sand_pct": 19.5,  # Arena (%)
        "clay_pct": 44.2,  # Arcilla (%)
        "sigma_esperado_dB": -24.8,  # σ⁰_HV simulado (dB) - Tabla 1
        "descripcion": "Suelo arcilloso, humedad media, rugosidad baja",
    }

    test_B12 = {
        "nombre": "Parcela B12",
        "mv": 23.7,
        "rms_cm": 1.63,
        "theta_deg": 38.6,
        "sand_pct": 19.5,
        "clay_pct": 44.2,
        "sigma_esperado_dB": -23.5,  # σ⁰_HV simulado (dB)
        "descripcion": "Suelo arcilloso, humedad alta, rugosidad media",
    }

    test_C08 = {
        "nombre": "Parcela C08",
        "mv": 20.3,
        "rms_cm": 0.82,
        "theta_deg": 38.8,
        "sand_pct": 19.5,
        "clay_pct": 44.2,
        "sigma_esperado_dB": -25.2,  # σ⁰_HV simulado (dB)
        "descripcion": "Suelo arcilloso, humedad media, rugosidad muy baja",
    }

    # Caso adicional para validar rango de humedad
    test_D15 = {
        "nombre": "Parcela D15 (validación)",
        "mv": 30.5,
        "rms_cm": 1.25,
        "theta_deg": 40.0,
        "sand_pct": 25.0,
        "clay_pct": 35.0,
        "sigma_esperado_dB": -22.0,  # Valor teórico esperado
        "descripcion": "Suelo franco-arcilloso, humedad muy alta",
    }

    tests = [test_A11, test_B12, test_C08, test_D15]

    print("\n" + "=" * 80)
    print("EJECUTANDO TESTS")
    print("=" * 80)

    resultados = []
    tests_passed = 0
    tests_failed = 0

    for i, test in enumerate(tests, 1):
        print(f"\n{'-' * 80}")
        print(f"Test {i}/4: {test['nombre']}")
        print(f"{'-' * 80}")
        print(f"Descripción: {test['descripcion']}")
        print(f"\nParámetros de entrada:")
        print(f"  Humedad volumétrica: mv = {test['mv']:.1f}%")
        print(f"  Rugosidad RMS: rms = {test['rms_cm']:.2f} cm")
        print(f"  Ángulo de incidencia: θ = {test['theta_deg']:.1f}°")
        print(f"  Textura del suelo:")
        print(f"    - Arena: {test['sand_pct']:.1f}%")
        print(f"    - Arcilla: {test['clay_pct']:.1f}%")
        print(f"    - Limo: {100 - test['sand_pct'] - test['clay_pct']:.1f}%")

        # 1. Instanciar modelo con textura correcta
        model = IEM_Model(sand_pct=test["sand_pct"], clay_pct=test["clay_pct"])

        # 2. Calcular retrodispersión
        sigma_calculado_dB = model.compute_backscatter(
            mv=test["mv"],
            rms_cm=test["rms_cm"],
            theta_deg=test["theta_deg"],
            polarization="HV",
        )

        # 3. Comparar resultados
        error = sigma_calculado_dB - test["sigma_esperado_dB"]
        error_abs = abs(error)

        print(f"\nResultados:")
        print(f"  σ⁰_HV Calculado: {sigma_calculado_dB:.2f} dB")
        print(f"  σ⁰_HV Esperado (Tabla 1): {test['sigma_esperado_dB']:.2f} dB")
        print(f"  Error: {error:+.2f} dB")
        print(f"  Error absoluto: {error_abs:.2f} dB")

        # Criterio: error < 1.0 dB (más tolerante para HV)
        umbral = 1.0
        if error_abs <= umbral:
            resultado = "✓ PASS"
            tests_passed += 1
            print(f"\n  {resultado} - Error dentro del umbral (< {umbral} dB)")
        else:
            resultado = "✗ FAIL"
            tests_failed += 1
            print(f"\n  {resultado} - Error excede umbral (> {umbral} dB)")

        resultados.append(
            {
                "nombre": test["nombre"],
                "calculado": sigma_calculado_dB,
                "esperado": test["sigma_esperado_dB"],
                "error": error,
                "resultado": resultado,
            }
        )

    # --- RESUMEN FINAL ---
    print("\n" + "=" * 80)
    print("RESUMEN DE RESULTADOS")
    print("=" * 80)
    print(
        f"\n{'Parcela':<20} {'Calculado':<12} {'Esperado':<12} {'Error':<10} {'Resultado':<10}"
    )
    print("-" * 80)

    for r in resultados:
        print(
            f"{r['nombre']:<20} {r['calculado']:>8.2f} dB  "
            f"{r['esperado']:>8.2f} dB  {r['error']:>+7.2f} dB  {r['resultado']:<10}"
        )

    print("-" * 80)
    print(f"Tests superados: {tests_passed}/{len(tests)}")
    print(f"Tests fallidos: {tests_failed}/{len(tests)}")

    # Calcular estadísticas
    errores = [abs(r["error"]) for r in resultados]
    mae = np.mean(errores)
    rmse = np.sqrt(np.mean([r["error"] ** 2 for r in resultados]))

    print(f"\nEstadísticas de error:")
    print(f"  MAE (Error Absoluto Medio): {mae:.2f} dB")
    print(f"  RMSE (Raíz del Error Cuadrático Medio): {rmse:.2f} dB")

    print("\n" + "=" * 80)
    if tests_failed == 0:
        print("RESULTADO FINAL: ✓ TODOS LOS TESTS SUPERADOS")
        print("\nEl modelo IEM_B calibrado para HV reproduce correctamente")
        print("los valores de simulación de Baghdadi (2011).")
    else:
        print("RESULTADO FINAL: ✗ ALGUNOS TESTS FALLARON")
        print(f"\n{tests_failed} de {len(tests)} tests fallaron.")
        print("Revisar:")
        print("  1. Implementación de términos cruzados F_hv")
        print("  2. Calibración de Lopt para polarización HV")
        print("  3. Valores de referencia de Tabla 1")
    print("=" * 80)


if __name__ == "__main__":
    run_baghdadi_test()
