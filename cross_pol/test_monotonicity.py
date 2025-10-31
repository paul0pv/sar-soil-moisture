import numpy as np
import matplotlib.pyplot as plt
from models import IEM_Model


def run_monotonicity_test():
    """
    Test 4: Prueba de Monotonicidad - Sensibilidad a Humedad del Suelo (HV).

    OBJETIVO:
    =========
    Verificar que σ⁰_HV aumenta monótonamente con la humedad del suelo.
    Esta es LA prueba más fundamental para teledetección de humedad.

    PRINCIPIO FÍSICO - POLARIZACIÓN CRUZADA:
    ========================================

    1. A medida que mv aumenta:
       - ε_r aumenta (Hallikainen)
       - La diferencia entre R_h y R_v aumenta
       - La despolarización aumenta
       - El scattering múltiple aumenta
       - Por lo tanto: σ⁰_HV DEBE aumentar

    2. Para HV, la sensibilidad a humedad es DIFERENTE que para VV:
       - HV es más sensible a rugosidad volumétrica (despolarización)
       - HV puede tener menor rango dinámico que VV
       - Pero SIEMPRE debe ser monotónico creciente

    3. Comportamiento esperado de las curvas:
       - Inicio (mv bajo): σ⁰_HV muy bajo (< -30 dB)
       - Crecimiento: Aproximadamente logarítmico
       - Final (mv alto): σ⁰_HV más alto pero aún < σ⁰_VV

    CRITERIO DE ÉXITO:
    ==================
    Para TODAS las configuraciones (rms, θ):
    - σ⁰_HV(mv₁) < σ⁰_HV(mv₂) cuando mv₁ < mv₂
    - Sin decrecimiento > 0.001 dB entre puntos consecutivos
    - Curvas suaves sin oscilaciones

    IMPORTANCIA:
    ============
    Si este test falla, el modelo NO puede usarse para teledetección
    de humedad, ya que la relación física fundamental está rota.
    """
    print("=" * 80)
    print("Test 4: Prueba de Monotonicidad - Sensibilidad a Humedad (HV)")
    print("=" * 80)
    print("\nObjetivo: Verificar que σ⁰_HV aumenta monótonamente con mv")
    print("Criterio: Curvas estrictamente crecientes (sin decrementos)")
    print("-" * 80)

    # --- RANGO DE HUMEDAD ---
    # Desde 1% (casi seco) hasta 45% (saturado)
    mv_values = np.linspace(1, 45, 88)  # 88 puntos para alta resolución

    # --- ESCENARIOS DE PRUEBA ---
    # Diferentes combinaciones de rugosidad y ángulo
    scenarios = {
        "Liso (rms=0.5cm), θ=30°": {
            "rms": 0.5,
            "theta": 30.0,
            "color": "blue",
            "descripcion": "Superficie muy lisa, ángulo bajo",
        },
        "Medio (rms=1.5cm), θ=30°": {
            "rms": 1.5,
            "theta": 30.0,
            "color": "orange",
            "descripcion": "Rugosidad moderada, ángulo bajo",
        },
        "Rugoso (rms=3.0cm), θ=30°": {
            "rms": 3.0,
            "theta": 30.0,
            "color": "green",
            "descripcion": "Superficie rugosa, ángulo bajo",
        },
        "Rugoso (rms=3.0cm), θ=45°": {
            "rms": 3.0,
            "theta": 45.0,
            "color": "red",
            "descripcion": "Superficie rugosa, ángulo alto",
        },
    }

    # Instanciar modelo (textura estándar)
    model = IEM_Model(sand_pct=40, clay_pct=30)

    print(f"\nParámetros de prueba:")
    print(f"  Rango de humedad: {mv_values.min():.1f}% a {mv_values.max():.1f}%")
    print(f"  Número de puntos: {len(mv_values)}")
    print(f"  Textura del suelo: Arena 40%, Arcilla 30%, Limo 30%")
    print(f"  Polarización: HV (cruzada)")

    print("\n" + "=" * 80)
    print("CALCULANDO CURVAS DE SENSIBILIDAD")
    print("=" * 80)

    resultados = {}
    tests_passed = 0
    tests_failed = 0

    for nombre, params in scenarios.items():
        print(f"\n{'-' * 80}")
        print(f"Escenario: {nombre}")
        print(f"{'-' * 80}")
        print(f"  {params['descripcion']}")
        print(f"  rms = {params['rms']} cm, θ = {params['theta']}°")
        print(f"  Calculando {len(mv_values)} puntos...", end=" ")

        # Calcular σ⁰ para todo el rango de humedad
        sigma_dB = model.compute_backscatter(
            mv=mv_values,
            rms_cm=params["rms"],
            theta_deg=params["theta"],
            polarization="HV",
        )

        print("✓")

        # --- VERIFICAR MONOTONICIDAD ---
        # np.diff() calcula diferencias entre elementos consecutivos
        # Si hay alguna diferencia negativa significativa → no monotónico
        diferencias = np.diff(sigma_dB)

        # Permitir pequeño error numérico (< 0.001 dB)
        decrementos = diferencias[diferencias < -1e-3]

        if len(decrementos) > 0:
            resultado = "✗ FAIL"
            tests_failed += 1
            print(f"\n  {resultado} - La curva NO es monotónica")
            print(f"    Decrementos encontrados: {len(decrementos)}")
            print(f"    Mayor decremento: {decrementos.min():.4f} dB")

            # Encontrar ubicación del primer decremento
            idx_decremento = np.where(diferencias < -1e-3)[0][0]
            print(f"    Primer decremento en:")
            print(
                f"      mv = {mv_values[idx_decremento]:.1f}% → {mv_values[idx_decremento + 1]:.1f}%"
            )
            print(
                f"      σ⁰ = {sigma_dB[idx_decremento]:.2f} dB → {sigma_dB[idx_decremento + 1]:.2f} dB"
            )
        else:
            resultado = "✓ PASS"
            tests_passed += 1
            print(f"\n  {resultado} - Curva monotónicamente creciente")

        # Estadísticas de la curva
        print(f"\n  Estadísticas:")
        print(f"    σ⁰_HV(mv=1%): {sigma_dB[0]:.2f} dB")
        print(f"    σ⁰_HV(mv=45%): {sigma_dB[-1]:.2f} dB")
        print(f"    Rango dinámico: {sigma_dB[-1] - sigma_dB[0]:.2f} dB")
        print(f"    Incremento promedio: {np.mean(diferencias):.4f} dB/punto")

        # Guardar resultados
        resultados[nombre] = {
            "mv": mv_values,
            "sigma": sigma_dB,
            "color": params["color"],
            "resultado": resultado,
            "diferencias": diferencias,
        }

    # --- ANÁLISIS COMPARATIVO ---
    print("\n" + "=" * 80)
    print("ANÁLISIS COMPARATIVO")
    print("=" * 80)

    print(f"\n{'Escenario':<35} {'Rango Dinámico':<20} {'Resultado':<10}")
    print("-" * 80)

    for nombre, data in resultados.items():
        rango = data["sigma"][-1] - data["sigma"][0]
        print(f"{nombre:<35} {rango:>12.2f} dB      {data['resultado']:<10}")

    print("-" * 80)

    # Validaciones físicas
    print(f"\nValidaciones físicas:")

    # 1. Mayor rugosidad → mayor rango dinámico (más despolarización)
    rangos = {
        nombre: data["sigma"][-1] - data["sigma"][0]
        for nombre, data in resultados.items()
    }

    print(f"\n  1. Efecto de la rugosidad en rango dinámico:")
    liso = list(resultados.keys())[0]
    rugoso = list(resultados.keys())[2]
    if rangos[rugoso] > rangos[liso]:
        print(f"     ✓ Rugoso ({rangos[rugoso]:.2f} dB) > Liso ({rangos[liso]:.2f} dB)")
    else:
        print(f"     ⚠ Inesperado: Liso > Rugoso en rango dinámico")

    # 2. Verificar valores finales razonables
    print(f"\n  2. Valores finales (mv=45%):")
    valores_finales = [data["sigma"][-1] for data in resultados.values()]
    if all(-25 < v < -15 for v in valores_finales):
        print(f"     ✓ Todos en rango esperado [-25, -15] dB para HV")
    else:
        print(f"     ⚠ Algunos valores fuera del rango esperado")

    # --- RESUMEN FINAL ---
    print("\n" + "=" * 80)
    print("RESUMEN DE RESULTADOS")
    print("=" * 80)

    print(f"\nTests superados: {tests_passed}/{len(scenarios)}")
    print(f"Tests fallidos: {tests_failed}/{len(scenarios)}")

    print("\n" + "=" * 80)
    if tests_failed == 0:
        print("RESULTADO FINAL: ✓ PRUEBA DE MONOTONICIDAD SUPERADA")
        print("\nTodas las curvas son monotónicamente crecientes.")
        print("El modelo es físicamente válido para teledetección de humedad.")
        print("\nEl modelo IEM_B para HV captura correctamente:")
        print("  • La sensibilidad de despolarización a la humedad del suelo")
        print("  • El comportamiento físico esperado en todo el rango")
        print("  • La interacción entre rugosidad, ángulo y humedad")
    else:
        print("RESULTADO FINAL: ✗ PRUEBA DE MONOTONICIDAD FALLIDA")
        print(f"\n{tests_failed} de {len(scenarios)} curvas NO son monotónicas.")
        print("\nEl modelo NO es válido para teledetección de humedad.")
        print("\nPosibles causas:")
        print("  1. Error en implementación de F_hv (término cruzado)")
        print("  2. Normalización incorrecta en la serie de scattering")
        print("  3. Espectro de rugosidad inestable para ciertos parámetros")
        print("  4. Interacción incorrecta entre términos f_hv y F_hv")
    print("=" * 80)

    # --- GRÁFICO ---
    print("\nGenerando gráfico de sensibilidad...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Panel izquierdo: Curvas de σ⁰ vs mv
    for nombre, data in resultados.items():
        linestyle = "-" if "PASS" in data["resultado"] else "--"
        linewidth = 2 if "PASS" in data["resultado"] else 1.5
        alpha = 1.0 if "PASS" in data["resultado"] else 0.6

        ax1.plot(
            data["mv"],
            data["sigma"],
            color=data["color"],
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=nombre,
        )

    ax1.set_xlabel("Humedad Volumétrica del Suelo ($M_v$, %)", fontsize=12)
    ax1.set_ylabel("Retrodispersión Simulada ($\\sigma^0_{HV}$, dB)", fontsize=12)
    ax1.set_title(
        "Test 4: Sensibilidad del Modelo IEM_B (HV) a la Humedad del Suelo",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend(fontsize=10, loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_xlim(0, 46)

    # Panel derecho: Incrementos (derivada numérica)
    for nombre, data in resultados.items():
        # Calcular incrementos suavizados
        mv_mid = (data["mv"][:-1] + data["mv"][1:]) / 2

        ax2.plot(
            mv_mid,
            data["diferencias"],
            color=data["color"],
            linewidth=1.5,
            alpha=0.7,
            label=nombre,
        )

    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Humedad Volumétrica del Suelo ($M_v$, %)", fontsize=12)
    ax2.set_ylabel("Incremento de $\\sigma^0$ (dB/punto)", fontsize=12)
    ax2.set_title(
        "Incrementos entre Puntos Consecutivos\n(Debe ser siempre ≥ 0)",
        fontsize=13,
        fontweight="bold",
    )
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_xlim(0, 46)

    # Sombrear región de decrementos (si existen)
    ax2.fill_between(
        [0, 46],
        -0.01,
        0,
        color="red",
        alpha=0.1,
        label="Región de decremento (NO física)",
    )

    plt.tight_layout()
    plt.savefig("test4_monotonicity_hv.png", dpi=150, bbox_inches="tight")
    print("Gráfico guardado: test4_monotonicity_hv.png")
    plt.show()


if __name__ == "__main__":
    run_monotonicity_test()
