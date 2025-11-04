import numpy as np
import matplotlib.pyplot as plt
from core.models import IEM_Model


def run_monotonicity_test():
    print("=" * 80)
    print("Validation Test 4 — Soil Moisture Monotonicity (HV polarization)")
    print("=" * 80)

    mv_values = np.linspace(1, 45, 88)
    scenarios = {
        "Smooth (rms=0.5 cm), θ=30°": {"rms": 0.5, "theta": 30.0, "color": "blue"},
        "Medium (rms=1.5 cm), θ=30°": {"rms": 1.5, "theta": 30.0, "color": "orange"},
        "Rough (rms=3.0 cm), θ=30°": {"rms": 3.0, "theta": 30.0, "color": "green"},
        "Rough (rms=3.0 cm), θ=45°": {"rms": 3.0, "theta": 45.0, "color": "red"},
    }

    model = IEM_Model(sand_pct=40, clay_pct=30)
    print(f"Moisture range: {mv_values.min():.1f}% → {mv_values.max():.1f}%")
    print(f"Soil texture: sand 40%, clay 30%, silt 30% | polarization: HV")
    print("-" * 80)

    resultados = {}
    passed, failed = 0, 0

    for name, params in scenarios.items():
        print(f"\nScenario: {name}")
        sigma_dB = model.compute_backscatter(
            mv=mv_values,
            rms_cm=params["rms"],
            theta_deg=params["theta"],
            polarization="HV",
        )

        diffs = np.diff(sigma_dB)
        dec = diffs[diffs < -1e-2]
        if len(dec) > 0:
            result = "✗ FAIL"
            failed += 1
            print(f"  → Non-monotonic: {len(dec)} negative increments")
            print(f"    Largest negative Δ: {dec.min():.3f} dB")
        else:
            result = "✓ PASS"
            passed += 1
            print("  → Monotonic increase confirmed")

        print(
            f"  σ⁰_HV(1%) = {sigma_dB[0]:.2f} dB | σ⁰_HV(45%) = {sigma_dB[-1]:.2f} dB"
        )
        print(f"  Dynamic range = {sigma_dB[-1] - sigma_dB[0]:.2f} dB")

        resultados[name] = {
            "mv": mv_values,
            "sigma": sigma_dB,
            "color": params["color"],
            "result": result,
            "diffs": diffs,
        }

    print("\n" + "=" * 80)
    print("Summary by scenario")
    print("=" * 80)
    print(f"{'Scenario':<40} {'Dynamic Range (dB)':<20} {'Result'}")
    print("-" * 80)
    for name, data in resultados.items():
        rng = data["sigma"][-1] - data["sigma"][0]
        print(f"{name:<40} {rng:>10.2f} dB        {data['result']}")
    print("-" * 80)

    print("\nPhysical sanity checks:")
    rangos = {n: d["sigma"][-1] - d["sigma"][0] for n, d in resultados.items()}
    smooth, rough = list(rangos.values())[0], list(rangos.values())[2]
    if rough > smooth:
        print(
            f"  ✓ Rough surface has greater dynamic range ({rough:.2f} > {smooth:.2f} dB)"
        )
    else:
        print("  ⚠ Unexpected: smooth > rough dynamic range")

    vals_final = [d["sigma"][-1] for d in resultados.values()]
    if all(-35 < v < -10 for v in vals_final):
        print("  ✓ All final σ⁰ values within physical range (−35 < σ⁰ < −10 dB)")
    else:
        print("  ⚠ Some final σ⁰ values outside expected HV range")

    print("\n" + "=" * 80)
    print(f"Passed: {passed}/{len(scenarios)} | Failed: {failed}/{len(scenarios)}")
    print("=" * 80)
    if failed == 0:
        print("FINAL RESULT: ✓ All HV curves increase monotonically with moisture")
    else:
        print(
            "FINAL RESULT: ✗ Non-monotonic behavior detected — review cross-pol terms"
        )
    print("=" * 80)

    # --- plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for name, data in resultados.items():
        style = "-" if "PASS" in data["result"] else "--"
        ax1.plot(
            data["mv"],
            data["sigma"],
            style,
            color=data["color"],
            lw=2,
            label=f"{name} ({data['result']})",
        )
    ax1.set_xlabel("Volumetric moisture Mv (%)")
    ax1.set_ylabel("Backscatter σ⁰_HV (dB)")
    ax1.set_title("σ⁰_HV vs Mv — Monotonicity test")
    ax1.grid(True, ls=":")
    ax1.legend()

    for name, data in resultados.items():
        mv_mid = (data["mv"][:-1] + data["mv"][1:]) / 2
        ax2.plot(mv_mid, data["diffs"], color=data["color"], lw=1.3, label=name)
    ax2.axhline(0, color="k", ls="--", lw=1)
    ax2.fill_between([0, 45], -0.05, 0, color="red", alpha=0.1)
    ax2.set_xlabel("Volumetric moisture Mv (%)")
    ax2.set_ylabel("Δσ⁰ (dB/step)")
    ax2.set_title("σ⁰ increments — should remain ≥ 0")
    ax2.grid(True, ls=":")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("validation_test4_monotonicity.png", dpi=150, bbox_inches="tight")
    print("Saved validation_test4_monotonicity.png")
    plt.close()


if __name__ == "__main__":
    run_monotonicity_test()
