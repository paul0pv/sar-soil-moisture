import numpy as np
try:
    from core.models import IEM_Model
except Exception:
    from models import IEM_Model


def run_baghdadi_test():
    """
    Validation Test 2 — Baghdadi et al. (2011), IEEE GRSL Vol. 8, No. 1
    "Semiempirical Calibration of the Integral Equation Model (IEM-B) in C-band and Cross Polarization"

    PURPOSE:
    --------
    Validate the *complete integrated model chain*:
        Hallikainen (εr) → Baghdadi (Lopt calibration) → Fung (IEM series)
    under C-band cross-polarized (HV) conditions, using Table 1 reference values.

    Additionally performs optional VV sanity validation to confirm physical consistency.

    CRITERIA:
    ---------
    - Absolute error |Δσ⁰| ≤ 1 dB → PASS (Baghdadi tolerance for HV)
    - RMSE ≤ 1 dB overall for good calibration

    REFERENCES:
    -----------
    Baghdadi et al. (2011), Table 1.
    Fung et al. (1992), Eq. (17–18).
    Hallikainen et al. (1985), dielectric model.
    """

    print("=" * 100)
    print(
        "Validation Test 2 — Baghdadi (2011) Semiempirical Calibration (HV polarization)"
    )
    print("=" * 100)

    # ----------------------------------------------------------------------
    # Test cases from Baghdadi (2011) Table 1, plus one extrapolation case
    # ----------------------------------------------------------------------
    cases = [
        {
            "id": "A11",
            "mv": 16.9,
            "rms_cm": 1.01,
            "theta_deg": 38.6,
            "sand_pct": 19.5,
            "clay_pct": 44.2,
            "sigma_expected_dB": -24.8,
            "desc": "Clay soil, medium moisture, low roughness",
            "tol_db": 1.0,
        },
        {
            "id": "B12",
            "mv": 23.7,
            "rms_cm": 1.63,
            "theta_deg": 38.6,
            "sand_pct": 19.5,
            "clay_pct": 44.2,
            "sigma_expected_dB": -23.5,
            "desc": "Clay soil, high moisture, moderate roughness",
            "tol_db": 1.0,
        },
        {
            "id": "C08",
            "mv": 20.3,
            "rms_cm": 0.82,
            "theta_deg": 38.8,
            "sand_pct": 19.5,
            "clay_pct": 44.2,
            "sigma_expected_dB": -25.2,
            "desc": "Clay soil, medium moisture, very low roughness",
            "tol_db": 1.0,
        },
        {
            "id": "D15",
            "mv": 30.5,
            "rms_cm": 1.25,
            "theta_deg": 40.0,
            "sand_pct": 25.0,
            "clay_pct": 35.0,
            "sigma_expected_dB": -22.0,  # extrapolated expected value
            "desc": "Loam-clay soil, very high moisture (extrapolated)",
            "tol_db": 2.0,  # lax tolerance cause extrapolation
        },
    ]

    # ----------------------------------------------------------------------
    # Execution
    # ----------------------------------------------------------------------
    results = []
    print("\nRunning IEM-B full chain simulations (Gaussian spectrum)...\n")

    for test in cases:
        print("-" * 90)
        print(f"Case {test['id']}: {test['desc']}")
        print("-" * 90)
        print(
            f"mv = {test['mv']:.2f}% | Hrms = {test['rms_cm']:.2f} cm | θ = {test['theta_deg']:.2f}°"
        )
        print(f"Texture → Sand: {test['sand_pct']:.1f}%, Clay: {test['clay_pct']:.1f}%")

        model = IEM_Model(
            sand_pct=test["sand_pct"],
            clay_pct=test["clay_pct"],
            spectrum_mode="gaussian",
            use_strict_fvv=False,
        )

        k = model.k
        s_m = test["rms_cm"] / 100.0
        ks = k * s_m
        Lopt_cm = model.roughness.compute_Lopt(test["rms_cm"], test["theta_deg"], "HV")
        L_m = max(float(np.atleast_1d(Lopt_cm).reshape(-1)[0]) / 100.0, 1e-6)
        print(f"kσ = {ks:.3f}  |  kL = {k*L_m:.3f}")
        if ks >= 3.0:
            print("  [WARN] kσ ≥ 3: fuera del dominio típico de validez de IEM-B para suelo.")

        # --- Compute cross-pol (HV) backscatter ---
        sigma_calc_dB = model.compute_backscatter(
            mv=test["mv"],
            rms_cm=test["rms_cm"],
            theta_deg=test["theta_deg"],
            polarization="HV",
        )
        if isinstance(sigma_calc_dB, np.ndarray):
            sigma_calc_dB = float(np.atleast_1d(sigma_calc_dB).reshape(-1)[0])

        sigma_exp_dB = test["sigma_expected_dB"]
        diff = sigma_calc_dB - sigma_exp_dB
        abs_diff = abs(diff)

        print(f"Calculated σ⁰_HV: {sigma_calc_dB:.2f} dB")
        print(f"Expected σ⁰_HV (Baghdadi 2011): {sigma_exp_dB:.2f} dB")
        print(f"Δ = {diff:+.2f} dB (abs {abs_diff:.2f} dB)")

        tol_db = test.get("tol_db", 1.0)
        status = "✓ PASS" if abs_diff <= tol_db else "✗ FAIL"
        print(f"Result: {status}\n")

        results.append(
            {
                "id": test["id"],
                "calc": sigma_calc_dB,
                "exp": sigma_exp_dB,
                "diff": diff,
                "abs_diff": abs_diff,
                "status": status,
            }
        )

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("RESULT SUMMARY — Baghdadi (2011) HV Validation")
    print("=" * 100)
    print(
        f"{'Case':<8}{'Calc(dB)':>12}{'Exp(dB)':>12}{'Δ(dB)':>12}{'|Δ|':>10}{'Result':>12}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['id']:<8}{r['calc']:>12.2f}{r['exp']:>12.2f}{r['diff']:>12.2f}{r['abs_diff']:>10.2f}{r['status']:>12}"
        )
    print("-" * 80)

    abs_errors = np.array([r["abs_diff"] for r in results], dtype=float)
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(np.square(abs_errors)))

    print(f"\nMean Absolute Error (MAE): {mae:.3f} dB")
    print(f"Root Mean Square Error (RMSE): {rmse:.3f} dB")
    pass_count = sum(r["status"].startswith("✓") for r in results)
    print(f"Passed: {pass_count}/{len(results)} cases")
    print("=" * 100)

    if pass_count == len(results) and rmse < 1.0:
        print(
            "FINAL RESULT: ✓ MODEL VALIDATED — IEM-B matches Baghdadi (2011) within 1 dB tolerance."
        )
    else:
        print(
            "FINAL RESULT: ✗ REVIEW REQUIRED — Deviations exceed 1 dB in one or more cases."
        )
        print("Check:")
        print(" • Lopt calibration parameters (Eq. 3 Baghdadi 2011)")
        print(" • Cross-pol terms F_hv and f_hv in IEM series (Eq. 17–18 Fung 1992)")
    print("=" * 100)

    try:
        t0 = cases[0]
        vv = model.compute_backscatter(
            mv=t0["mv"], rms_cm=t0["rms_cm"], theta_deg=t0["theta_deg"], polarization="VV"
        )
        if isinstance(vv, np.ndarray):
            vv = float(np.atleast_1d(vv).reshape(-1)[0])
        print("\nSanity VV vs HV (case A11):")
        print(f"  σ⁰_VV ≈ {vv:.2f} dB,  σ⁰_HV ≈ {results[0]['calc']:.2f} dB,  Δ(VV−HV) ≈ {vv - results[0]['calc']:.2f} dB")
    except Exception as e:
        print(f"[WARN] VV sanity check skipped: {e}")


if __name__ == "__main__":
    run_baghdadi_test()
