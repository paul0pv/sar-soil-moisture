import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from models import IEM_Model, SurfaceRoughness


# -------------------------
# Helpers
# -------------------------
def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sqrt(np.mean((a - b) ** 2))


def nearest_indices(xvals, targets):
    return [int(np.argmin(np.abs(xvals - t))) for t in targets]


# -------------------------
# SECTION 1 — Fung (1992) Physical Validation
# -------------------------
def run_fung_test(iem):
    freq = iem.frequency
    k = iem.k
    ks = 1.0
    kl = 6.0
    s_m = ks / k
    L_m = kl / k
    theta_deg = np.linspace(20.0, 70.0, 51)
    # compute VV and HV white-box using toy dielectric
    eps_r = complex(15.0, 0.0)
    theta_rad = np.deg2rad(theta_deg)
    sint, cost = np.sin(theta_rad), np.cos(theta_rad)
    k_z = k * cost
    k_x = k * sint
    rough = SurfaceRoughness()

    # HV
    R_h = iem._fresnel_h(eps_r, theta_rad)
    R_v = iem._fresnel_v(eps_r, theta_rad)
    f_hv = iem._f_hv(eps_r, sint, cost, R_h, R_v)
    F_hv = iem._F_hv(eps_r, sint, cost)

    exp_term = np.exp(-2.0 * (k_z * s_m) ** 2)
    series_hv = np.zeros_like(theta_deg, dtype=float)
    for n in range(1, iem.N_TERMS + 1):
        n_fact = float(factorial(n))
        Wn = rough.get_spectrum(2.0 * k_x, L_m, n)
        I_n = (2.0 * k_z) ** n * f_hv + 0.5 * (k_z ** (2 * n)) * F_hv
        term_n = (1.0 / n_fact) * (np.abs(I_n) ** 2) * Wn
        series_hv += np.real(term_n)
    sigma0_lin_hv = (k**2 / (2.0 * np.pi)) * (cost**2) * exp_term * series_hv
    sigma0_lin_hv = np.clip(np.real(sigma0_lin_hv), 1e-15, None)
    sigma0_dB_hv = 10.0 * np.log10(sigma0_lin_hv)

    # VV
    f_vv = iem._f_vv(eps_r, sint, cost, R_v)
    F_vv = iem._F_vv(eps_r, sint, cost)
    series_vv = np.zeros_like(theta_deg, dtype=float)
    for n in range(1, iem.N_TERMS + 1):
        n_fact = float(factorial(n))
        Wn = rough.get_spectrum(2.0 * k_x, L_m, n)
        I_n_vv = (2.0 * k_z) ** n * f_vv + 0.5 * (k_z ** (2 * n)) * F_vv
        term_n = (1.0 / n_fact) * (np.abs(I_n_vv) ** 2) * Wn
        series_vv += np.real(term_n)
    sigma0_lin_vv = (k**2 / (2.0 * np.pi)) * (cost**2) * exp_term * series_vv
    sigma0_lin_vv = np.clip(np.real(sigma0_lin_vv), 1e-15, None)
    sigma0_dB_vv = 10.0 * np.log10(sigma0_lin_vv)

    # Reference (approximate) from Fung Fig.2 for VV and HV ~ VV - 12 dB
    ref_theta = np.array([20, 30, 40, 50, 60, 70])
    ref_sigma_vv = np.array([-13.0, -15.5, -18.0, -20.5, -23.5, -27.0])
    ref_sigma_hv = ref_sigma_vv - 12.0

    idxs = nearest_indices(theta_deg, ref_theta)
    calc_vv_at_refs = sigma0_dB_vv[idxs]
    calc_hv_at_refs = sigma0_dB_hv[idxs]

    rmse_vv = rmse(calc_vv_at_refs, ref_sigma_vv)
    rmse_hv = rmse(calc_hv_at_refs, ref_sigma_hv)

    # physical checks
    monotonic_vv = np.all(np.diff(sigma0_dB_vv) < 1e-3 * -1)
    monotonic_hv = np.all(np.diff(sigma0_dB_hv) < 1e-3 * -1)
    range_ok_vv = (np.min(sigma0_dB_vv) > -50.0) and (np.max(sigma0_dB_vv) < 0.0)
    range_ok_hv = (np.min(sigma0_dB_hv) > -60.0) and (np.max(sigma0_dB_hv) < 0.0)

    # Plot combined VV/HV
    plt.figure(figsize=(9, 6))
    plt.plot(theta_deg, sigma0_dB_vv, label="VV (computed)", lw=2)
    plt.plot(theta_deg, sigma0_dB_hv, label="HV (computed)", lw=2)
    plt.plot(ref_theta, ref_sigma_vv, "rs--", label="VV (ref Fung approx)")
    plt.plot(ref_theta, ref_sigma_hv, "bo--", label="HV (ref ~VV-12dB)")
    plt.xlabel("Incidence angle (deg)")
    plt.ylabel("σ⁰ (dB)")
    plt.title("Section 1 — Fung (1992) Physical Validation (VV & HV)")
    plt.legend()
    plt.grid(True)
    plt.ylim(-50, 5)
    fname = "section1_fung_vv_hv.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()

    verdict = {
        "rmse_vv": float(rmse_vv),
        "rmse_hv": float(rmse_hv),
        "monotonic_vv": bool(monotonic_vv),
        "monotonic_hv": bool(monotonic_hv),
        "range_ok_vv": bool(range_ok_vv),
        "range_ok_hv": bool(range_ok_hv),
    }
    return verdict


# -------------------------
# SECTION 2 — Baghdadi (2011) Semiempirical Validation
# -------------------------
def run_baghdadi_test(iem):
    cases = [
        {
            "id": "A11",
            "mv": 16.9,
            "rms_cm": 1.01,
            "theta_deg": 38.6,
            "sand_pct": 19.5,
            "clay_pct": 44.2,
            "sigma_expected_hv": -24.8,
            "sigma_expected_vv": None,
        },
        {
            "id": "B12",
            "mv": 23.7,
            "rms_cm": 1.63,
            "theta_deg": 38.6,
            "sand_pct": 19.5,
            "clay_pct": 44.2,
            "sigma_expected_hv": -23.5,
            "sigma_expected_vv": None,
        },
        {
            "id": "C08",
            "mv": 20.3,
            "rms_cm": 0.82,
            "theta_deg": 38.8,
            "sand_pct": 19.5,
            "clay_pct": 44.2,
            "sigma_expected_hv": -25.2,
            "sigma_expected_vv": None,
        },
        {
            "id": "D15",
            "mv": 30.5,
            "rms_cm": 1.25,
            "theta_deg": 40.0,
            "sand_pct": 25.0,
            "clay_pct": 35.0,
            "sigma_expected_hv": -22.0,
            "sigma_expected_vv": None,
        },
    ]

    results = []
    for test in cases:
        model = IEM_Model(
            sand_pct=test["sand_pct"],
            clay_pct=test["clay_pct"],
            frequency=iem.frequency,
        )
        sigma_hv = model.compute_backscatter(
            test["mv"], test["rms_cm"], test["theta_deg"], polarization="HV"
        )
        sigma_vv = model.compute_backscatter(
            test["mv"], test["rms_cm"], test["theta_deg"], polarization="VV"
        )
        expected_hv = test["sigma_expected_hv"]
        diff_hv = sigma_hv - expected_hv
        results.append(
            {
                "id": test["id"],
                "calc_hv": float(sigma_hv),
                "exp_hv": expected_hv,
                "diff_hv": float(diff_hv),
                "calc_vv": float(sigma_vv),
            }
        )

        # Combined plot per case
        plt.figure(figsize=(6, 4))
        mv_range = np.linspace(max(1, test["mv"] - 10), min(45, test["mv"] + 10), 50)
        vv_curve = model.compute_backscatter(
            mv_range, test["rms_cm"], test["theta_deg"], polarization="VV"
        )
        hv_curve = model.compute_backscatter(
            mv_range, test["rms_cm"], test["theta_deg"], polarization="HV"
        )
        plt.plot(mv_range, vv_curve, label="VV")
        plt.plot(mv_range, hv_curve, label="HV")
        plt.scatter(
            [test["mv"]], [sigma_vv], marker="s", color="tab:orange", label="VV @ case"
        )
        plt.scatter(
            [test["mv"]], [sigma_hv], marker="o", color="tab:blue", label="HV @ case"
        )
        plt.xlabel("Mv (%)")
        plt.ylabel("σ⁰ (dB)")
        plt.title(f"Baghdadi case {test['id']} — VV & HV")
        plt.legend()
        plt.grid(True)
        fname = f"section2_baghdadi_case_{test['id']}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.show()

    # summary metrics
    diffs = np.array([r["diff_hv"] for r in results])
    mae = np.mean(np.abs(diffs))
    rmse_val = np.sqrt(np.mean(diffs**2))
    verdict = {"cases": results, "mae_hv": float(mae), "rmse_hv": float(rmse_val)}
    return verdict


# -------------------------
# SECTION 3 — Physical Limits
# -------------------------
def run_limits_test(iem):
    freq = iem.frequency
    k = iem.k
    kl = 6.0
    theta_deg = np.linspace(20.0, 70.0, 51)
    theta_rad = np.deg2rad(theta_deg)
    eps_r = complex(15.0, 0.0)

    rough = SurfaceRoughness()
    ks_values = [0.3, 1.0, 3.0]
    figs = []
    all_data = []

    plt.figure(figsize=(9, 6))
    for ks in ks_values:
        s_m = ks / k
        L_m = kl / k
        # compute VV and HV via model (use compute_backscatter for realistic chain)
        vv = iem.compute_backscatter(20.0, s_m * 100.0, theta_deg, polarization="VV")
        hv = iem.compute_backscatter(20.0, s_m * 100.0, theta_deg, polarization="HV")
        plt.plot(theta_deg, vv, label=f"VV ks={ks}")
        plt.plot(theta_deg, hv, label=f"HV ks={ks}", linestyle="--")
        all_data.append({"ks": ks, "vv": vv, "hv": hv})

    plt.xlabel("Incidence angle (deg)")
    plt.ylabel("σ⁰ (dB)")
    plt.title("Section 3 — Limits (Roughness sweep) — VV & HV")
    plt.grid(True)
    plt.legend()
    plt.ylim(-50, 5)
    fname = "section3_limits_roughness.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()

    # Dielectric / moisture limits (several mv values)
    mv_values = [1.0, 10.0, 25.0, 40.0]
    plt.figure(figsize=(8, 6))
    for mv in mv_values:
        vv = iem.compute_backscatter(mv, 1.0, 35.0, polarization="VV")
        hv = iem.compute_backscatter(mv, 1.0, 35.0, polarization="HV")
        plt.scatter(mv, vv, c="tab:blue")
        plt.scatter(mv, hv, c="tab:orange")
    # lines for guidance
    vv_line = [
        iem.compute_backscatter(mv, 1.0, 35.0, polarization="VV") for mv in mv_values
    ]
    hv_line = [
        iem.compute_backscatter(mv, 1.0, 35.0, polarization="HV") for mv in mv_values
    ]
    plt.plot(mv_values, vv_line, "b-", label="VV")
    plt.plot(mv_values, hv_line, "r--", label="HV")
    plt.xlabel("Mv (%)")
    plt.ylabel("σ⁰ (dB)")
    plt.title("Section 3 — Limits (Moisture sweep) — VV & HV")
    plt.grid(True)
    plt.legend()
    fname2 = "section3_limits_moisture.png"
    plt.savefig(fname2, dpi=150, bbox_inches="tight")
    plt.show()

    return {"roughness_data": all_data, "mv_points": mv_values}


# -------------------------
# SECTION 4 — Monotonicity vs Moisture
# -------------------------
def run_monotonicity_test(iem):
    mv_values = np.linspace(1, 45, 88)
    scenarios = {
        "Smooth (rms=0.5 cm), θ=30°": {"rms": 0.5, "theta": 30.0, "color": "blue"},
        "Medium (rms=1.5 cm), θ=30°": {"rms": 1.5, "theta": 30.0, "color": "orange"},
        "Rough (rms=3.0 cm), θ=30°": {"rms": 3.0, "theta": 30.0, "color": "green"},
        "Rough (rms=3.0 cm), θ=45°": {"rms": 3.0, "theta": 45.0, "color": "red"},
    }

    summary = {}
    plt.figure(figsize=(10, 6))
    for name, params in scenarios.items():
        sigma_hv = iem.compute_backscatter(
            mv_values, params["rms"], params["theta"], polarization="HV"
        )
        sigma_vv = iem.compute_backscatter(
            mv_values, params["rms"], params["theta"], polarization="VV"
        )
        diffs_hv = np.diff(sigma_hv)
        diffs_vv = np.diff(sigma_vv)
        nonmono_hv = np.sum(diffs_hv < -1e-2)
        nonmono_vv = np.sum(diffs_vv < -1e-2)

        result_hv = "PASS" if nonmono_hv == 0 else "FAIL"
        result_vv = "PASS" if nonmono_vv == 0 else "FAIL"
        summary[name] = {
            "hv_nonmono": int(nonmono_hv),
            "vv_nonmono": int(nonmono_vv),
            "hv_start": float(sigma_hv[0]),
            "hv_end": float(sigma_hv[-1]),
            "vv_start": float(sigma_vv[0]),
            "vv_end": float(sigma_vv[-1]),
            "hv_range": float(sigma_hv[-1] - sigma_hv[0]),
            "vv_range": float(sigma_vv[-1] - sigma_vv[0]),
            "hv_result": result_hv,
            "vv_result": result_vv,
        }
        plt.plot(
            mv_values,
            sigma_hv,
            color=params["color"],
            linestyle="--",
            label=f"HV {name} ({result_hv})",
        )
        plt.plot(
            mv_values,
            sigma_vv,
            color=params["color"],
            linestyle="-",
            label=f"VV {name} ({result_vv})",
        )

    plt.xlabel("Mv (%)")
    plt.ylabel("σ⁰ (dB)")
    plt.title("Section 4 — Monotonicity vs Moisture (VV & HV)")
    plt.legend(fontsize=8)
    plt.grid(True)
    fname = "section4_monotonicity_vv_hv.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()

    return summary


# -------------------------
# MAIN: run all tests and summarize
# -------------------------
def main():
    iem = IEM_Model()  # default C-band, default texture unless overridden per-case
    overall = {}

    print("\n\n=== RUNNING SECTION 1: FUNG PHYSICAL VALIDATION ===\n")
    fung_res = run_fung_test(iem)
    overall["fung"] = fung_res
    print("Section 1 results:", fung_res)

    print("\n\n=== RUNNING SECTION 2: BAGHDADI EMPIRICAL VALIDATION ===\n")
    baghdadi_res = run_baghdadi_test(iem)
    overall["baghdadi"] = baghdadi_res
    print(
        "Section 2 results (summary): MAE_HV={:.3f} dB RMSE_HV={:.3f} dB".format(
            baghdadi_res["mae_hv"], baghdadi_res["rmse_hv"]
        )
    )

    print("\n\n=== RUNNING SECTION 3: PHYSICAL LIMITS ===\n")
    limits_res = run_limits_test(iem)
    overall["limits"] = limits_res
    print("Section 3 results saved plots and data keys:", list(limits_res.keys()))

    print("\n\n=== RUNNING SECTION 4: MONOTONICITY TEST ===\n")
    mono_res = run_monotonicity_test(iem)
    overall["monotonicity"] = mono_res
    print("Section 4 results (per scenario):")
    for k, v in mono_res.items():
        print(
            f"  {k}: HV_result={v['hv_result']}, VV_result={v['vv_result']}, HV_range={v['hv_range']:.2f} dB"
        )

    # Final summary verdict
    print("\n\n" + "=" * 80)
    print("FINAL SUITE SUMMARY")
    print("=" * 80)
    print(
        "Section 1 — Fung: RMSE VV={:.3f} dB | RMSE HV={:.3f} dB".format(
            overall["fung"]["rmse_vv"], overall["fung"]["rmse_hv"]
        )
    )
    print(
        "Section 2 — Baghdadi: MAE HV={:.3f} dB | RMSE HV={:.3f} dB".format(
            overall["baghdadi"]["mae_hv"], overall["baghdadi"]["rmse_hv"]
        )
    )
    print(
        "Section 3 — Limits: Roughness cases:",
        [d["ks"] for d in overall["limits"]["roughness_data"]],
    )
    print("Section 4 — Monotonicity: per-scenario HV/VV results:")
    for name, data in overall["monotonicity"].items():
        print(
            f"  {name}: HV_nonmono={data['hv_nonmono']} | VV_nonmono={data['vv_nonmono']} | HV_range={data['hv_range']:.2f} dB | VV_range={data['vv_range']:.2f} dB"
        )

    print("\nEnd of validation suite.")


if __name__ == "__main__":
    main()
