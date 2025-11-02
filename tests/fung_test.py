import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from models import IEM_Model, SurfaceRoughness


def run_fung_test():
    """
    Validation Test 1 (extended): Physical validation vs Fung (1992)
    - White-box HV cross-pol series (no soil model / calibration)
    - White-box VV co-pol series compared to Fung Fig.2 approximate values
    """
    print("=" * 80)
    print("Test 1: Fung (1992) physical validation — HV (cross-pol) & VV (co-pol)")
    print("=" * 80)

    # --- Shared reference / toy geometry ---
    freq = 5.405e9  # C-band
    model_shell = IEM_Model(frequency=freq)
    k = model_shell.k

    # dimensionless test parameters similar to Fung Fig.2
    ks = 1.0  # k*s
    kl = 6.0  # k*L

    s_m = ks / k
    L_m = kl / k

    print(f"Frequency: {freq / 1e9:.4f} GHz, k = {k:.4e} rad/m")
    print(
        f"k*s = {ks} -> s = {s_m * 100:.4f} cm; k*L = {kl} -> L = {L_m * 100:.4f} cm\n"
    )

    # angle sweep
    theta_deg = np.linspace(20.0, 70.0, 51)
    theta_rad = np.deg2rad(theta_deg)
    sint = np.sin(theta_rad)
    cost = np.cos(theta_rad)
    k_z = k * cost
    k_x = k * sint

    roughness = SurfaceRoughness()

    # -----------------------
    # PART A — HV (cross-pol)
    # -----------------------
    print("\n" + "-" * 60)
    print("PART A — HV (cross-polarized) white-box series")
    print("-" * 60)

    eps_r = complex(15.0, 0.0)  # toy dielectric
    R_h = model_shell._fresnel_h(eps_r, theta_rad)
    R_v = model_shell._fresnel_v(eps_r, theta_rad)
    f_hv = model_shell._f_hv(eps_r, sint, cost, R_h, R_v)
    F_hv = model_shell._F_hv(eps_r, sint, cost)

    print(
        f"mean |f_hv| = {np.mean(np.abs(f_hv)):.6e}, mean |F_hv| = {np.mean(np.abs(F_hv)):.6e}"
    )

    exp_term = np.exp(-2.0 * (k_z * s_m) ** 2)
    series_sum_hv = np.zeros_like(R_h, dtype=float)

    for n in range(1, model_shell.N_TERMS + 1):
        n_fact = float(factorial(n))
        Wn = roughness.get_spectrum(2.0 * k_x, L_m, n)
        I_n = (2.0 * k_z) ** n * f_hv + 0.5 * (k_z ** (2 * n)) * F_hv
        term_n = (s_m ** (2 * n) / n_fact) * (np.abs(I_n) ** 2) * Wn
        series_sum_hv += np.real(term_n)

    sigma0_lin_hv = 0.5 * (k**2) * exp_term * series_sum_hv
    sigma0_lin_hv = np.where(sigma0_lin_hv <= 1e-20, 1e-20, sigma0_lin_hv)
    sigma0_dB_hv = 10.0 * np.log10(sigma0_lin_hv)

    # Reference HV from VV-Fung offset (approx)
    ref_theta = np.array([20, 30, 40, 50, 60, 70])
    ref_sigma_vv = np.array(
        [-13.0, -15.5, -18.0, -20.5, -23.5, -27.0]
    )  # Fung Fig.2 approximate
    ref_sigma_hv = ref_sigma_vv - 12.0  # rule-of-thumb

    # Compare at nearest angles
    diffs_hv = []
    print("\nAngle  Calc_HV(dB)  Ref_HV(dB)  Diff(dB)")
    for ang, ref in zip(ref_theta, ref_sigma_hv):
        idx = np.argmin(np.abs(theta_deg - ang))
        calc = sigma0_dB_hv[idx]
        diff = calc - ref
        diffs_hv.append(diff)
        print(f"{ang:5.0f}  {calc:12.2f}  {ref:10.2f}  {diff:9.2f}")

    rmse_hv = np.sqrt(np.mean(np.square(diffs_hv)))
    print(f"\nHV RMSE vs ref: {rmse_hv:.3f} dB")

    # Physical checks HV
    monotonic_hv = np.all(np.diff(sigma0_dB_hv) < -1e-6)
    range_ok_hv = (np.min(sigma0_dB_hv) > -40.0) and (np.max(sigma0_dB_hv) < -10.0)
    print(f"HV monotonic decrease with angle: {'PASS' if monotonic_hv else 'FAIL'}")
    print(f"HV magnitude range check: {'PASS' if range_ok_hv else 'FAIL'}")

    # Save HV plot
    plt.figure(figsize=(8, 5))
    plt.plot(theta_deg, sigma0_dB_hv, "r-", lw=2, label="IEM computed HV")
    plt.plot(ref_theta, ref_sigma_hv, "bo--", label="Reference HV (VV-12dB)")
    plt.plot(
        ref_theta, ref_sigma_vv, "gs--", alpha=0.6, label="Reference VV (Fung Fig.2)"
    )
    plt.xlabel("Incidence angle (deg)")
    plt.ylabel("σ⁰ (dB)")
    plt.title("Validation Test 1 — HV cross-pol (Fung reference)")
    plt.grid(True)
    plt.legend()
    plt.ylim(-45, -5)
    plt.savefig("validation_test1_fung_hv.png", dpi=150, bbox_inches="tight")
    print("Saved validation_test1_fung_hv.png")
    plt.close()

    # -----------------------
    # PART B — VV (co-polarized) white-box series
    # -----------------------
    print("\n" + "-" * 60)
    print("PART B — VV (co-polarized) white-box series")
    print("-" * 60)

    # For VV, use same toy dielectric but compute f_vv and F_vv
    eps_r = complex(15.0, 0.0)
    R_h = model_shell._fresnel_h(eps_r, theta_rad)
    R_v = model_shell._fresnel_v(eps_r, theta_rad)

    # f_vv and F_vv from IEM version (simplified)
    f_vv = model_shell._f_vv(eps_r, sint, cost, R_v)
    F_vv = model_shell._F_vv(eps_r, sint, cost)

    print(
        f"mean |f_vv| = {np.mean(np.abs(f_vv)):.6e}, mean |F_vv| = {np.mean(np.abs(F_vv)):.6e}"
    )

    exp_term = np.exp(-2.0 * (k_z * s_m) ** 2)
    series_sum_vv = np.zeros_like(R_v, dtype=float)

    for n in range(1, model_shell.N_TERMS + 1):
        n_fact = float(factorial(n))
        Wn = roughness.get_spectrum(2.0 * k_x, L_m, n)
        I_n_vv = (2.0 * k_z) ** n * f_vv + 0.5 * (k_z ** (2 * n)) * F_vv
        term_n = (s_m ** (2 * n) / n_fact) * (np.abs(I_n_vv) ** 2) * Wn
        series_sum_vv += np.real(term_n)

    sigma0_lin_vv = 0.5 * (k**2) * exp_term * series_sum_vv
    sigma0_lin_vv = np.where(sigma0_lin_vv <= 1e-20, 1e-20, sigma0_lin_vv)
    sigma0_dB_vv = 10.0 * np.log10(sigma0_lin_vv)

    # Compare to Fung Fig.2 approximate reference (we used ref_sigma_vv above)
    diffs_vv = []
    print("\nAngle  Calc_VV(dB)  Ref_VV(dB)  Diff(dB)")
    for ang, ref in zip(ref_theta, ref_sigma_vv):
        idx = np.argmin(np.abs(theta_deg - ang))
        calc = sigma0_dB_vv[idx]
        diff = calc - ref
        diffs_vv.append(diff)
        print(f"{ang:5.0f}  {calc:12.2f}  {ref:10.2f}  {diff:9.2f}")

    rmse_vv = np.sqrt(np.mean(np.square(diffs_vv)))
    print(f"\nVV RMSE vs Fung Fig.2 approx: {rmse_vv:.3f} dB")

    # Physical checks VV
    monotonic_vv = np.all(np.diff(sigma0_dB_vv) < -1e-6)
    range_ok_vv = (np.min(sigma0_dB_vv) > -35.0) and (np.max(sigma0_dB_vv) < -5.0)
    print(f"VV monotonic decrease with angle: {'PASS' if monotonic_vv else 'FAIL'}")
    print(f"VV magnitude range check: {'PASS' if range_ok_vv else 'FAIL'}")

    # Save VV plot
    plt.figure(figsize=(8, 5))
    plt.plot(theta_deg, sigma0_dB_vv, "b-", lw=2, label="IEM computed VV")
    plt.plot(ref_theta, ref_sigma_vv, "rs--", label="Reference VV (Fung Fig.2 approx)")
    plt.xlabel("Incidence angle (deg)")
    plt.ylabel("σ⁰ (dB)")
    plt.title("Validation Test 1 — VV co-pol (Fung Fig.2 reference)")
    plt.grid(True)
    plt.legend()
    plt.ylim(-35, -5)
    plt.savefig("validation_test1_fung_vv.png", dpi=150, bbox_inches="tight")
    print("Saved validation_test1_fung_vv.png")
    plt.close()

    # Summarize results
    print("\n" + "=" * 60)
    print("Summary:")
    print(
        f"HV RMSE vs ref: {rmse_hv:.3f} dB  | HV monotonic: {monotonic_hv} | HV range_ok: {range_ok_hv}"
    )
    print(
        f"VV RMSE vs ref: {rmse_vv:.3f} dB  | VV monotonic: {monotonic_vv} | VV range_ok: {range_ok_vv}"
    )
    final_pass = (monotonic_hv and range_ok_hv and (rmse_hv < 3.0)) and (
        monotonic_vv and range_ok_vv and (rmse_vv < 3.0)
    )
    print(f"\nFinal overall verdict: {'PASS' if final_pass else 'FAIL'}")
    print("=" * 80)


if __name__ == "__main__":
    run_fung_test()
