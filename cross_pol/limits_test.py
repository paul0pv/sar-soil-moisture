import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from models import IEM_Model, SurfaceRoughness


def run_limits_test():
    print("=" * 80)
    print("Validation Test 3 — IEM Physical Limits and Angular Dependence")
    print("=" * 80)

    freq = 5.405e9
    model = IEM_Model(frequency=freq)
    k = model.k
    print(f"Frequency: {freq / 1e9:.4f} GHz | k = {k:.4e} rad/m")

    # parameter sweeps
    ks_values = [0.3, 1.0, 3.0]  # smooth → rough surfaces
    kl = 6.0
    theta_deg = np.linspace(20, 70, 51)
    theta_rad = np.deg2rad(theta_deg)
    sint, cost = np.sin(theta_rad), np.cos(theta_rad)
    k_z, k_x = k * cost, k * sint

    roughness = SurfaceRoughness()
    eps_r = complex(15.0, 0.0)
    R_v = model._fresnel_v(eps_r, theta_rad)
    f_vv = model._f_vv(eps_r, sint, cost, R_v)
    F_vv = model._F_vv(eps_r, sint, cost)

    # ---- test surface-roughness limits
    plt.figure(figsize=(9, 6))
    for ks in ks_values:
        s_m = ks / k
        L_m = kl / k
        exp_term = np.exp(-2 * (k_z * s_m) ** 2)
        series_sum = np.zeros_like(R_v, dtype=float)
        for n in range(1, model.N_TERMS + 1):
            n_fact = float(factorial(n))
            Wn = roughness.get_spectrum(2 * k_x, L_m, n)
            I_n = (2 * k_z) ** n * f_vv + 0.5 * (k_z ** (2 * n)) * F_vv
            term_n = (np.abs(I_n) ** 2) * Wn / n_fact
            series_sum += np.real(term_n)

        sigma0_lin = (k**2 / (2 * np.pi)) * (cost**2) * exp_term * series_sum
        sigma0_lin = np.clip(np.real(sigma0_lin), 1e-15, None)
        sigma0_dB = 10 * np.log10(sigma0_lin)

        plt.plot(theta_deg, sigma0_dB, lw=2, label=f"k*s = {ks}")

    plt.title("VV Backscatter vs Angle — Roughness Sensitivity (Fung 1992)")
    plt.xlabel("Incidence angle θ (deg)")
    plt.ylabel("σ⁰ (dB)")
    plt.ylim(-40, -5)
    plt.legend()
    plt.grid(True)
    plt.savefig("validation_test3_limits_roughness.png", dpi=150, bbox_inches="tight")
    print("Saved validation_test3_limits_roughness.png")

    # ---- test dielectric limit (dry vs wet)
    mv_values = [5, 20, 40]
    rms_cm = 1.0
    theta_deg = 35
    plt.figure(figsize=(8, 6))
    for mv in mv_values:
        sigma_vv = model.compute_backscatter(mv, rms_cm, theta_deg, polarization="VV")
        sigma_hv = model.compute_backscatter(mv, rms_cm, theta_deg, polarization="HV")
        plt.scatter(mv, sigma_vv, c="b")
        plt.scatter(mv, sigma_hv, c="r")
    plt.plot(
        mv_values,
        [model.compute_backscatter(mv, rms_cm, theta_deg, "VV") for mv in mv_values],
        "b-",
        label="VV",
    )
    plt.plot(
        mv_values,
        [model.compute_backscatter(mv, rms_cm, theta_deg, "HV") for mv in mv_values],
        "r--",
        label="HV",
    )
    plt.xlabel("Volumetric moisture Mv (%)")
    plt.ylabel("σ⁰ (dB)")
    plt.title("σ⁰ vs Moisture — Dielectric Limit Behavior")
    plt.grid(True)
    plt.legend()
    plt.savefig("validation_test3_limits_moisture.png", dpi=150, bbox_inches="tight")
    print("Saved validation_test3_limits_moisture.png")

    print("\nPhysical expectations:")
    print(" • Increasing ks → higher σ⁰ (rougher surface stronger return)")
    print(" • Increasing θ → decreasing σ⁰ (angle attenuation)")
    print(" • Increasing Mv → higher σ⁰ (wet soils reflect more)")
    print("=" * 80)
    print("Result: qualitative physical trends should hold for all tests.")
    print("=" * 80)


if __name__ == "__main__":
    run_limits_test()
