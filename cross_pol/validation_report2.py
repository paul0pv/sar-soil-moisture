import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from models import IEM_Model, SurfaceRoughness
import base64, io, os, datetime


# ===================== UTILITIES =====================
def rmse(a, b):
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def nearest_indices(xvals, targets):
    return [int(np.argmin(np.abs(xvals - t))) for t in targets]


def img_to_base64(fname):
    return base64.b64encode(open(fname, "rb").read()).decode("utf-8")


# ===================== TESTS =====================
def run_fung_test(iem, config_label):
    freq = iem.frequency
    k = iem.k
    ks, kl = 1.0, 6.0
    s_m, L_m = ks / k, kl / k
    theta_deg = np.linspace(20.0, 70.0, 51)
    eps_r = complex(15.0, 0.0)
    theta_rad = np.deg2rad(theta_deg)
    sint, cost = np.sin(theta_rad), np.cos(theta_rad)
    k_z, k_x = k * cost, k * sint
    rough = SurfaceRoughness()
    R_h = iem._fresnel_h(eps_r, theta_rad)
    R_v = iem._fresnel_v(eps_r, theta_rad)

    def white_box(f_func, F_func):
        exp_term = np.exp(-2.0 * (k_z * s_m) ** 2)
        series_sum = np.zeros_like(theta_deg, dtype=float)
        for n in range(1, iem.N_TERMS + 1):
            n_fact = float(factorial(n))
            Wn = rough.get_spectrum(2.0 * k_x, L_m, n)
            I_n = (2.0 * k_z) ** n * f_func + 0.5 * (k_z ** (2 * n)) * F_func
            series_sum += (1.0 / n_fact) * np.abs(I_n) ** 2 * Wn
        sigma = (k**2 / (2.0 * np.pi)) * (cost**2) * exp_term * series_sum
        return 10 * np.log10(np.clip(np.real(sigma), 1e-20, None))

    hv = white_box(iem._f_hv(eps_r, sint, cost, R_h, R_v), iem._F_hv(eps_r, sint, cost))
    vv = white_box(iem._f_vv(eps_r, sint, cost, R_v), iem._F_vv(eps_r, sint, cost))

    ref_theta = np.array([20, 30, 40, 50, 60, 70])
    ref_vv = np.array([-13.0, -15.5, -18.0, -20.5, -23.5, -27.0])
    ref_hv = ref_vv - 12.0
    idx = nearest_indices(theta_deg, ref_theta)
    rmse_vv, rmse_hv = rmse(vv[idx], ref_vv), rmse(hv[idx], ref_hv)

    plt.figure(figsize=(9, 6))
    plt.plot(theta_deg, vv, label="VV (computed)", lw=2)
    plt.plot(theta_deg, hv, label="HV (computed)", lw=2)
    plt.plot(ref_theta, ref_vv, "rs--", label="VV ref")
    plt.plot(ref_theta, ref_hv, "bo--", label="HV ref")
    plt.xlabel("Angle (°)")
    plt.ylabel("σ⁰ (dB)")
    plt.title(f"Fung (1992) Validation – {config_label}")
    plt.legend()
    plt.grid(True)
    plt.ylim(-50, 5)
    fpath = f"fung_{config_label.replace(' ', '_')}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    return {
        "rmse_vv": float(rmse_vv),
        "rmse_hv": float(rmse_hv),
        "img": img_to_base64(fpath),
    }


def run_baghdadi_test(iem, config_label):
    cases = [
        {
            "id": "A11",
            "mv": 16.9,
            "rms_cm": 1.01,
            "theta": 38.6,
            "sand": 19.5,
            "clay": 44.2,
            "exp": -24.8,
        },
        {
            "id": "B12",
            "mv": 23.7,
            "rms_cm": 1.63,
            "theta": 38.6,
            "sand": 19.5,
            "clay": 44.2,
            "exp": -23.5,
        },
        {
            "id": "C08",
            "mv": 20.3,
            "rms_cm": 0.82,
            "theta": 38.8,
            "sand": 19.5,
            "clay": 44.2,
            "exp": -25.2,
        },
    ]
    results = []
    for c in cases:
        model = IEM_Model(
            frequency=iem.frequency, sand_pct=c["sand"], clay_pct=c["clay"]
        )
        hv = model.compute_backscatter(c["mv"], c["rms_cm"], c["theta"], "HV")
        vv = model.compute_backscatter(c["mv"], c["rms_cm"], c["theta"], "VV")
        diff = hv - c["exp"]
        results.append(
            {
                "id": c["id"],
                "calc_hv": float(hv),
                "calc_vv": float(vv),
                "exp": c["exp"],
                "diff": float(diff),
            }
        )
    diffs = np.array([r["diff"] for r in results])
    mae, rms = float(np.mean(np.abs(diffs))), float(np.sqrt(np.mean(diffs**2)))

    plt.figure(figsize=(7, 4))
    ids = [r["id"] for r in results]
    hv = [r["calc_hv"] for r in results]
    vv = [r["calc_vv"] for r in results]
    plt.plot(ids, hv, "o-", label="HV calc")
    plt.plot(ids, vv, "s--", label="VV calc")
    plt.xlabel("Case")
    plt.ylabel("σ⁰ (dB)")
    plt.title(f"Baghdadi Validation – {config_label}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fpath = f"baghdadi_{config_label.replace(' ', '_')}.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    return {"mae": mae, "rmse": rms, "cases": results, "img": img_to_base64(fpath)}


def run_limits_test(iem, config_label):
    ks_vals = [0.3, 1.0, 3.0]
    kl = 6.0
    k = iem.k
    theta = np.linspace(20, 70, 51)
    plt.figure(figsize=(8, 5))
    for ks in ks_vals:
        s = ks / k
        vv = iem.compute_backscatter(20.0, s * 100, theta, "VV")
        hv = iem.compute_backscatter(20.0, s * 100, theta, "HV")
        plt.plot(theta, vv, label=f"VV ks={ks}")
        plt.plot(theta, hv, "--", label=f"HV ks={ks}")
    plt.xlabel("Angle (°)")
    plt.ylabel("σ⁰ (dB)")
    plt.title(f"Physical Limits – {config_label}")
    plt.grid(True)
    plt.legend()
    fpath = f"limits_{config_label.replace(' ', '_')}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    return {"img": img_to_base64(fpath)}


def run_monotonicity_test(iem, config_label):
    mv = np.linspace(1, 45, 88)
    scen = [(0.5, 30, "blue"), (3.0, 45, "red")]
    plt.figure(figsize=(8, 5))
    results = []
    for rms, theta, c in scen:
        hv = iem.compute_backscatter(mv, rms, theta, "HV")
        vv = iem.compute_backscatter(mv, rms, theta, "VV")
        plt.plot(mv, hv, "--", color=c, label=f"HV s={rms} θ={theta}")
        plt.plot(mv, vv, "-", color=c, alpha=0.7, label=f"VV s={rms} θ={theta}")
        nonmono_hv = int(np.sum(np.diff(hv) < -1e-3))
        nonmono_vv = int(np.sum(np.diff(vv) < -1e-3))
        results.append(
            {
                "rms": rms,
                "theta": theta,
                "hv_nonmono": nonmono_hv,
                "vv_nonmono": nonmono_vv,
            }
        )
    plt.xlabel("Mv (%)")
    plt.ylabel("σ⁰ (dB)")
    plt.title(f"Monotonicity vs Moisture – {config_label}")
    plt.legend(fontsize=9)
    plt.grid(True)
    fpath = f"monotonicity_{config_label.replace(' ', '_')}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    return {"results": results, "img": img_to_base64(fpath)}


# ===================== HTML REPORT =====================
def generate_html_report(all_data):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>IEM-B Multi-Frequency Validation</title>",
        "<style>body{font-family:Arial;margin:40px;background:#fafafa;color:#111}h1{color:#063;}h2{color:#046;}",
        "details{margin-bottom:25px;} summary{font-weight:bold;font-size:1.1em;} img{max-width:100%;margin:10px 0;border:1px solid #ccc;} ",
        "table{border-collapse:collapse;width:100%;margin:10px 0;}td,th{border:1px solid #999;padding:4px;text-align:center;} </style></head><body>",
        f"<h1>IEM-B Validation Report (Multi-Configuration)</h1><p>Generated: {now}</p>",
    ]

    for label, data in all_data.items():
        html.append(f"<details open><summary>Configuration: {label}</summary>")
        html.append(
            f"<h3>Fung (1992) Validation</h3><p>RMSE VV={data['fung']['rmse_vv']:.2f} | RMSE HV={data['fung']['rmse_hv']:.2f}</p><img src='data:image/png;base64,{data['fung']['img']}'>"
        )
        html.append(
            f"<h3>Baghdadi (2011) Validation</h3><p>MAE={data['baghdadi']['mae']:.2f} | RMSE={data['baghdadi']['rmse']:.2f}</p><img src='data:image/png;base64,{data['baghdadi']['img']}'>"
        )
        html.append(
            f"<h3>Physical Limits</h3><img src='data:image/png;base64,{data['limits']['img']}'>"
        )
        html.append(
            f"<h3>Monotonicity vs Moisture</h3><img src='data:image/png;base64,{data['monotonicity']['img']}'>"
        )
        html.append("</details>")

    html.append("</body></html>")
    with open("validation_report_multi.html", "w") as f:
        f.write("\n".join(html))
    print("✅ Report saved as validation_report_multi.html")


# ===================== MAIN =====================
def main():
    configs = [
        (5.405e9, 40, 30, "C-band 5.4GHz Loam"),
        (1.25e9, 35, 40, "L-band 1.25GHz Clay"),
        (10.0e9, 60, 20, "X-band 10GHz Sandy"),
    ]

    all_data = {}
    for freq, sand, clay, label in configs:
        print(f"Running config: {label}")
        iem = IEM_Model(frequency=freq, sand_pct=sand, clay_pct=clay)
        all_data[label] = {
            "fung": run_fung_test(iem, label),
            "baghdadi": run_baghdadi_test(iem, label),
            "limits": run_limits_test(iem, label),
            "monotonicity": run_monotonicity_test(iem, label),
        }
    generate_html_report(all_data)
    print("All done.")


if __name__ == "__main__":
    main()
from scipy.special import factorial
from models import IEM_Model, SurfaceRoughness
import base64, io, os, datetime


# ===================== UTILITIES =====================
def rmse(a, b):
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def nearest_indices(xvals, targets):
    return [int(np.argmin(np.abs(xvals - t))) for t in targets]


def img_to_base64(fname):
    return base64.b64encode(open(fname, "rb").read()).decode("utf-8")


# ===================== TESTS =====================
def run_fung_test(iem, config_label):
    freq = iem.frequency
    k = iem.k
    ks, kl = 1.0, 6.0
    s_m, L_m = ks / k, kl / k
    theta_deg = np.linspace(20.0, 70.0, 51)
    eps_r = complex(15.0, 0.0)
    theta_rad = np.deg2rad(theta_deg)
    sint, cost = np.sin(theta_rad), np.cos(theta_rad)
    k_z, k_x = k * cost, k * sint
    rough = SurfaceRoughness()
    R_h = iem._fresnel_h(eps_r, theta_rad)
    R_v = iem._fresnel_v(eps_r, theta_rad)

    def white_box(f_func, F_func):
        exp_term = np.exp(-2.0 * (k_z * s_m) ** 2)
        series_sum = np.zeros_like(theta_deg, dtype=float)
        for n in range(1, iem.N_TERMS + 1):
            n_fact = float(factorial(n))
            Wn = rough.get_spectrum(2.0 * k_x, L_m, n)
            I_n = (2.0 * k_z) ** n * f_func + 0.5 * (k_z ** (2 * n)) * F_func
            series_sum += (1.0 / n_fact) * np.abs(I_n) ** 2 * Wn
        sigma = (k**2 / (2.0 * np.pi)) * (cost**2) * exp_term * series_sum
        return 10 * np.log10(np.clip(np.real(sigma), 1e-20, None))

    hv = white_box(iem._f_hv(eps_r, sint, cost, R_h, R_v), iem._F_hv(eps_r, sint, cost))
    vv = white_box(iem._f_vv(eps_r, sint, cost, R_v), iem._F_vv(eps_r, sint, cost))

    ref_theta = np.array([20, 30, 40, 50, 60, 70])
    ref_vv = np.array([-13.0, -15.5, -18.0, -20.5, -23.5, -27.0])
    ref_hv = ref_vv - 12.0
    idx = nearest_indices(theta_deg, ref_theta)
    rmse_vv, rmse_hv = rmse(vv[idx], ref_vv), rmse(hv[idx], ref_hv)

    plt.figure(figsize=(9, 6))
    plt.plot(theta_deg, vv, label="VV (computed)", lw=2)
    plt.plot(theta_deg, hv, label="HV (computed)", lw=2)
    plt.plot(ref_theta, ref_vv, "rs--", label="VV ref")
    plt.plot(ref_theta, ref_hv, "bo--", label="HV ref")
    plt.xlabel("Angle (°)")
    plt.ylabel("σ⁰ (dB)")
    plt.title(f"Fung (1992) Validation – {config_label}")
    plt.legend()
    plt.grid(True)
    plt.ylim(-50, 5)
    fpath = f"fung_{config_label.replace(' ', '_')}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    return {
        "rmse_vv": float(rmse_vv),
        "rmse_hv": float(rmse_hv),
        "img": img_to_base64(fpath),
    }


def run_baghdadi_test(iem, config_label):
    cases = [
        {
            "id": "A11",
            "mv": 16.9,
            "rms_cm": 1.01,
            "theta": 38.6,
            "sand": 19.5,
            "clay": 44.2,
            "exp": -24.8,
        },
        {
            "id": "B12",
            "mv": 23.7,
            "rms_cm": 1.63,
            "theta": 38.6,
            "sand": 19.5,
            "clay": 44.2,
            "exp": -23.5,
        },
        {
            "id": "C08",
            "mv": 20.3,
            "rms_cm": 0.82,
            "theta": 38.8,
            "sand": 19.5,
            "clay": 44.2,
            "exp": -25.2,
        },
    ]
    results = []
    for c in cases:
        model = IEM_Model(
            frequency=iem.frequency, sand_pct=c["sand"], clay_pct=c["clay"]
        )
        hv = model.compute_backscatter(c["mv"], c["rms_cm"], c["theta"], "HV")
        vv = model.compute_backscatter(c["mv"], c["rms_cm"], c["theta"], "VV")
        diff = hv - c["exp"]
        results.append(
            {
                "id": c["id"],
                "calc_hv": float(hv),
                "calc_vv": float(vv),
                "exp": c["exp"],
                "diff": float(diff),
            }
        )
    diffs = np.array([r["diff"] for r in results])
    mae, rms = float(np.mean(np.abs(diffs))), float(np.sqrt(np.mean(diffs**2)))

    plt.figure(figsize=(7, 4))
    ids = [r["id"] for r in results]
    hv = [r["calc_hv"] for r in results]
    vv = [r["calc_vv"] for r in results]
    plt.plot(ids, hv, "o-", label="HV calc")
    plt.plot(ids, vv, "s--", label="VV calc")
    plt.xlabel("Case")
    plt.ylabel("σ⁰ (dB)")
    plt.title(f"Baghdadi Validation – {config_label}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fpath = f"baghdadi_{config_label.replace(' ', '_')}.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    return {"mae": mae, "rmse": rms, "cases": results, "img": img_to_base64(fpath)}


def run_limits_test(iem, config_label):
    ks_vals = [0.3, 1.0, 3.0]
    kl = 6.0
    k = iem.k
    theta = np.linspace(20, 70, 51)
    plt.figure(figsize=(8, 5))
    for ks in ks_vals:
        s = ks / k
        vv = iem.compute_backscatter(20.0, s * 100, theta, "VV")
        hv = iem.compute_backscatter(20.0, s * 100, theta, "HV")
        plt.plot(theta, vv, label=f"VV ks={ks}")
        plt.plot(theta, hv, "--", label=f"HV ks={ks}")
    plt.xlabel("Angle (°)")
    plt.ylabel("σ⁰ (dB)")
    plt.title(f"Physical Limits – {config_label}")
    plt.grid(True)
    plt.legend()
    fpath = f"limits_{config_label.replace(' ', '_')}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    return {"img": img_to_base64(fpath)}


def run_monotonicity_test(iem, config_label):
    mv = np.linspace(1, 45, 88)
    scen = [(0.5, 30, "blue"), (3.0, 45, "red")]
    plt.figure(figsize=(8, 5))
    results = []
    for rms, theta, c in scen:
        hv = iem.compute_backscatter(mv, rms, theta, "HV")
        vv = iem.compute_backscatter(mv, rms, theta, "VV")
        plt.plot(mv, hv, "--", color=c, label=f"HV s={rms} θ={theta}")
        plt.plot(mv, vv, "-", color=c, alpha=0.7, label=f"VV s={rms} θ={theta}")
        nonmono_hv = int(np.sum(np.diff(hv) < -1e-3))
        nonmono_vv = int(np.sum(np.diff(vv) < -1e-3))
        results.append(
            {
                "rms": rms,
                "theta": theta,
                "hv_nonmono": nonmono_hv,
                "vv_nonmono": nonmono_vv,
            }
        )
    plt.xlabel("Mv (%)")
    plt.ylabel("σ⁰ (dB)")
    plt.title(f"Monotonicity vs Moisture – {config_label}")
    plt.legend(fontsize=9)
    plt.grid(True)
    fpath = f"monotonicity_{config_label.replace(' ', '_')}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    return {"results": results, "img": img_to_base64(fpath)}


# ===================== HTML REPORT =====================
def generate_html_report(all_data):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>IEM-B Multi-Frequency Validation</title>",
        "<style>body{font-family:Arial;margin:40px;background:#fafafa;color:#111}h1{color:#063;}h2{color:#046;}",
        "details{margin-bottom:25px;} summary{font-weight:bold;font-size:1.1em;} img{max-width:100%;margin:10px 0;border:1px solid #ccc;} ",
        "table{border-collapse:collapse;width:100%;margin:10px 0;}td,th{border:1px solid #999;padding:4px;text-align:center;} </style></head><body>",
        f"<h1>IEM-B Validation Report (Multi-Configuration)</h1><p>Generated: {now}</p>",
    ]

    for label, data in all_data.items():
        html.append(f"<details open><summary>Configuration: {label}</summary>")
        html.append(
            f"<h3>Fung (1992) Validation</h3><p>RMSE VV={data['fung']['rmse_vv']:.2f} | RMSE HV={data['fung']['rmse_hv']:.2f}</p><img src='data:image/png;base64,{data['fung']['img']}'>"
        )
        html.append(
            f"<h3>Baghdadi (2011) Validation</h3><p>MAE={data['baghdadi']['mae']:.2f} | RMSE={data['baghdadi']['rmse']:.2f}</p><img src='data:image/png;base64,{data['baghdadi']['img']}'>"
        )
        html.append(
            f"<h3>Physical Limits</h3><img src='data:image/png;base64,{data['limits']['img']}'>"
        )
        html.append(
            f"<h3>Monotonicity vs Moisture</h3><img src='data:image/png;base64,{data['monotonicity']['img']}'>"
        )
        html.append("</details>")

    html.append("</body></html>")
    with open("validation_report_multi.html", "w") as f:
        f.write("\n".join(html))
    print("✅ Report saved as validation_report_multi.html")


# ===================== MAIN =====================
def main():
    configs = [
        (5.405e9, 40, 30, "C-band 5.4GHz Loam"),
        (1.25e9, 35, 40, "L-band 1.25GHz Clay"),
        (10.0e9, 60, 20, "X-band 10GHz Sandy"),
    ]

    all_data = {}
    for freq, sand, clay, label in configs:
        print(f"Running config: {label}")
        iem = IEM_Model(frequency=freq, sand_pct=sand, clay_pct=clay)
        all_data[label] = {
            "fung": run_fung_test(iem, label),
            "baghdadi": run_baghdadi_test(iem, label),
            "limits": run_limits_test(iem, label),
            "monotonicity": run_monotonicity_test(iem, label),
        }
    generate_html_report(all_data)
    print("All done.")


if __name__ == "__main__":
    main()
