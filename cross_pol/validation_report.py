import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from models import IEM_Model, SurfaceRoughness
import base64, io, os, datetime, json


# ===================== UTILITIES =====================
def rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.sqrt(np.mean((a - b) ** 2))


def nearest_indices(xvals, targets):
    return [int(np.argmin(np.abs(xvals - t))) for t in targets]


def img_to_base64(fname):
    with open(fname, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ===================== SECTION 1 – FUNG =====================
def run_fung_test(iem):
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
    plt.title("Section 1 – Fung (1992) Validation (VV & HV)")
    plt.legend()
    plt.grid(True)
    plt.ylim(-50, 5)
    fpath = "section1_fung.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "rmse_vv": float(rmse_vv),
        "rmse_hv": float(rmse_hv),
        "img": img_to_base64(fpath),
    }


# ===================== SECTION 2 – BAGHDADI =====================
def run_baghdadi_test(iem):
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
        {
            "id": "D15",
            "mv": 30.5,
            "rms_cm": 1.25,
            "theta": 40.0,
            "sand": 25.0,
            "clay": 35.0,
            "exp": -22.0,
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
                "exp_hv": c["exp"],
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
    plt.title("Section 2 – Baghdadi (2011) Validation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fpath = "section2_baghdadi.png"
    plt.savefig(fpath, dpi=150)
    plt.close()

    return {"mae": mae, "rmse": rms, "cases": results, "img": img_to_base64(fpath)}


# ===================== SECTION 3 – LIMITS =====================
def run_limits_test(iem):
    ks_vals = [0.3, 1.0, 3.0]
    kl = 6.0
    k = iem.k
    theta = np.linspace(20, 70, 51)
    plt.figure(figsize=(8, 5))
    for ks in ks_vals:
        s = ks / k
        L = kl / k
        vv = iem.compute_backscatter(20.0, s * 100, theta, "VV")
        hv = iem.compute_backscatter(20.0, s * 100, theta, "HV")
        plt.plot(theta, vv, label=f"VV ks={ks}")
        plt.plot(theta, hv, "--", label=f"HV ks={ks}")
    plt.xlabel("Angle (°)")
    plt.ylabel("σ⁰ (dB)")
    plt.title("Section 3 – Physical Limits")
    plt.grid(True)
    plt.legend()
    fpath = "section3_limits.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    return {"img": img_to_base64(fpath)}


# ===================== SECTION 4 – MONOTONICITY =====================
def run_monotonicity_test(iem):
    mv = np.linspace(1, 45, 88)
    scen = [
        (0.5, 30, "blue"),
        (1.5, 30, "orange"),
        (3.0, 30, "green"),
        (3.0, 45, "red"),
    ]
    plt.figure(figsize=(9, 6))
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
    plt.title("Section 4 – Monotonicity vs Moisture (VV & HV)")
    plt.legend(fontsize=8)
    plt.grid(True)
    fpath = "section4_monotonicity.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    return {"results": results, "img": img_to_base64(fpath)}


# ===================== REPORT GENERATOR =====================
def generate_html_report(data, out_file="validation_report.html"):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def img_tag(b64):
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #ccc;margin:10px 0;">'

    html = f"""<!DOCTYPE html><html><head>
<meta charset='utf-8'><title>IEM-B Validation Report</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:40px;background:#fafafa;color:#222;}}
h1{{color:#0a3;}} h2{{border-bottom:2px solid #ccc;padding-bottom:3px;}}
table{{border-collapse:collapse;width:100%;margin:10px 0;}} td,th{{border:1px solid #999;padding:5px;text-align:center;}}
</style></head><body>
<h1>IEM-B Model Validation Report</h1>
<p>Generated: {now}</p>

<h2>Section 1 – Fung (1992)</h2>
<p>RMSE VV: {data["fung"]["rmse_vv"]:.3f} dB | RMSE HV: {data["fung"]["rmse_hv"]:.3f} dB</p>
{img_tag(data["fung"]["img"])}

<h2>Section 2 – Baghdadi (2011)</h2>
<p>MAE HV: {data["baghdadi"]["mae"]:.3f} dB | RMSE HV: {data["baghdadi"]["rmse"]:.3f} dB</p>
<table><tr><th>Case</th><th>Calc HV</th><th>Exp HV</th><th>Δ (dB)</th></tr>
"""
    for c in data["baghdadi"]["cases"]:
        html += f"<tr><td>{c['id']}</td><td>{c['calc_hv']:.2f}</td><td>{c['exp_hv']:.2f}</td><td>{c['diff']:.2f}</td></tr>"
    html += "</table>" + img_tag(data["baghdadi"]["img"])

    html += """<h2>Section 3 – Physical Limits</h2>""" + img_tag(data["limits"]["img"])

    html += """<h2>Section 4 – Monotonicity vs Moisture</h2>
<table><tr><th>RMS (cm)</th><th>θ (°)</th><th>HV non-monotonic pts</th><th>VV non-monotonic pts</th></tr>"""
    for r in data["monotonicity"]["results"]:
        html += f"<tr><td>{r['rms']}</td><td>{r['theta']}</td><td>{r['hv_nonmono']}</td><td>{r['vv_nonmono']}</td></tr>"
    html += "</table>" + img_tag(data["monotonicity"]["img"])

    html += "</body></html>"
    with open(out_file, "w") as f:
        f.write(html)
    print(f"HTML report saved: {out_file}")


# ===================== MAIN =====================
def main():
    iem = IEM_Model()
    print("Running IEM-B validation suite...")
    data = {
        "fung": run_fung_test(iem),
        "baghdadi": run_baghdadi_test(iem),
        "limits": run_limits_test(iem),
        "monotonicity": run_monotonicity_test(iem),
    }
    generate_html_report(data)
    print("All done.")


if __name__ == "__main__":
    main()
