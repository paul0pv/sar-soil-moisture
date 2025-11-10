import numpy as np
from scipy.special import factorial

# -------------------------------------------------------------------
# CLASE 1: MODELO DIELÉCTRICO (Hallikainen 1985)
# -------------------------------------------------------------------


class DielectricModel:
    """
    Modelo de Hallikainen (1985) para ε_r(mv, textura).

    Calcula la constante dieléctrica compleja del suelo en función de:
    - Humedad volumétrica (mv)
    - Composición textural (arena, arcilla, limo)
    - Frecuencia

    Referencia: Hallikainen et al., IEEE TGRS, Vol. GE-23, No. 1, 1985
    Ecuaciones (1)-(4), Tablas IV y V
    """

    def __init__(self, sand_pct: float, clay_pct: float):
        """
        Inicializa el modelo dieléctrico.

        Args:
            sand_pct: Porcentaje de arena (0-100)
            clay_pct: Porcentaje de arcilla (0-100)
        """
        s = float(sand_pct); c = float(clay_pct)
        if s < 0 or c < 0 or s + c > 100:
            raise ValueError("Textura inválida: arena>=0, arcilla>=0 y arena+arcilla<=100.")
        self.sand = s
        self.clay = c
        self.silt = 100.0 - s - c

        # Coeficientes de regresión para Epsilon Prima (Parte Real) - Tabla IV
        # Estructura: coef = alpha + beta*f + gamma*f² (f en GHz)
        self.coeffs_real = {
            "a0": {"alpha": 3.51, "beta": -6.83e-2, "gamma": 1.05e-3},
            "a1": {"alpha": -2.96e-2, "beta": 5.21e-4, "gamma": -6.3e-6},
            "a2": {"alpha": -4.54e-2, "beta": 9.24e-4, "gamma": -1.43e-5},
            "b0": {"alpha": 2.06e1, "beta": -1.04, "gamma": 2.04e-2},
            "b1": {"alpha": -2.14e-1, "beta": 1.13e-2, "gamma": -2.18e-4},
            "b2": {"alpha": -2.85e-1, "beta": 1.63e-2, "gamma": -3.22e-4},
            "c0": {"alpha": 1.11e1, "beta": 1.48, "gamma": -4.27e-2},
            "c1": {"alpha": 4.11e-1, "beta": -2.25e-2, "gamma": 4.34e-4},
            "c2": {"alpha": 5.21e-1, "beta": -3.37e-2, "gamma": 6.7e-4},
        }

        # Coeficientes para Epsilon Doble Prima (Parte Imaginaria) - Tabla V
        self.coeffs_imag = {
            "d0": {"alpha": 0.63, "beta": -2.16e-2, "gamma": 3.6e-4},
            "d1": {"alpha": -6.22e-3, "beta": 2.22e-4, "gamma": -3.7e-6},
            "d2": {"alpha": -9.7e-3, "beta": 3.7e-4, "gamma": -6.3e-6},
            "e0": {"alpha": 3.14, "beta": 4.09e-1, "gamma": -9.1e-3},
            "e1": {"alpha": -2.59e-2, "beta": -4.4e-3, "gamma": 9.8e-5},
            "e2": {"alpha": -4.31e-2, "beta": -7.5e-3, "gamma": 1.7e-4},
            "f0": {"alpha": 1.81e1, "beta": -1.03, "gamma": 2.1e-2},
            "f1": {"alpha": -2.11e-1, "beta": 1.25e-2, "gamma": -2.6e-4},
            "f2": {"alpha": -3.27e-1, "beta": 1.95e-2, "gamma": -4.0e-4},
        }

    def _poly(self, coeff, f_ghz):
        """
        Evalúa polinomio cuadrático: alpha + beta*f + gamma*f².

        Args:
            coeff: Diccionario con {alpha, beta, gamma}
            f_ghz: Frecuencia en GHz

        Returns:
            Valor del coeficiente
        """
        return coeff["alpha"] + coeff["beta"] * f_ghz + coeff["gamma"] * f_ghz**2

    def compute_dielectric(self, mv, frequency=5.405e9):
        """
        Calcula la constante dieléctrica compleja del suelo.

        Implementa Hallikainen (1985), Ecuaciones (1)-(4):

        ε' = A + B·mv + C·mv²
        ε'' = D + E·mv + F·mv²

        Donde A, B, C (y D, E, F) dependen de textura y frecuencia.

        Args:
            mv: Humedad volumétrica (%) - escalar o array
            frequency: Frecuencia en Hz (default: 5.405 GHz, banda C)

        Returns:
            Constante dieléctrica compleja ε_r = ε' + jε''
        """
        f_ghz = frequency / 1e9
        mv = np.asarray(mv, dtype=np.float64)

        # Normalizar a fracción (0-1)
        mv_frac = np.where(mv > 1.0, mv / 100.0, mv)
        # Clipping razonable para evitar extrapolaciones extremas
        mv_frac = np.clip(mv_frac, 0.0, 0.6)

        S, C = self.sand, self.clay

        # --- PARTE REAL (ε') ---
        a0 = self._poly(self.coeffs_real["a0"], f_ghz)
        a1 = self._poly(self.coeffs_real["a1"], f_ghz)
        a2 = self._poly(self.coeffs_real["a2"], f_ghz)
        b0 = self._poly(self.coeffs_real["b0"], f_ghz)
        b1 = self._poly(self.coeffs_real["b1"], f_ghz)
        b2 = self._poly(self.coeffs_real["b2"], f_ghz)
        c0 = self._poly(self.coeffs_real["c0"], f_ghz)
        c1 = self._poly(self.coeffs_real["c1"], f_ghz)
        c2 = self._poly(self.coeffs_real["c2"], f_ghz)

        A = a0 + a1 * S + a2 * C
        B = b0 + b1 * S + b2 * C
        Cc = c0 + c1 * S + c2 * C

        eps_real = A + B * mv_frac + Cc * mv_frac**2

        # --- PARTE IMAGINARIA (ε'') ---
        d0 = self._poly(self.coeffs_imag["d0"], f_ghz)
        d1 = self._poly(self.coeffs_imag["d1"], f_ghz)
        d2 = self._poly(self.coeffs_imag["d2"], f_ghz)
        e0 = self._poly(self.coeffs_imag["e0"], f_ghz)
        e1 = self._poly(self.coeffs_imag["e1"], f_ghz)
        e2 = self._poly(self.coeffs_imag["e2"], f_ghz)
        f0 = self._poly(self.coeffs_imag["f0"], f_ghz)
        f1 = self._poly(self.coeffs_imag["f1"], f_ghz)
        f2 = self._poly(self.coeffs_imag["f2"], f_ghz)

        D = d0 + d1 * S + d2 * C
        E = e0 + e1 * S + e2 * C
        F = f0 + f1 * S + f2 * C

        eps_imag = D + E * mv_frac + F * mv_frac**2

        return eps_real + 1j * eps_imag


# -------------------------------------------------------------------
# CLASE 2: MODELO DE RUGOSIDAD (Baghdadi 2011 + Fung 1992)
# -------------------------------------------------------------------


class SurfaceRoughness:
    """
    Parametrización de rugosidad con calibración IEM_B (Baghdadi 2011)
    y espectro de potencia Gaussiano (Fung 1992).

    Referencias:
    - Baghdadi et al., IEEE GRSL, Vol. 8, No. 1, 2011
      * Ecuación (2): Lopt2 para HV (polarización cruzada)
      * Ecuación (3): Lopt para VV (polarización co-polarizada)
    - Fung et al., IEEE TGRS, Vol. 30, No. 2, 1992
      * Ecuación (4-A.3): Espectro de potencia Gaussiano
    """

    def __init__(self, correlation="gaussian"):
        """
        Inicializa el modelo de rugosidad.

        Args:
            correlation: Tipo de función de autocorrelación ("gaussian")
        """
        correlation = correlation.lower()
        if correlation not in ("gaussian", "fractal"):
            raise ValueError("correlation debe ser 'gaussian' o 'fractal'.")
        self.correlation_type = correlation
        self.t = 1.33

    def compute_Lopt(self, rms_cm, theta_deg, polarization="VV"):
        """
        Calcula la longitud de correlación óptima calibrada.

        CRÍTICO: Ecuaciones DIFERENTES para VV y HV.

        Para VV (co-polarizada):
        ------------------------
        Baghdadi (2011), Ecuación (3):
        Lopt(s,θ) = 1.281 + 0.134 × [sin(0.19θ)]^(-1.59) × s

        Para HV (cross-polarizada):
        ---------------------------
        Baghdadi (2011), Ecuación (2), Sección V, página 3:
        Lopt2(s,θ) = 0.9157 + 1.2289 × [sin(0.1543θ)]^(-0.3139) × s

        IMPORTANTE: Lopt2 (valor alto) asegura comportamiento físico correcto:
        σ° aumenta con s (en lugar de decrecer para s>1cm con Lopt1).

        Args:
            rms_cm: Altura RMS de rugosidad en cm (escalar o array)
            theta_deg: Ángulo de incidencia en grados (escalar o array)
            polarization: Polarización ("VV", "HV", "VH")

        Returns:
            Lopt en cm (mismo shape que los inputs)
        """
        rms_cm = np.asarray(rms_cm)
        theta_rad = np.deg2rad(theta_deg).astype(np.float64)
        theta_deg_arr = np.asarray(theta_deg)

        if polarization.upper() == "VV":
            # Baghdadi (2006), Ecuación (5), página 812
            d_vv = 3.289
            m = -1.744
            g = -0.0025
            j_vv = 1.222

            sin_term = np.sin(0.19 * theta_rad)
            sin_term = np.where(np.abs(sin_term) < 1e-6, 1e-6, sin_term)

            # Lopt2 = d × sin(θ)^m × rms^(g×θ + j)
            # Lopt = d_vv * (sin_term**m) * (rms_cm ** (g * theta_deg_arr + j_vv))
            Lopt = 1.281 + 0.134 * sin_term ** (-1.59) * rms_cm

        elif polarization.upper() in ("HV", "VH"):
            # Baghdadi (2011), Ecuación (2)
            # (Paper específico para HV, calibrado independientemente)
            sin_term = np.sin(0.1543 * theta_rad)
            sin_term = np.where(np.abs(sin_term) < 1e-6, 1e-6, sin_term)
            Lopt = 0.9157 + 1.2289 * sin_term ** (-0.3139) * rms_cm

        elif polarization.upper() == "HH":
            # Baghdadi (2006), Ecuación (5)
            d_hh = 4.026
            m = -1.744
            g = -0.0025
            j_hh = 1.551

            sin_term = np.sin(1.23 * theta_rad)
            sin_term = np.where(np.abs(sin_term) < 1e-6, 1e-6, sin_term)

            Lopt = 0.162 + 3.006 * sin_term ** (-1.494) * rms_cm

        else:
            raise NotImplementedError(f"Polarización {polarization} no soportada.")

        return np.clip(Lopt, 1e-3, np.inf)

    def get_spectrum(self, k_x, L_m, n):
        """
        Espectro de potencia para función de autocorrelación Gaussiana.

        Implementa Fung (1992), Apéndice 4-A, Ecuación (4-A.3):

        W^(n)(k_x) = (L²/2n) × exp(-k_x²L²/4n)

        NOTA CRÍTICA: El factor 4 en el exponente es esencial.

        Args:
            k_x: Número de onda horizontal = k×sin(θ) (escalar o array)
            L_m: Longitud de correlación en metros (escalar o array)
            n: Orden del término en la serie de scattering (1, 2, ..., N)

        Returns:
            W^(n): Espectro de potencia (mismo shape que inputs)

        Referencias:
        - Fung (1992), página 367, Ecuación (4-A.3)
        """
        Lm2 = L_m**2

        if self.correlation_type == "fractal":
            # Baghdadi (2006), Ecuación (1) con t=1.33
            kx_L = np.abs(k_x * L_m)
            exp_frac = 2.0 / self.t
            norm = n ** (2.0 / self.t)
            Wn = (Lm2 / n) * np.exp(-((kx_L**exp_frac) / norm))

        elif self.correlation_type == "gaussian":
            # Fung (1992), Eq. (4-A.3)
            kx2 = k_x**2
            Wn = (Lm2 / (2.0 * n)) * np.exp(-kx2 * Lm2 / (4.0 * n))

        else:
            raise ValueError(f"Correlación {self.correlation_type} no soportada")

        return Wn


# -------------------------------------------------------------------
# CLASE 3: MODELO IEM (Fung 1992 + Baghdadi 2011)
# -------------------------------------------------------------------


class IEM_Model:
    """
    Modelo de Ecuación Integral calibrado (IEM_B) para banda C.

    Soporta polarizaciones:
    - VV (co-polarizada vertical)
    - HV/VH (cross-polarizada)

    Implementa:
    - Física de scattering: Fung et al. (1992), IEEE TGRS, Vol. 30, No. 2
    - Calibración Lopt: Baghdadi et al. (2011), IEEE GRSL, Vol. 8, No. 1
    - Modelo dieléctrico: Hallikainen et al. (1985), IEEE TGRS, Vol. GE-23, No. 1

    ECUACIONES PRINCIPALES (Fung 1992):
    ====================================

    Ecuación (17) - Coeficiente de scattering:
    ------------------------------------------
    σ⁰ = (k²/2) × exp(-2k_z²s²) × Σ[(W^(n)/n!) × |I_pp^(n)|²]

    Ecuación (18) - Términos de la serie:
    -------------------------------------
    I_pp^(n) = (2k_z·s)^n × f_pp + (k_z·s)^(2n)/2 × F_pp

    Donde:
    - k = 2π/λ: número de onda
    - k_z = k·cos(θ): componente vertical
    - s: altura RMS de rugosidad (metros)
    - W^(n): espectro de potencia de orden n
    - f_pp, F_pp: términos de reflexión (dependen de polarización)
    """

    def __init__(self, frequency=5.405e9, sand_pct=20, clay_pct=30, *, spectrum_mode: str = "gaussian", max_terms: int = 10, use_strict_fvv: bool = False):
        """
        Inicializa el modelo IEM calibrado.

        Args:
            frequency: Frecuencia en Hz (default: 5.405 GHz, Sentinel-1 C-band)
            sand_pct: Porcentaje de arena (0-100)
            clay_pct: Porcentaje de arcilla (0-100)
        """
        self.frequency = float(frequency)
        self.wavelength = 2.99792e8 / frequency  # λ = c/f
        self.k = 2.0 * np.pi / self.wavelength  # Número de onda (rad/m)

        # Submodelos
        self.dielectric = DielectricModel(sand_pct, clay_pct)
        self.roughness = SurfaceRoughness(correlation=spectrum_mode)

        # Número de términos en la serie de scattering
        self.N_TERMS = int(max(1, max_terms))
        self.use_strict_fvv = bool(use_strict_fvv)

    # --- COEFICIENTES DE FRESNEL ---

    def _fresnel_h(self, eps_r, theta_rad):
        """
        Coeficiente de Fresnel para polarización horizontal (H).

        Fung (1992), página 357, después de Ecuación (2):

        R_h = (cos(θ) - √(ε_r - sin²(θ))) / (cos(θ) + √(ε_r - sin²(θ)))

        Args:
            eps_r: Constante dieléctrica compleja
            theta_rad: Ángulo de incidencia en radianes

        Returns:
            R_h: Coeficiente de Fresnel horizontal (complejo)
        """
        cost = np.cos(theta_rad)
        sint2 = np.sin(theta_rad) ** 2
        sqrt_term = np.lib.scimath.sqrt(eps_r - sint2)

        R_h = (cost - sqrt_term) / (cost + sqrt_term)

        return R_h

    def _fresnel_v(self, eps_r, theta_rad):
        """
        Coeficiente de Fresnel para polarización vertical (V).

        Fung (1992), página 357:

        R_v = (ε_r·cos(θ) - √(ε_r - sin²(θ))) / (ε_r·cos(θ) + √(ε_r - sin²(θ)))

        Args:
            eps_r: Constante dieléctrica compleja
            theta_rad: Ángulo de incidencia en radianes

        Returns:
            R_v: Coeficiente de Fresnel vertical (complejo)
        """
        cost = np.cos(theta_rad)
        sint2 = np.sin(theta_rad) ** 2
        sqrt_term = np.lib.scimath.sqrt(eps_r - sint2)

        R_v = (eps_r * cost - sqrt_term) / (eps_r * cost + sqrt_term)

        return R_v

    # --- TÉRMINOS DE SCATTERING VV (CO-POLARIZADA) ---

    def _f_vv(self, eps_r, sint, cost, R_v):
        """
        Término f_vv para polarización co-polarizada VV.

        Fung (1992), Ecuación (22):

        f_vv = 2·R_v / cos(θ)

        Simplificación para IEM: f_vv ≈ (1 + R_v) / 2

        Args:
            eps_r: Constante dieléctrica compleja
            sint: sin(θ)
            cost: cos(θ)
            R_v: Coeficiente de Fresnel vertical

        Returns:
            f_vv: Término de scattering de primer orden
        """
        if self.use_strict_fvv:
            # Protección ante cos(θ) pequeño
            cost_safe = np.where(np.abs(cost) < 1e-6, 1e-6, cost)
            return 2.0 * R_v / cost_safe
        else:
            # Versión simplificada usada en implementaciones IEM rápidas
            return (1.0 + R_v) / 2.0

    def _F_vv(self, eps_r, sint, cost):
        """
        Término F_vv para polarización co-polarizada VV (orden superior).

        Fung (1992), Ecuación (24):

        F_vv = [(ε_r - 1) × sin(θ) × cos(θ)] / [ε_r·cos(θ) + √(ε_r - sin²(θ))]²

        Este término captura scattering múltiple.

        Args:
            eps_r: Constante dieléctrica compleja
            sint: sin(θ)
            cost: cos(θ)

        Returns:
            F_vv: Término de scattering de orden superior
        """
        sint2 = sint**2
        sqrt_term = np.lib.scimath.sqrt(eps_r - sint2)

        num = (eps_r - 1.0) * sint * cost
        den = (eps_r * cost + sqrt_term) ** 2

        # Evitar división por cero
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)

        return num / den

    # --- TÉRMINOS DE SCATTERING HV (CROSS-POLARIZADA) ---

    def _f_hv(self, eps_r, sint, cost, R_h, R_v):
        """
        Término f_hv para polarización cruzada HV.

        Para backscattering (θ_i = θ_s), f_hv = 0 por simetría.

        Fung (1992), página 357:
        "Cross-polarized backscattering coefficients satisfy reciprocity
        and contain only multiple scattering terms."

        Args:
            eps_r: Constante dieléctrica compleja
            sint: sin(θ)
            cost: cos(θ)
            R_h: Coeficiente de Fresnel horizontal
            R_v: Coeficiente de Fresnel vertical

        Returns:
            f_hv: Término de scattering (= 0 para backscatter)
        """
        # En backscattering puro, f_hv = 0 por simetría
        return np.zeros_like(R_h)

    def _F_hv(self, eps_r, sint, cost):
        """
        Término F_hv para polarización cruzada HV (orden superior).

        Este término NO es cero y captura la despolarización causada
        por la rugosidad superficial.

        Basado en Fung (1992), ecuaciones (54)-(57), adaptado para HV:

        F_hv ≈ [(ε_r - 1) × sin(θ) × cos(θ) × (1 + sin²(θ))] /
               [ε_r·cos(θ) + √(ε_r - sin²(θ))]²

        Args:
            eps_r: Constante dieléctrica compleja
            sint: sin(θ)
            cost: cos(θ)

        Returns:
            F_hv: Término de despolarización (scattering múltiple)
        """
        sint2 = sint**2
        sqrt_term = np.lib.scimath.sqrt(eps_r - sint2)

        # Término de despolarización para cross-pol
        num = (eps_r - 1.0) * sint * cost * (1.0 + sint2)
        den = (eps_r * cost + sqrt_term) ** 2

        # Evitar división por cero
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)

        return num / den

    # --- CÁLCULO PRINCIPAL DE BACKSCATTERING ---

    def compute_backscatter(self, mv, rms_cm, theta_deg, polarization="VV"):
        """
        Calcula el coeficiente de retrodispersión σ⁰ en dB.

        CADENA DE PROCESAMIENTO:
        ========================
        1. [Hallikainen 1985]: mv + textura → ε_r
        2. [Baghdadi 2011]: rms + θ → Lopt (calibrado por polarización)
        3. [Fung 1992 Eq. 17-18]: ε_r + rugosidad → σ⁰

        ECUACIÓN IMPLEMENTADA (Fung 1992, Eq. 17):
        ===========================================
        σ⁰ = (k²/2) × exp(-2k_z²s²) × Σ[(W^(n)/n!) × |I_pp^(n)|²]

        Donde (Fung 1992, Eq. 18):
        I_pp^(n) = (2k_z·s)^n × f_pp + (k_z·s)^(2n)/2 × F_pp

        CRÍTICO: El factor s (altura RMS en metros) DEBE estar presente.
        Sin s, los términos (k_z)^n explotan exponencialmente.

        DIFERENCIAS POR POLARIZACIÓN:
        =============================

        VV (Co-polarizada):
        ------------------
        - f_vv ≠ 0 (término dominante)
        - F_vv contribuye (scattering múltiple)
        - Típicamente σ⁰_VV ∈ [-20, -10] dB

        HV (Cross-polarizada):
        ---------------------
        - f_hv = 0 (por simetría en backscatter)
        - F_hv dominante (única contribución)
        - Típicamente σ⁰_HV ∈ [-35, -20] dB
        - σ⁰_HV < σ⁰_VV siempre (8-15 dB menor)

        Args:
            mv: Humedad volumétrica (%) - escalar o array
            rms_cm: Altura RMS de rugosidad (cm) - escalar o array
            theta_deg: Ángulo de incidencia (grados) - escalar o array
            polarization: "VV", "HV", o "VH"

        Returns:
            σ⁰ en dB (mismo shape que los inputs)

        Referencias:
        -----------
        - Fung et al., IEEE TGRS, Vol. 30, No. 2, March 1992
        - Baghdadi et al., IEEE GRSL, Vol. 8, No. 1, January 2011
        - Hallikainen et al., IEEE TGRS, Vol. GE-23, No. 1, January 1985
        """
        # --- 1. VECTORIZACIÓN Y NORMALIZACIÓN ---
        mv = np.asarray(mv, dtype=np.float64)
        rms_cm = np.asarray(rms_cm, dtype=np.float64)
        theta_rad = np.deg2rad(theta_deg).astype(np.float64)
        polarization = polarization.upper()

        # --- 2. PARÁMETROS GEOMÉTRICOS ---
        sint = np.sin(theta_rad)
        cost = np.cos(theta_rad)

        k_z = self.k * cost  # Componente vertical del número de onda
        k_x = self.k * sint  # Componente horizontal del número de onda

        # Convertir rugosidad de cm a metros
        s_m = np.clip(rms_cm, 0.0, np.inf) / 100.0

        # --- 3. CALIBRACIÓN BAGHDADI: Lopt ---
        # Usa ecuaciones DIFERENTES para VV y HV
        Lopt_cm = self.roughness.compute_Lopt(rms_cm, theta_deg, polarization)
        L_m = np.clip(Lopt_cm, 1e-3, np.inf) / 100.0  # Convertir a metros

        # --- 4. MODELO DIELÉCTRICO HALLIKAINEN: ε_r ---
        eps_r = self.dielectric.compute_dielectric(mv, self.frequency)

        # --- 5. COEFICIENTES DE FRESNEL ---
        R_h = self._fresnel_h(eps_r, theta_rad)
        R_v = self._fresnel_v(eps_r, theta_rad)

        # --- 6. TÉRMINOS DE SCATTERING (DEPENDEN DE POLARIZACIÓN) ---
        if polarization == "VV":
            # Co-polarizada VV
            f_term = self._f_vv(eps_r, sint, cost, R_v)
            F_term = self._F_vv(eps_r, sint, cost)

        elif polarization in ("HV", "VH"):
            # Cross-polarizada HV/VH
            f_term = self._f_hv(eps_r, sint, cost, R_h, R_v)
            F_term = self._F_hv(eps_r, sint, cost)

        else:
            raise NotImplementedError(
                f"Polarización {polarization} no soportada. Use 'VV', 'HV', o 'VH'."
            )

        # --- 7. SERIE DE SCATTERING (Fung 1992, Eq. 17) ---

        # Factor de atenuación exponencial
        kz_s = k_z * s_m
        kz_s2 = np.clip(kz_s * kz_s, 0.0, 700.0)  # exp(-2*700) ~ 0 numéricamente
        exp_term = np.exp(-2.0 * kz_s2)

        # Acumulador de la serie
        series_sum = np.zeros_like(R_h, dtype=complex)

        for n in range(1, self.N_TERMS + 1):
            n_fact = factorial(n)

            # Espectro de potencia Gaussiano (Fung Eq. 4-A.3)
            Wn = self.roughness.get_spectrum(2.0 * k_x, L_m, n)

            # ECUACIÓN (18) DE FUNG
            # I_pp^(n) = (2·k_z·s)^n × f + (k_z·s)^(2n)/2 × F
            #
            # NO se usa √n_fact (causa sobre-atenuación ~35 dB)
            # La normalización 1/n! en la serie es suficiente
            p1 = np.power(2.0 * kz_s, n, dtype=np.float64)
            p2 = np.power(kz_s, 2 * n, dtype=np.float64)
            I_pp_n = p1 * f_term + 0.5 * p2 * F_term

            # Acumulación de la serie (Eq. 17)
            # Σ[(W^(n)/n!) × |I_pp^(n)|²]
            series_sum += (Wn / n_fact) * np.abs(I_pp_n) ** 2

        # --- 8. COEFICIENTE FINAL (Fung 1992, Eq. 17) ---
        # σ⁰ = (k²/2) × exp(-2k_z²s²) × Σ[...]
        sigma0_lin = (self.k**2 / 2.0) * exp_term * np.real(series_sum)

        # Protección contra valores no físicos (log de negativos o cero)
        sigma0_lin =  np.where(~np.isfinite(sigma0_lin) | (sigma0_lin <= 1e-20), 1e-20, sigma0_lin)

        # Convertir a dB
        sigma0_dB = 10.0 * np.log10(sigma0_lin)

        return sigma0_dB
