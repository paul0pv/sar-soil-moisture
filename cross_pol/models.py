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
    Tablas IV y V, páginas 26-27
    """

    def __init__(self, sand_pct, clay_pct):
        """
        Inicializa el modelo dieléctrico con la textura del suelo.

        Args:
            sand_pct: Porcentaje de arena (0-100)
            clay_pct: Porcentaje de arcilla (0-100)
        """
        self.sand = sand_pct
        self.clay = clay_pct
        self.silt = 100 - sand_pct - clay_pct

        # Coeficientes de regresión para Epsilon Prima (Parte Real) - Tabla IV
        # Estructura: P = alpha + beta*f + gamma*f^2 (f en GHz)
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

    def _get_poly_coeff(self, coeff_dict, f_ghz):
        """
        Calcula un coeficiente (a0, b0, etc.) usando el polinomio cuadrático.

        Fórmula: coeff = alpha + beta*f + gamma*f²

        Args:
            coeff_dict: Diccionario con alpha, beta, gamma
            f_ghz: Frecuencia en GHz

        Returns:
            Valor del coeficiente
        """
        return (
            coeff_dict["alpha"]
            + coeff_dict["beta"] * f_ghz
            + coeff_dict["gamma"] * (f_ghz**2)
        )

    def compute_dielectric(self, mv, frequency=5.405e9):
        """
        Calcula la constante dieléctrica compleja del suelo.

        Implementa Hallikainen (1985), Ecuaciones (1)-(4):

        ε' = A + B·mv + C·mv²
        ε'' = D + E·mv + F·mv²

        Donde:
        A = a₀ + a₁·S + a₂·C
        B = b₀ + b₁·S + b₂·C
        C = c₀ + c₁·S + c₂·C
        (similar para D, E, F)

        Args:
            mv: Humedad volumétrica (%) - escalar o array
            frequency: Frecuencia en Hz (default: 5.405 GHz)

        Returns:
            Constante dieléctrica compleja ε_r = ε' - jε''
        """
        # Convertir frecuencia a GHz
        f_ghz = frequency / 1e9

        # Asegurar que mv es array y normalizar a fracción
        mv = np.asarray(mv)
        mv_frac = np.where(mv > 1.0, mv / 100.0, mv)

        S = self.sand
        C = self.clay

        # --- PARTE REAL (Epsilon Prima) ---
        # Calcular coeficientes dependientes de frecuencia (Tabla IV)
        a0 = self._get_poly_coeff(self.coeffs_real["a0"], f_ghz)
        a1 = self._get_poly_coeff(self.coeffs_real["a1"], f_ghz)
        a2 = self._get_poly_coeff(self.coeffs_real["a2"], f_ghz)
        b0 = self._get_poly_coeff(self.coeffs_real["b0"], f_ghz)
        b1 = self._get_poly_coeff(self.coeffs_real["b1"], f_ghz)
        b2 = self._get_poly_coeff(self.coeffs_real["b2"], f_ghz)
        c0 = self._get_poly_coeff(self.coeffs_real["c0"], f_ghz)
        c1 = self._get_poly_coeff(self.coeffs_real["c1"], f_ghz)
        c2 = self._get_poly_coeff(self.coeffs_real["c2"], f_ghz)

        # Coeficientes de la ecuación cuadrática
        A = a0 + a1 * S + a2 * C
        B = b0 + b1 * S + b2 * C
        C_coeff = c0 + c1 * S + c2 * C

        # Ecuación (1): ε' = A + B·mv + C·mv²
        epsilon_prime = A + B * mv_frac + C_coeff * (mv_frac**2)

        # --- PARTE IMAGINARIA (Epsilon Doble Prima) ---
        # Calcular coeficientes dependientes de frecuencia (Tabla V)
        d0 = self._get_poly_coeff(self.coeffs_imag["d0"], f_ghz)
        d1 = self._get_poly_coeff(self.coeffs_imag["d1"], f_ghz)
        d2 = self._get_poly_coeff(self.coeffs_imag["d2"], f_ghz)
        e0 = self._get_poly_coeff(self.coeffs_imag["e0"], f_ghz)
        e1 = self._get_poly_coeff(self.coeffs_imag["e1"], f_ghz)
        e2 = self._get_poly_coeff(self.coeffs_imag["e2"], f_ghz)
        f0 = self._get_poly_coeff(self.coeffs_imag["f0"], f_ghz)
        f1 = self._get_poly_coeff(self.coeffs_imag["f1"], f_ghz)
        f2 = self._get_poly_coeff(self.coeffs_imag["f2"], f_ghz)

        D = d0 + d1 * S + d2 * C
        E = e0 + e1 * S + e2 * C
        F = f0 + f1 * S + f2 * C

        # Ecuación (2): ε'' = D + E·mv + F·mv²
        epsilon_double_prime = D + E * mv_frac + F * (mv_frac**2)

        # Constante dieléctrica compleja
        return epsilon_prime + 1j * epsilon_double_prime


# -------------------------------------------------------------------
# CLASE 2: MODELO DE RUGOSIDAD (Baghdadi 2011 + Fung 1992)
# -------------------------------------------------------------------


class SurfaceRoughness:
    """
    Parametrización de rugosidad con calibración IEM_B (Baghdadi 2011)
    y espectro de potencia Gaussiano (Fung 1992).

    Referencias:
    - Baghdadi et al., IEEE GRSL, Vol. 8, No. 1, 2011, Eq. (3)
    - Fung et al., IEEE TGRS, Vol. 30, No. 2, 1992, Eq. (4-A.3)
    """

    def __init__(self, correlation="gaussian"):
        """
        Inicializa el modelo de rugosidad.

        Args:
            correlation: Tipo de función de autocorrelación ("gaussian")
        """
        self.correlation_type = correlation

    def compute_Lopt(self, rms_cm, theta_deg, polarization="HV"):
        """
        Calcula la longitud de correlación óptima calibrada.

        Implementa Baghdadi et al. (2011), Ecuación (3):

        Para HV: Lopt = 1.281 + 0.134 × [sin(0.19θ)]^(-1.59) × s

        Nota: El paper de Baghdadi 2011 presenta calibraciones separadas
        para HH, VV y HV. La ecuación es la misma para todas, pero los
        coeficientes pueden variar ligeramente.

        Args:
            rms_cm: Altura RMS de rugosidad en cm (escalar o array)
            theta_deg: Ángulo de incidencia en grados (escalar o array)
            polarization: Polarización ("HV")

        Returns:
            Lopt en cm (mismo shape que los inputs)
        """
        if polarization not in ["HV", "VH", "VV"]:
            raise NotImplementedError(f"Polarización {polarization} no implementada.")

        theta_rad = np.deg2rad(theta_deg)

        if polarization in ["HV", "VH"]:
            # POLARIZACIÓN CRUZADA (HV/VH)
            # Baghdadi (2011), Ecuación (2) - Página 3, Sección V
            # Lopt2(s,θ) = 0.9157 + 1.2289 * [sin(0.1543*θ)]^(-0.3139) * s

            sin_term = np.sin(0.1543 * theta_rad)

            # Evitar división por cero
            sin_term = np.where(np.abs(sin_term) < 1e-6, 1e-6, sin_term)

            # Ecuación (2) - Lopt2 para HV (valor alto, Gaussiana)
            Lopt = 0.9157 + 1.2289 * (sin_term ** (-0.3139)) * rms_cm

        else:  # polarization == "VV" (o "HH")
            # POLARIZACIÓN CO-POLARIZADA (VV/HH)
            # Baghdadi (2011), Ecuación (3) - Original
            # Lopt(s,θ) = 1.281 + 0.134 * [sin(0.19*θ)]^(-1.59) * s

            sin_term = np.sin(0.19 * theta_rad)
            sin_term = np.where(np.abs(sin_term) < 1e-6, 1e-6, sin_term)

            Lopt = 1.281 + 0.134 * (sin_term ** (-1.59)) * rms_cm

        return Lopt

    def get_spectrum(self, k_x, L_m, n):
        """
        Espectro de potencia para función de autocorrelación Gaussiana.

        Implementa Fung (1992), Apéndice 4-A, Ecuación (4-A.3):

        W^(n)(k_x) = (L²/2n) × exp(-k_x²L²/4n)

        NOTA CRÍTICA: El factor 4 en el exponente es esencial.
        Sin él, el espectro decae demasiado lento y causa inestabilidad.

        Args:
            k_x: Número de onda horizontal = k*sin(θ) (escalar o array)
            L_m: Longitud de correlación en metros (escalar o array)
            n: Orden del término en la serie de scattering

        Returns:
            W^(n): Espectro de potencia (mismo shape que inputs)

        Referencias:
        - Fung (1992), página 367, Ecuación (4-A.3)
        """
        L_m2 = L_m**2
        k_x2 = k_x**2

        # Ecuación (4-A.3) con factor 4 correcto
        Wn = (L_m2 / (2 * n)) * np.exp(-k_x2 * L_m2 / (4 * n))

        return Wn


# -------------------------------------------------------------------
# CLASE 3: MODELO IEM POLARIZACIÓN CRUZADA (Fung 1992 + Baghdadi 2011)
# -------------------------------------------------------------------


class IEM_Model:
    """
    Modelo de Ecuación Integral calibrado (IEM_B) para polarización cruzada HV.

    Implementa:
    - Física de scattering: Fung et al. (1992), IEEE TGRS
    - Calibración de Lopt: Baghdadi et al. (2011), IEEE GRSL
    - Modelo dieléctrico: Hallikainen et al. (1985), IEEE TGRS

    POLARIZACIÓN CRUZADA (HV/VH):
    ================================
    A diferencia de las polarizaciones co-polarizadas (HH, VV), la
    polarización cruzada NO tiene término de Kirchhoff coherente.
    Solo tiene scattering múltiple (término complementario).

    Ecuaciones principales (Fung 1992, páginas 356-357):

    1. NO HAY término de Kirchhoff para HV:
       σ⁰_K^HV = 0  [ver Fung, discusión después de Eq. 16]

    2. Solo término complementario (Eq. 17):
       σ⁰_C^HV = (k²/2) × exp(-2k_z²s²) × Σ[(W^(n)/n!) × |I_hv^(n)|²]

    3. Términos de scattering cruzado (Eq. 18 adaptada para HV):
       I_hv^(n) = [(2k_z·s)^n / √n!] × f_hv + [(k_z·s)^(2n) / (2√n!)] × F_hv

    Donde (de Fung, Eq. después de (24)):

    4. f_hv y F_hv contienen términos mixtos de reflexión H→V:
       f_hv ≈ 0 para backscattering (simetría)
       F_hv depende de términos de segundo orden
    """

    def __init__(self, frequency=5.405e9, sand_pct=20, clay_pct=30):
        """
        Inicializa el modelo IEM calibrado para polarización cruzada.

        Args:
            frequency: Frecuencia en Hz (default: 5.405 GHz - banda C)
            sand_pct: Porcentaje de arena (0-100)
            clay_pct: Porcentaje de arcilla (0-100)
        """
        self.frequency = frequency
        self.wavelength = 2.99792e8 / frequency  # λ = c/f
        self.k = 2 * np.pi / self.wavelength  # Número de onda

        # Submodelos
        self.dielectric = DielectricModel(sand_pct, clay_pct)
        self.roughness = SurfaceRoughness(correlation="gaussian")

        # Número de términos en la serie de scattering
        self.N_TERMS = 10

    def _fresnel_h(self, eps_r, theta_rad):
        """
        Coeficiente de Fresnel para polarización horizontal (H).

        Fung (1992), página 357, ecuación después de (2):

        R_h = (cos(θ) - √(ε_r - sin²(θ))) / (cos(θ) + √(ε_r - sin²(θ)))

        Args:
            eps_r: Constante dieléctrica compleja
            theta_rad: Ángulo de incidencia en radianes

        Returns:
            R_h: Coeficiente de Fresnel horizontal
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
            R_v: Coeficiente de Fresnel vertical
        """
        cost = np.cos(theta_rad)
        sint2 = np.sin(theta_rad) ** 2

        sqrt_term = np.lib.scimath.sqrt(eps_r - sint2)

        R_v = (eps_r * cost - sqrt_term) / (eps_r * cost + sqrt_term)

        return R_v

    def _f_hv(self, eps_r, sint, cost, R_h, R_v):
        """
        Término f_hv para polarización cruzada.

        Fung (1992), página 368, Ecuaciones (54)-(57) simplificadas:

        Para backscattering, f_hv involucra productos cruzados de R_h y R_v.
        En la aproximación de primer orden y para geometría de backscatter:

        f_hv ≈ (R_h - R_v) × [término geométrico]

        Para backscattering simple, este término es pequeño o cero por simetría.

        Args:
            eps_r: Constante dieléctrica compleja
            sint: sin(θ)
            cost: cos(θ)
            R_h: Coeficiente de Fresnel horizontal
            R_v: Coeficiente de Fresnel vertical

        Returns:
            f_hv: Término de scattering cruzado (≈ 0 en backscatter)
        """
        # En backscattering puro (θ_i = θ_s), f_hv = 0 por simetría
        # Fung (1992) página 357: "cross-polarized coefficients satisfy reciprocity"
        return np.zeros_like(R_h)

    def _F_hv(self, eps_r, sint, cost):
        """
        Término F_hv para polarización cruzada de segundo orden.

        Fung (1992), página 368, ecuaciones (54)-(57):

        F_hv contiene términos de scattering múltiple que SÍ contribuyen
        significativamente en polarización cruzada.

        Aproximación basada en la diferencia entre coeficientes H y V:

        F_hv ≈ [(ε_r - 1) × sin(θ) × cos(θ)] / [denominador complejo]

        Este término captura la despolarización causada por la rugosidad.

        Args:
            eps_r: Constante dieléctrica compleja
            sint: sin(θ)
            cost: cos(θ)

        Returns:
            F_hv: Término de scattering cruzado de orden superior
        """
        sint2 = sint**2
        sqrt_term = np.lib.scimath.sqrt(eps_r - sint2)

        # Término de despolarización (Fung 1992, adaptado de Eq. 24)
        # Para HV, necesitamos el término cruzado que surge de la curvatura
        num = (eps_r - 1) * sint * cost * (1 + sint2)
        den = (eps_r * cost + sqrt_term) ** 2

        # Evitar división por cero
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)

        return num / den

    def compute_backscatter(self, mv, rms_cm, theta_deg, polarization="HV"):
        """
        Calcula σ⁰ en dB para polarización cruzada usando IEM_B.

        ESTRUCTURA DE LA IMPLEMENTACIÓN:
        =================================

        1. [Hallikainen 1985]: mv + textura → ε_r
        2. [Baghdadi 2011]: rms + θ → Lopt
        3. [Fung 1992 Eq. 17]: ε_r + rugosidad → σ⁰

        ECUACIÓN PRINCIPAL (Fung 1992, Eq. 17 adaptada para HV):

        σ⁰_HV = (k²/2) × exp(-2k_z²s²) × Σ[(W^(n)/n!) × |I_hv^(n)|²]

        Donde:
        - NO hay término de Kirchhoff (σ⁰_K = 0 para cross-pol)
        - I_hv^(n) = [(2k_z·s)^n/√n!]·f_hv + [(k_z·s)^(2n)/(2√n!)]·F_hv
        - W^(n) = espectro de potencia Gaussiano
        - n = 1, 2, ..., N_TERMS

        Args:
            mv: Humedad volumétrica (%) - escalar o array
            rms_cm: Altura RMS en cm - escalar o array
            theta_deg: Ángulo de incidencia en grados - escalar o array
            polarization: "HV" o "VH" (equivalentes por reciprocidad)

        Returns:
            σ⁰ en dB (mismo shape que los inputs)

        Referencias:
        - Fung et al., IEEE TGRS, Vol. 30, No. 2, March 1992
        - Baghdadi et al., IEEE GRSL, Vol. 8, No. 1, January 2011
        - Hallikainen et al., IEEE TGRS, Vol. GE-23, No. 1, January 1985
        """
        if polarization not in ["HV", "VH"]:
            raise NotImplementedError(
                f"Polarización {polarization} no implementada. "
                "Esta clase solo soporta polarización cruzada HV/VH."
            )

        # --- 1. VECTORIZACIÓN ---
        mv = np.asarray(mv)
        rms_cm = np.asarray(rms_cm)
        theta_deg = np.asarray(theta_deg)

        # --- 2. PARÁMETROS GEOMÉTRICOS ---
        theta_rad = np.deg2rad(theta_deg)
        sint = np.sin(theta_rad)
        cost = np.cos(theta_rad)

        k_z = self.k * cost  # Componente vertical
        k_x = self.k * sint  # Componente horizontal

        s_m = rms_cm / 100.0  # Convertir cm a metros

        # --- 3. CALIBRACIÓN BAGHDADI: Lopt ---
        Lopt_cm = self.roughness.compute_Lopt(rms_cm, theta_deg, polarization)
        L_m = Lopt_cm / 100.0  # Convertir cm a metros

        # --- 4. MODELO DIELÉCTRICO HALLIKAINEN: ε_r ---
        eps_r = self.dielectric.compute_dielectric(mv, self.frequency)

        # --- 5. COEFICIENTES DE FRESNEL ---
        R_h = self._fresnel_h(eps_r, theta_rad)
        R_v = self._fresnel_v(eps_r, theta_rad)

        # --- 6. TÉRMINOS DE SCATTERING CRUZADO (Fung Eq. adaptada) ---
        f_hv = self._f_hv(eps_r, sint, cost, R_h, R_v)
        F_hv = self._F_hv(eps_r, sint, cost)

        # --- 7. SERIE DE SCATTERING (Fung Eq. 17) ---

        # NO HAY término de Kirchhoff para polarización cruzada
        # Fung (1992) página 357: "cross-polarized scattering contains
        # only multiple scattering terms"

        # Término complementario (incoherente)
        k_z2_s2 = (k_z * s_m) ** 2
        exp_term = np.exp(-2 * k_z2_s2)

        series_sum = np.zeros_like(R_h, dtype=complex)

        for n in range(1, self.N_TERMS + 1):
            n_fact = factorial(n)
            sqrt_n_fact = np.sqrt(n_fact)

            # Espectro de potencia Gaussiano (Fung Eq. 4-A.3)
            Wn = self.roughness.get_spectrum(2 * k_x, L_m, n)

            # Términos de la serie I_hv^(n) (Fung Eq. 18 adaptada para HV)
            term1 = ((2 * k_z * s_m) ** n / sqrt_n_fact) * f_hv
            term2 = ((k_z * s_m) ** (2 * n) / (2 * sqrt_n_fact)) * F_hv

            I_hv_n = term1 + term2

            # Acumulación de la serie (Fung Eq. 17)
            series_sum += (Wn / n_fact) * np.abs(I_hv_n) ** 2

        # Coeficiente de scattering total (solo complementario para cross-pol)
        sigma0_C = (self.k**2 / 2) * exp_term * series_sum

        # --- 8. RESULTADO FINAL ---
        sigma0_linear = np.real(sigma0_C)

        # Protección contra valores no físicos
        sigma0_linear = np.where(sigma0_linear <= 1e-10, 1e-10, sigma0_linear)

        # Convertir a dB
        sigma0_dB = 10 * np.log10(sigma0_linear)

        return sigma0_dB
