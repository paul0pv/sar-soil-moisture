`README.md`

# Implementación del Modelo Físico IEM-B para Simulación SAR

Este documento describe la arquitectura y la lógica de implementación del archivo `models.py`, un motor de simulación física diseñado para generar datos de retrodispersión de radar (SAR).

## 1. Objetivo del Proyecto (Metas)

El objetivo principal de este código es servir como el "motor" para generar un **conjunto de datos sintético a gran escala**.

Este conjunto de datos es la piedra angular de un estudio de aprendizaje automático (Machine Learning) cuyo fin es **invertir la señal de radar de la Banda C (ej. Sentinel-1)** para estimar la **humedad volumétrica del suelo ($M_v$)**, corrigiendo al mismo tiempo el efecto de la **rugosidad superficial ($H_{rms}$)**.

## 2. Arquitectura de Modelos Integrados (Integración)

La implementación está contenida en `models.py` y sigue una cadena de física modular. El cálculo de la retrodispersión ($\sigma^0$) se basa en tres modelos científicos fundamentales que se integran de la siguiente manera:

1.  **Entrada del Usuario:** El modelo principal (`IEM_Model.compute_backscatter`) recibe los parámetros físicos:
    * Humedad del Suelo (`mv` en %)
    * Rugosidad RMS (`rms_cm` en cm)
    * Ángulo de Incidencia (`theta_deg` en grados)

2.  **Paso 1: Modelo Dieléctrico (Hallikainen 1985):** La `mv` (junto con la textura del suelo definida en `__init__`) se pasa al `DielectricModel` para calcular la constante dieléctrica compleja del suelo ($\epsilon_r$).

3.  **Paso 2: Calibración de Rugosidad (Baghdadi 2011):** Los `rms_cm` y `theta_deg` se pasan al `SurfaceRoughness.compute_Lopt` para calcular la longitud de correlación óptima (`Lopt`), que es la calibración clave del "IEM-B".

4.  **Paso 3: Modelo de Dispersión (Fung 1992):** Los valores $\epsilon_r$ (del Paso 1) y `Lopt` (del Paso 2), junto con `rms_cm` y `theta_deg`, se introducen en las ecuaciones físicas del Modelo de Ecuación Integral (IEM) para calcular el coeficiente de retrodispersión final, $\sigma^0$ (en dB).

## 3. Desglose de Componentes y Fuentes (Documentos Base)

La implementación debe ser una transcripción precisa de las siguientes ecuaciones de la literatura:

### 3.1. `DielectricModel`

* **Fuente:** Hallikainen et al. (1985), "Microwave Dielectric Behavior of Wet Soil-Part 1", IEEE TGRS.
* **Lógica de Implementación:**
    * Se implementan las **Ecuaciones (1) y (2)**, que son regresiones polinómicas (en frecuencia) basadas en las **Tablas IV y V** para encontrar los coeficientes texturales (`A, B, C` y `D, E, F`).
    * Estos coeficientes se utilizan en las **Ecuaciones (3) y (4)** para calcular la parte real ($\epsilon'$) y la parte imaginaria ($\epsilon''$) de la constante dieléctrica a partir de la humedad `mv`.
    * La función `compute_dielectric` debe estar vectorizada para `mv`.

### 3.2. `SurfaceRoughness`

Esta clase implementa dos funciones de dos artículos diferentes:

1.  **`compute_Lopt`**
    * **Fuente:** Baghdadi et al. (2011), "Semiempirical Calibration of the IEM... in C-Band".
    * **Lógica de Implementación:** Implementa la **Ecuación (3)** para `Lopt` en polarización VV. Esta calibración es lo que define al modelo como "IEM-B" y relaciona `Lopt` con `H_{rms}` y `theta`.

2.  **`get_spectrum`**
    * **Fuente:** Fung et al. (1992), "Backscattering from a Randomly Rough Dielectric Surface".
    * **Lógica de Implementación:** Implementa el espectro de potencia Gaussiano, **Ecuación (4-A.3)**.
    * La fórmula es: $W^{(n)}(k_x) = \frac{L_m^2}{2n} \exp\left(-\frac{k_x^2 L_m^2}{4n}\right)$.
    * **Punto Crítico:** El factor de `/ (4 * n)` en el exponente es fundamental y una fuente común de error.

### 3.3. `IEM_Model`

Esta es la clase principal que implementa la física del IEM.

* **Fuente:** Fung et al. (1992), "Backscattering from a Randomly Rough Dielectric Surface".
* **Lógica de Implementación de `compute_backscatter`:**
    1.  **Términos de Fresnel:** Se calculan los términos $R_{vv}$ (Fresnel), $f_{vv}$ (**Eq. 22**) y $F_{vv}$ (**Eq. 24**). Estos términos contienen toda la dependencia dieléctrica ($\epsilon_r$).
    2.  **Cálculo de la Serie (Eq. 17 y 18):** El modelo calcula el término complementario (incoherente) $\sigma^c_{pp}$ (Eq. 17), ya que el término de Kirchhoff (coherente) se omite para superficies agrícolas.
    3.  La **Ecuación (17)** es la suma principal:
        $\sigma^c_{pp} = \frac{k^2}{2} e^{-2k_z^2s^2} \sum_{n=1}^{\infty} \frac{s_m^{2n}}{n!} |I_{pp}^{(n)}|^2 W^{(n)}$
    4.  **Lógica del Bucle (Implementación Correcta):** El código debe iterar de `n=1` a `N_TERMS` y, en cada paso:
        * Calcular el término de campo $I_{vv}^{(n)}$ usando la **Ecuación (18)**. Esta ecuación **NO** depende de $s_m$:
            $I_{vv}^{(n)} = (2k_z)^n f_{vv} + \frac{(k_z)^{2n}}{2} F_{vv}$
        * Calcular el espectro $W^{(n)}$ (usando `get_spectrum`).
        * Combinar los términos de la suma de Eq. (17):
            `term_n = ( (s_m**(2*n)) / n_fact ) * np.abs(I_vv_n)**2 * Wn`
    5.  **Resultado Final:** La suma de `term_n` se multiplica por los pre-factores de Eq. (17):
        `sigma0_C = (self.k**2 / 2) * exp_term * series_sum`
    6.  **Punto Crítico:** El pre-factor es $\frac{k^2}{2}$ (Ecuación 17). Errores comunes (como $\frac{k^2}{4\pi}$) introducen un sesgo de ~8 dB.
    7.  Finalmente, se convierte a decibelios (dB), manejando `log(0)`.

## 4. Proceso de Validación y Estudio Previo

La implementación de este modelo es notoriamente difícil. Un "estudio previo" (nuestro proceso de depuración) reveló que los errores de transcripción de las fórmulas (ej. omitir el `/4` en `Wn` o interpretar mal la Eq. 17) son comunes y catastróficos.

Por lo tanto, este `models.py` ha sido diseñado para pasar rigurosamente 4 scripts de prueba:

1.  **`validation_test_1_fung.py`:** Valida la física pura del *scattering* (Eq. 17 y 18) contra la Figura 2 del paper de Fung (1992).
2.  **`validation_test_2_baghdadi.py`:** Valida la **cadena de integración completa** (Hallikainen + Lopt + IEM) contra los valores publicados en la Tabla 1 de Baghdadi (2011). Es la prueba de "éxito" del modelo calibrado (error < 0.5 dB).
3.  **`validation_test_3_limits.py`:** Un "sanity check" que asegura que el suelo seco (`mv` -> 0) produce una señal físicamente baja (ej. < -25 dB).
4.  **`validation_test_4_monotonicity.py`:** La prueba más crítica. Verifica que la señal $\sigma^0$ **aumenta monotónicamente** con la humedad `mv`. Los fallos en esta prueba (curvas planas o colapsadas) fueron el principal indicador de errores en la implementación de la Eq. (17) y (18).

**Implicación:** Cualquier modificación futura a `models.py` debe volver a pasar este conjunto de pruebas para ser considerada válida.

## 5. Contexto e Implicaciones de la Investigación

La recuperación de la humedad del suelo ($M_v$) a partir de datos SAR es un "problema mal planteado" (ill-posed) porque la señal $\sigma^0$ es una mezcla de $M_v$ y $H_{rms}$.

Este código es la solución a ese problema. Al implementar un modelo físico validado, podemos:

1.  **Generar un Dataset Controlado:** Ejecutar `models.py` $10^5$ o $10^6$ veces para crear una tabla masiva de `[mv, hrms, theta, sigma0_vv]`.
2.  **Entrenar un Inversor de ML:** Usar esta tabla para entrenar una red neuronal que aprenda la función de inversión no lineal: `f(sigma0_vv, theta, ...) -> mv`.
3.  **Habilitar la Corrección de Rugosidad:** Dado que el modelo fue entrenado con datos que cubren todo el espectro de rugosidad (gracias a `Lopt` y `Hrms` en las ecuaciones), la red neuronal aprende implícitamente a corregir el efecto de la rugosidad.

En resumen, la **precisión física de este código es la base del éxito de todo el proyecto de Machine Learning**. Si este modelo es correcto, el conjunto de datos sintético será una representación precisa de la física de la Banda C, permitiendo al modelo de ML operar con éxito en datos satelitales reales de Sentinel-1.
