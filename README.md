# IEM-C: Simulación e Inversión de Humedad del Suelo en Banda C mediante Modelado Físico y Aprendizaje Híbrido

## Propósito

Repositorio para simulación e inversión de humedad del suelo en banda C usando:

1. Modelo físico IEM-B calibrado con Hallikainen (dieléctrico) y Baghdadi (Lopt).
2. Validación física y numérica con tests unitarios orientados a papers.
3. Inversión con redes neuronales y modelo híbrido Física+ML con pérdida guiada por física.
4. Propagación de incertidumbres analítica y Monte Carlo.

El objetivo es reproducible: generar datasets sintéticos, entrenar inversores, estimar incertidumbre y validar resultados frente a literatura (Fung 1992; Baghdadi 2011).

---

## Estructura

```
├── analysis
│   ├── ambiguity_mapper.py          # mapas de ambigüedad y regiones mal condicionadas
│   ├── response_surfaces.py         # superficies σ⁰=f(mv, s, θ)
│   ├── sensitivity_analysis.py      # ∂σ⁰/∂mv, ∂σ⁰/∂s, ∂σ⁰/∂θ
├── core
│   ├── constants.py                 # dominios válidos, banderas de calidad, tamaños de figuras, etc.
│   ├── models.py                    # DielectricModel (Hallikainen), SurfaceRoughness (Baghdadi/Fung), IEM_Model
│   ├── utils.py                     # métricas (RMSE, MAE, sesgo), utilitarios numéricos
├── data
│   ├── generate_dataset.py          # generación de dataset sintético σ⁰ con variantes VV/HV
├── inversion
│   ├── hybrid_physics_ml.py         # red guiada por física con pérdida de consistencia forward
│   ├── neural_network.py            # RN simple para inversión σ⁰→mv (y opcional rms)
│   ├── uncertainty_propagation.py   # O2: propagación analítica y Monte Carlo; mapa de calidad sin validación in situ
├── tests
│   ├── baghdadi_test.py             # validación HV vs Baghdadi 2011 (tabla de referencia)
│   ├── fung_test.py                 # validación “white-box” HV/VV vs Fung 1992 (serie y tendencias)
│   ├── limits_test.py               # límites físicos: ks, kL y dependencia angular
│   ├── monotonicity_test.py         # monotonía σ⁰_HV vs mv en escenarios de θ y s
├── validation
│   ├── sentinel1_validation.py      # esqueleto para validación con Sentinel-1 (si se dispone de datos)
├── requirements.txt
├── run_full_pipeline.sh             # atajo para pipeline completo en Linux/macOS
├── setup.sh / setup.bat             # entorno local (venv) en Linux/macOS o Windows
```

---

## Requisitos

* Python ≥ 3.10
* Paquetes ver `requirements.txt`
* CPU con BLAS; GPU opcional para PyTorch
* Espacio en disco para datasets sintéticos (∼100–500 MB según configuración)

Instalación rápida (Linux/macOS):

```bash
chmod +x setup.sh
./setup.sh
# o manual:
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
.\setup.bat
# o manual
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

---

## Flujo de trabajo (pipeline)

### 1) Generación del dataset sintético

```bash
# variables por defecto en data/generate_dataset.py (frecuencia, rejillas de mv/s/θ, etc.)
python -m data.generate_dataset
# salida típica: data/synthetic_dataset_100k.csv
```

### 2) Exploración física básica

```bash
# superficies de respuesta, análisis de sensibilidad
python -m analysis.response_surfaces
python -m analysis.sensitivity_analysis
python -m analysis.ambiguity_mapper
```

### 3) Inversión con redes

**Red simple (supervisada):**

```bash
python -m inversion.neural_network \
  --dataset_path data/synthetic_dataset_100k.csv \
  --epochs 100
# genera trained_model_<escenario>.pth e inversion_comparison.png
```

**Modelo híbrido Física+ML:**

```bash
python -m inversion.hybrid_physics_ml \
  --train_csv data/training_dataset.csv \
  --val_csv data/validation_dataset.csv \
  --epochs 100 --batch_size 256
# guarda inversion/models/best_physics_guided_model.pth
```

> Nota: `hybrid_physics_ml.py` calcula una pérdida de consistencia física llamando a `IEM_Model.compute_backscatter` con las predicciones. Se recomienda lotes moderados si no hay GPU.

### 4) Incertidumbre

```bash
# modo completo: analítica + Monte Carlo + mapas de calidad + análisis angular
python -m inversion.uncertainty_propagation --mode full --n_mc 10000
# salidas: analysis/outputs/quality_map_predicted_theta35.png, uncertainty_vs_angle.png
```

### 5) Validación física y numérica (tests)

Ejecutar cada test desde la raíz:

```bash
python -m tests.fung_test
python -m tests.baghdadi_test
python -m tests.limits_test
python -m tests.monotonicity_test
```

Qué valida cada uno:

* **fung_test.py**: aplica la serie de IEM “white-box” con dieléctrico fijo, compara VV con valores aproximados de Fung Fig.2 y HV como VV−12 dB; verifica monotonicidad con ángulo.
* **baghdadi_test.py**: compara HV con valores tabulados de Baghdadi 2011 (Table 1). Tolerancia típica: |Δσ⁰|≤1 dB.
* **limits_test.py**: barres ks∈{0.3,1.0,3.0}, fijo kL=6, reporta σ⁰_VV(θ) y tendencia física esperada; también σ⁰ vs mv en θ=35° para VV/HV.
* **monotonicity_test.py**: verifica que σ⁰_HV crece con mv para distintos s y θ; reporta rango dinámico y descensos.

---

## Componentes clave

### `core/models.py`

* **DielectricModel (Hallikainen, 1985)**:
  (\varepsilon_r(mv) = (\text{poly}(mv; f, textura))*\text{real} + j,(\text{poly}(mv; f, textura))*\text{imag})
  Entrada: mv (% o fracción), textura (sand, clay), f (Hz).

* **SurfaceRoughness**:
  (L_\text{opt}) calibrado por polarización (Baghdadi 2011) y espectro (W^{(n)}) (Fung 4-A.3 gaussiano).
  `compute_Lopt(rms_cm, θ, pol)` y `get_spectrum(2k_x, L, n)`.

* **IEM_Model (Fung 1992 + Baghdadi 2011 + Hallikainen 1985)**:
  [
  \sigma^0=\frac{k^2}{2}e^{-2(k_z s)^2}\sum_{n=1}^{N}\frac{W^{(n)}(2k_x)}{n!},\left|, (2k_z s)^n f_{pp} + \frac{(k_z s)^{2n}}{2} F_{pp} ,\right|^2
  ]
  Con (f_{vv}, F_{vv}) y (f_{hv}=0, F_{hv}) (cross-pol multiescattering).
  Retorna σ⁰ en dB.

### `inversion/uncertainty_propagation.py`

* **Analítico**: linealiza con Jacobiano numérico, invierte derivadas para propagar varianzas.
* **Monte Carlo**: muestrea ruido en σ⁰, s y θ; invierte por optimización o LUT; reporta media, std, IC, RMSE, sesgo; tasa de éxito.
* **Mapa de calidad (O2)**: predice incertidumbre esperada por pixel sin validación in situ usando sólo sensibilidad.

### `inversion/neural_network.py`

* RN feed-forward para inversión σ⁰→mv (y variantes con `rms` o clases).
* Escenarios: sin prior, rms conocido, rango de mv como clase auxiliar.
* Guarda modelos y compara RMSE/MAE/Bias entre escenarios.

### `inversion/hybrid_physics_ml.py`

* Physics-Guided NN: salida [mv, rms], fusión α·(física) + β·(ML).
* Pérdida con término de consistencia física “forward”: (\sigma^0(\hat{mv},\hat{rms}) \approx \sigma^0_\text{obs}) usando `IEM_Model`.
* Penalización de rangos válidos para mv y rms.

---

## Uso básico (receta mínima)

1. Crear entorno y dependencias.
2. `python -m data.generate_dataset` para CSV sintético.
3. `python -m inversion.neural_network --dataset_path data/synthetic_dataset_100k.csv`.
4. `python -m inversion.uncertainty_propagation --mode full`.
5. `python -m tests.fung_test` y `python -m tests.baghdadi_test` para validaciones.
6. Revisar figuras en `analysis/outputs/` y archivos `.png` generados por tests.

---

## Consideraciones de dominio

* Banda C: `frequency=5.405e9` (Sentinel-1).
* Validez IEM: controlar (k s < 3) y (kL) en rango moderado; fuera de dominio la serie puede desestabilizarse o producir sesgos.
* Polarización: HV nace de multiescattering; típicamente σ⁰_HV es 8–15 dB menor que σ⁰_VV en suelo desnudo.
* Textura: Hallikainen depende de arena/arcilla; mantener porcentajes físicos (0–100).

---

## Reproducibilidad

* Fijar semilla en generadores (NumPy, PyTorch) según necesidad.
* Registrar configuraciones de dataset y arquitectura en nombres de salida o en un JSON adjunto.
* Versionar `requirements.txt` y documentar GPU/CPU.

---

## Troubleshooting

**Síntoma:** `ImportError: cannot import name 'IEM_Model'`
**Causa:** diferencias de layout de paquete.
**Solución:** usar import con fallback (`from core.models import IEM_Model` y si falla `from models import IEM_Model`). Los tests ya incluyen esta lógica en los parches propuestos.

**Síntoma:** `NaN` o `inf` en σ⁰ o pérdida física.
**Causa:** violación de dominio, desbordes numéricos en serie o raíz compleja.
**Solución:**

* Verificar (k s) y (k L).
* Clampear σ⁰ lineal mínima (`1e-20`) antes de pasar a dB.
* Reducir `N_TERMS` o revisar `rms_cm` extremos.

**Síntoma:** Curvas HV no monotónicas vs mv en pequeños tramos.
**Causa:** ruido numérico de derivadas o sensibilidad angular.
**Solución:** tolerancia de 0.05 dB por paso en `monotonicity_test.py`; revisar `θ` y paso de mv.

**Síntoma:** RMSE > 1 dB en `baghdadi_test.py`.
**Causa:** discrepancia en espectro y/o términos HV.
**Solución:** asegurar espectro gaussiano en pruebas físicas, revisar implementación de (F_{hv}), verificar Lopt(HV) y θ.

**Síntoma:** Entrenamiento híbrido lento.
**Causa:** pérdida física llama a `compute_backscatter` por batch en CPU.
**Solución:** reducir `batch_size`, cachear partes invariantes por batch, o vectorizar llamada.

**Síntoma:** CUDA no disponible.
**Solución:** usar CPU, bajar `batch_size`, revisar instalación de PyTorch acorde a CUDA local.

---

## Implementaciones faltantes / futuras

1. **Guardas de dominio en `IEM_Model`**: avisos explícitos cuando (k s\ge 3) o (kL) fuera de rango; clamp y “quality flags”.
2. **Selección de espectro por defecto**: usar Gaussiano por defecto para validación con Fung/Baghdadi; exponer parámetro público para alternar “gaussian”/“fractal”.
3. **Términos (f_{vv}) y (F_{hv}) exactos**: reemplazar aproximaciones por formulaciones más fieles o LUT/ajustes con referencias. Parámetro `use_simplified_terms`.
4. **Vectorización de la pérdida física** en `hybrid_physics_ml.py`: eliminar bucles Python internos y mover cómputo a tensores cuando sea viable.
5. **Errores correlacionados en incertidumbre**: extender propagación analítica con covarianzas completas y Monte Carlo con correlación σ⁰–θ–s.
6. **Convergencia de serie**: criterio adaptativo por ángulo y rugosidad, salida de diagnóstico N vs N−1.
7. **Validación con Sentinel-1**: completar `validation/sentinel1_validation.py` con lectura de GRD/SLC, geocodificación, máscaras y comparación con in-situ si hay datos.
8. **CLI unificado**: `python -m cli ...` para orquestar dataset, entrenamiento, incertidumbre y tests.
9. **Documentación API**: docstrings completos y ejemplos mínimos por función clave.
10. **Integración continua**: workflow de CI para ejecutar `tests/*.py` y generar artefactos.

---

## Licencia y citación

* Citar fuentes primarias al reportar resultados: Hallikainen (1985), Fung (1992), Baghdadi (2011).
* Indicar versión del repositorio y commit al usar figuras o métricas.

---

## Contacto

Para contribuciones, abrir *issue* o *pull request* con:

* Descripción clara del cambio
* Impacto en reproducibilidad y validaciones
* Comparación antes/después (curvas o métricas)
