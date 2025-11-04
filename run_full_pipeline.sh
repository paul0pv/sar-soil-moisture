#!/bin/bash

set -e

echo "======================================================================"
echo "PIPELINE COMPLETO DE ANÁLISIS IEM-B"
echo "======================================================================"
echo ""

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_step() {
  echo -e "${GREEN}[PASO $1]${NC} $2"
}

log_warning() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

log_step "0" "Verificando dependencias..."
python -c "import numpy, scipy, matplotlib, pandas, torch" 2>/dev/null
if [ $? -eq 0 ]; then
  echo "  ✓ Todas las dependencias instaladas"
else
  log_error "Faltan dependencias. Ejecute: pip install -r requirements.txt"
  exit 1
fi

mkdir -p data
mkdir -p analysis/outputs
mkdir -p inversion/models

log_step "1" "Generando dataset sintético (100k training + 20k validation)..."
python data/generate_dataset.py --mode both --n_train 100000 --n_val 20000 --output_dir data

if [ ! -f "data/training_dataset.csv" ]; then
  log_error "Falló generación de dataset"
  exit 1
fi
echo "  ✓ Dataset generado"

log_step "2" "Ejecutando análisis de sensibilidad..."
python analysis/sensitivity_analysis.py --output_dir analysis/outputs --theta_values 25 30 35 40 45

if [ ! -f "analysis/outputs/invertibility_report.txt" ]; then
  log_warning "No se generó reporte de invertibilidad"
fi
echo "  ✓ Análisis de sensibilidad completado"

log_step "3" "Mapeando regiones de ambigüedad..."
python analysis/ambiguity_mapper.py --output_dir analysis/outputs --theta_values 30 35 40 45

if [ ! -f "analysis/outputs/problematic_zones.txt" ]; then
  log_warning "No se generó reporte de zonas problemáticas"
fi
echo "  ✓ Mapeo de ambigüedad completado"

log_step "4" "Analizando propagación de incertidumbres..."
python inversion/uncertainty_propagation.py --mode full --n_mc 5000 --output_dir analysis/outputs

if [ ! -f "analysis/outputs/quality_map_predicted_theta35.png" ]; then
  log_warning "No se generó mapa de calidad"
fi
echo "  ✓ Análisis de incertidumbres completado"

read -p "¿Entrenar modelo híbrido física-ML? (requiere ~1h en CPU) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  log_step "5" "Entrenando modelo híbrido física-ML..."
  python inversion/hybrid_physics_ml.py \
    --train_csv data/training_dataset.csv \
    --val_csv data/validation_dataset.csv \
    --epochs 100 \
    --batch_size 256 \
    --output_dir inversion/models

  if [ ! -f "inversion/models/best_physics_guided_model.pth" ]; then
    log_warning "No se guardó modelo entrenado"
  fi
  echo "  ✓ Modelo híbrido entrenado"
else
  log_warning "Entrenamiento de modelo híbrido omitido"
fi

echo ""
echo "======================================================================"
echo "PIPELINE COMPLETADO"
echo "======================================================================"
echo ""
echo "Resultados generados:"
echo "  Datasets:        data/training_dataset.csv, data/validation_dataset.csv"
echo "  Análisis:        analysis/outputs/*.png + *.txt"
echo "  Modelos:         inversion/models/*.pth (si se entrenó)"
echo ""
echo "Siguiente pasos sugeridos:"
echo "  1. Revisar analysis/outputs/invertibility_report.txt"
echo "  2. Revisar analysis/outputs/problematic_zones.txt"
echo "  3. Inspeccionar figuras en analysis/outputs/"
echo "  4. (Opcional) Validar con datos Sentinel-1 reales"
echo ""
echo "Pipeline ejecutado exitosamente"
