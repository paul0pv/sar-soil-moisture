#!/bin/bash

echo "========================================="
echo "Configuración del entorno para"
echo "Tesis: Estimación de Humedad con SAR"
echo "========================================="

# Verificar Python
if ! command -v python3 &>/dev/null; then
  echo "ERROR: Python 3 no está instalado"
  exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Python detectado: $PYTHON_VERSION"

# Crear entorno virtual
echo "Creando entorno virtual..."
python3 -m venv venv

# Activar
source venv/bin/activate

# Actualizar pip
echo "Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "Instalando dependencias..."
pip install -r requirements.txt

# Verificar instalación
echo ""
echo "Verificando instalación..."
python3 -c "
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib

print(f'✓ NumPy {np.__version__}')
print(f'✓ Pandas {pd.__version__}')
print(f'✓ TensorFlow {tf.__version__}')
print(f'✓ Matplotlib {matplotlib.__version__}')
"

echo ""
echo "========================================="
echo "Instalación completada exitosamente"
echo "========================================="
echo ""
echo "Para activar el entorno, ejecuta:"
echo "  source venv/bin/activate"
echo ""
echo "Para ejecutar tests de validación:"
echo "  pytest tests/"
