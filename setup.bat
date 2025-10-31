@echo off
echo =========================================
echo Configuracion del entorno para
echo Tesis: Estimacion de Humedad con SAR
echo =========================================

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado
    exit /b 1
)

echo Python detectado

REM Crear entorno virtual
echo Creando entorno virtual...
python -m venv venv

REM Activar
call venv\Scripts\activate.bat

REM Actualizar pip
echo Actualizando pip...
python -m pip install --upgrade pip

REM Instalar dependencias
echo Instalando dependencias...
pip install -r requirements.txt

echo.
echo =========================================
echo Instalacion completada exitosamente
echo =========================================
echo.
echo Para activar el entorno, ejecuta:
echo   venv\Scripts\activate.bat
