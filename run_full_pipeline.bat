@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0env\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

echo [1/8] Installing pinned dependencies
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 exit /b %errorlevel%

echo [2/8] Generating or resuming the synthetic CSG library
"%PYTHON%" -m scripts.generate_library --config configs/synthetic.yaml
if errorlevel 1 exit /b %errorlevel%

echo [3/8] Building train/validation classical meshes
"%PYTHON%" -m scripts.run_pipeline --split train_val
if errorlevel 1 exit /b %errorlevel%

echo [4/8] Verifying the specification data and GNN configuration
"%PYTHON%" -m scripts.check_project_data
if errorlevel 1 exit /b %errorlevel%
"%PYTHON%" -m scripts.check_gnn_spec
if errorlevel 1 exit /b %errorlevel%

echo [5/8] Training on train and calibrating on validation only
"%PYTHON%" -m src.training.train_gnn
if errorlevel 1 exit /b %errorlevel%

echo [6/8] Reviewing deterministic validation previews
"%PYTHON%" -m scripts.preview_steps --split val --n 3
if errorlevel 1 exit /b %errorlevel%
echo Review data\previews\preview_report.md and data\evaluation_reports\selected_config.json.
pause

echo [7/8] Running the final validation study
"%PYTHON%" -m scripts.run_final --split val
if errorlevel 1 exit /b %errorlevel%
echo Inspect data\evaluation_reports\final_val.csv before using the held-out test split.
set /p "CONFIRM=Run the held-out test exactly once with the frozen configuration? (Y/N): "
if /I not "%CONFIRM%"=="Y" exit /b 0

"%PYTHON%" -m scripts.run_pipeline --split test
if errorlevel 1 exit /b %errorlevel%
"%PYTHON%" -m scripts.run_final --split test --confirm-test
if errorlevel 1 exit /b %errorlevel%

echo [8/8] Analyzing only the final test results
"%PYTHON%" -m scripts.analyze_results
if errorlevel 1 exit /b %errorlevel%

echo Pipeline complete. Reports are in data\evaluation_reports.
exit /b 0
