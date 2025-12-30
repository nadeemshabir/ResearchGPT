@echo off
echo ========================================
echo Starting ResearchGPT Web Interface
echo ========================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found
    echo Please create one with: python -m venv venv
    echo.
)

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Streamlit not found. Installing dependencies...
    pip install -r requirements.txt
)

REM Run the Streamlit app
echo.
echo Starting Streamlit server...
echo.
streamlit run app.py

pause
