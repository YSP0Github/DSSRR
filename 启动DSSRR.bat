@echo off
rem ===========================================================================
rem  DSSRR GUI launcher (Windows)
rem
rem  Double-click this file to start the DSSRR desktop application.
rem
rem  Requirements: PyQt5 + matplotlib (+ obspy if you want to open MiniSEED
rem  files).  Install them with:
rem
rem      pip install -e ".[gui]"
rem
rem  To use a specific interpreter (e.g. a conda env) without activating it,
rem  set the environment variable DSSRR_PYTHON before running, for example:
rem
rem      set DSSRR_PYTHON=C:\path\to\envs\myenv\python.exe
rem
rem ===========================================================================
setlocal
title DSSRR GUI

rem Always run from the directory that contains this script, so the package
rem is importable even when the repo has not been pip-installed.
cd /d "%~dp0"

if defined DSSRR_PYTHON (
    set "PY=%DSSRR_PYTHON%"
) else (
    set "PY=python"
)

%PY% -m dssrr.gui
if errorlevel 1 (
    echo.
    echo ---------------------------------------------------------------------
    echo  DSSRR GUI failed to start.
    echo.
    echo  Most likely PyQt5 / matplotlib are missing.  Install them with:
    echo      pip install -e ".[gui]"
    echo.
    echo  Or point DSSRR_PYTHON at an interpreter that already has them.
    echo ---------------------------------------------------------------------
    pause
)

endlocal
