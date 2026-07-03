@echo off
setlocal

cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON=py -3"
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    set "PYTHON=python"
  ) else (
    echo Python 3 was not found.
    echo Install Python from https://www.python.org/downloads/windows/
    pause
    exit /b 1
  )
)

%PYTHON% "HCI-images-deep learning\test_2.py"
set "STATUS=%errorlevel%"

if not "%STATUS%"=="0" (
  echo.
  echo The app stopped with an error.
  pause
)

exit /b %STATUS%
