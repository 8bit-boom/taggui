@echo off
git pull
call venv\Scripts\activate.bat
pip install -r requirements.txt --upgrade
pause
