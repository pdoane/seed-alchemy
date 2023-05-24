if exist .venv rmdir /s/q .venv
python -m venv .venv --prompt SeedAlchemy
call .venv\Scripts\activate.bat
pip install -r requirements.txt
call .venv\Scripts\deactivate.bat
