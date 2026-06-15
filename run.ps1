# Simple startup script - always uses Python from .venv
# Just run: .\run.ps1

& ".\\.venv\Scripts\python.exe" -m streamlit run esco_kg_streamlit.py --server.headless false --server.port 8501 --server.address localhost

