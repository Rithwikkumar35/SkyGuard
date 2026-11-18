"""
app.py
Root entrypoint for SkyGuard Streamlit dashboard.
Delegates to the modular application in app/app.py.
"""

from pathlib import Path
import runpy

app_path = Path(__file__).resolve().parent / "app" / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
