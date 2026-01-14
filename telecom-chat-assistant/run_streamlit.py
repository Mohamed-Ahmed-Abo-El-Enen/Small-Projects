import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    subprocess.run([
        "streamlit", "run",
        "app/streamlit_app.py",
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ])