
import sys
import os

# Ensure Django app is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from portfolio.fastapi_app import app  # Import FastAPI app
