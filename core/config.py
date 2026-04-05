"""Biến môi trường (DB_URI, MODEL_PATH, WEIGHTS)"""

import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("DB_URI", "postgresql://leafsearch:leafsearch123@localhost:5432/leafsearch_db")
MODEL_PATH = os.getenv("MODEL_PATH", "models/")
WEIGHTS = os.getenv("WEIGHTS", "0.2,0.3,0.25,0.15,0.1")

FEATURE_WEIGHTS = [float(w) for w in WEIGHTS.split(",")]
