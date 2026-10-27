
import sys
from pathlib import Path

# Ensure the project root is available to Python
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import streamlit as st
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.config import MODEL_DIR, OUTPUT_DIR
from src.models import build_model, MODEL_NAMES
from src.explainability import (
    image_to_array,
    predict_batch,
    make_lime_explanation,
    make_shap_explanation,
)
