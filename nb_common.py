import nbformat as nbf
from pathlib import Path

BASE_DIR = Path("D:/Downloads/credit_risk")


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def save(cells, path, nb_name):
    output_dir = BASE_DIR / "notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / Path(path).name

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (credit_risk)",
            "language": "python",
            "name": "credit_risk",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    with open(target_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", target_path, "cells:", len(cells))


SETUP_SNIPPET = '''
import os
from pathlib import Path

# ---------------------------------------------------------------
# LOCAL PATH CONFIG
# Edit BASE_DIR if your project folder is somewhere else.
# Everything else (data / processed / models folders) is derived
# from this one path, so you only ever need to change it here.
# ---------------------------------------------------------------
BASE_DIR = Path("D:/Downloads/credit_risk")

DATA_DIR = BASE_DIR / "data"                 # put the raw Home Credit CSVs here
PROCESSED_DIR = BASE_DIR / "data" / "processed"  # outputs of notebook 02 land here
MODELS_DIR = BASE_DIR / "models"             # outputs of notebook 03 (best model, metrics, test data)
IMAGES_DIR = BASE_DIR / "images"             # outputs of notebook 04 (plots and curves)

for _p in (DATA_DIR, PROCESSED_DIR, MODELS_DIR, IMAGES_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# Databricks' `display()` shows a rendered widget. Plain Jupyter doesn't have
# this by default, so we fall back to IPython's display (renders DataFrames
# as HTML tables) and finally to plain print if that's unavailable.
try:
    from IPython.display import display
except ImportError:
    display = print

import warnings
warnings.filterwarnings("ignore")
'''

