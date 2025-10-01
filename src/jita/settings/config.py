from pathlib import Path
from datetime import date
import locale  
locale.setlocale(locale.LC_TIME, "es_MX.UTF-8")

def find_project_root(start_path: Path, marker_file: str = "pyproject.toml") -> Path:
    current = start_path.resolve()
    while not (current / marker_file).exists() and current != current.parent:
        current = current.parent
    return current

BASE_DIR = find_project_root(Path(__file__))

DATA = BASE_DIR / "datos"

CASA_LEY_DATA = DATA / "casa_ley" / date.today().strftime("%B_%Y")

VECTORS_DIR = BASE_DIR / "datos" / "vectorstores"
PRODUCTS_VECTOR_PATH = VECTORS_DIR / "products_vector_store"
SALES_PRODUCTS_VECTOR_PATH = VECTORS_DIR / "sales_products_vector_store"
SUPPORT_INFO_VECTOR_PATH = VECTORS_DIR / "guarantees_vector_store"

ID_SUCURSAL = BASE_DIR / "datos" / "idSucursal.json"
BASE_KNOWLEDGE = BASE_DIR / "datos" / "base_de_conocimientos"

for path in [DATA, CASA_LEY_DATA]:
    path.mkdir(parents=True, exist_ok=True)
