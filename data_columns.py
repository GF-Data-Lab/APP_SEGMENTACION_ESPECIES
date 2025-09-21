"""Canonical data column names and helpers for the segmentation app."""
from __future__ import annotations

from typing import Dict
import unicodedata
import re
import pandas as pd

COL_ESPECIE = "Especie"
COL_VARIEDAD = "Variedad"
COL_FRUTO = "N° Muestra"
COL_BRIX = "Solidos solubles (%)"
COL_ACIDEZ = "Acidez (%)"
COL_TEMPORADA = "Temporada"
COL_CAMPO = "Campo"
COL_PORTAINJERTO = "Portainjerto"
COL_FECHA_COSECHA = "Fecha cosecha"
COL_FECHA_EVALUACION = "Fecha evaluación"
COL_COLOR_PULPA = "Color de pulpa"
COL_PESO = "Peso (g)"
COL_PUNTA = "Punta"
COL_QUILLA = "Quilla"
COL_HOMBRO = "Hombro"
COL_MEJILLA_1 = "Mejilla 1"
COL_MEJILLA_2 = "Mejilla 2"
COL_TOTAL_FRUTOS = "Total frutos"

# Normalization map keyed by sanitized column tokens.
_NORMALIZATION_MAP: Dict[str, str] = {
    "especie": COL_ESPECIE,
    "variedad": COL_VARIEDAD,
    "campo": COL_CAMPO,
    "cuartel": "Cuartel",
    "pmg": "PMG",
    "portainjerto": COL_PORTAINJERTO,
    "temporada": COL_TEMPORADA,
    "season": COL_TEMPORADA,
    "temp": COL_TEMPORADA,
    "fecha cosecha": COL_FECHA_COSECHA,
    "fecha evaluacion": COL_FECHA_EVALUACION,
    "fecha de evaluacion": COL_FECHA_EVALUACION,
    "fecha evaluacion": COL_FECHA_EVALUACION,
    "fecha de evaluacion": COL_FECHA_EVALUACION,
    "fecha evaluacin": COL_FECHA_EVALUACION,
    "evaluacion": "Evaluación",
    "raleo frutos pl": "Raleo (frutos/pl)",
    "planta": "Planta",
    "perimetro": "Perimetro",
    "rendimiento": "Rendimiento",
    "repeticion": "Repetición",
    "fruto n": COL_FRUTO,
    "fruto nro": COL_FRUTO,
    "fruto numero": COL_FRUTO,
    "n muestra": COL_FRUTO,
    "nro muestra": COL_FRUTO,
    "muestra": COL_FRUTO,
    "solidos solubles %": COL_BRIX,
    "solidos solubles": COL_BRIX,
    "acidez %": COL_ACIDEZ,
    "acidez": COL_ACIDEZ,
    "color de pulpa": COL_COLOR_PULPA,
    "peso g": COL_PESO,
    "diametro mm": "Diametro (mm)",
    "punta": COL_PUNTA,
    "quilla": COL_QUILLA,
    "hombro": COL_HOMBRO,
    "mejilla 1": COL_MEJILLA_1,
    "mejilla 2": COL_MEJILLA_2,
    "traslucidez": "Traslucidez",
    "gelificacion": "Gelificación",
    "harinosidad": "Harinosidad",
    "total": "Total",
    "total frutos": COL_TOTAL_FRUTOS,
    "total frutos 1": "Total frutos.1",
    "total frutos 2": "Total frutos.2",
    "total frutos 3": "Total frutos.3",
    "observaciones": "Observaciones",
}


def _normalize_key(value: str) -> str:
    """Return a simplified lowercase key for column matching."""
    if value is None:
        return ""
    text = str(value).strip()
    # Replace typical artefacts before normalization
    text = text.replace("º", "").replace("°", "").replace("�", "")
    text = text.replace("_", " ")
    # Remove slashes but keep meaning with space
    text = text.replace("/", " ")
    # Normalize accents
    text = unicodedata.normalize("NFKD", text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9% ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known columns to canonical names and strip whitespace."""
    rename_map: Dict[str, str] = {}
    existing = set(df.columns)

    for col in df.columns:
        canonical = _NORMALIZATION_MAP.get(_normalize_key(col))
        if canonical and canonical != col:
            # Avoid collisions with existing columns (including ones queued for rename)
            if canonical not in existing and canonical not in rename_map.values():
                rename_map[col] = canonical
                existing.add(canonical)
            else:
                # Skip renaming if it would overwrite a different column
                continue

    if rename_map:
        df = df.rename(columns=rename_map)

    # Strip residual whitespace from column labels
    df.columns = [str(c).strip() for c in df.columns]
    return df


__all__ = [
    "COL_ESPECIE",
    "COL_VARIEDAD",
    "COL_FRUTO",
    "COL_BRIX",
    "COL_ACIDEZ",
    "COL_TEMPORADA",
    "COL_CAMPO",
    "COL_PORTAINJERTO",
    "COL_FECHA_COSECHA",
    "COL_FECHA_EVALUACION",
    "COL_COLOR_PULPA",
    "COL_PESO",
    "COL_PUNTA",
    "COL_QUILLA",
    "COL_HOMBRO",
    "COL_MEJILLA_1",
    "COL_MEJILLA_2",
    "COL_TOTAL_FRUTOS",
    "standardize_columns",
]
