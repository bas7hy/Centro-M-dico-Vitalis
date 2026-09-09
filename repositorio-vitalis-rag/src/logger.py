"""
logger.py
---------
Módulo de trazabilidad: registra cada consulta respondida junto con las
fuentes documentales utilizadas, para cumplir el requisito de auditabilidad
(IE1.4) sin almacenar datos identificables de pacientes (Ley 19.628).
"""

import csv
import datetime
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "docs" / "log_trazabilidad.csv"


def registrar_interaccion(pregunta: str, intencion: str, fuentes: list[str]) -> None:
    """
    Guarda un registro de auditoría. No almacena identificadores de pacientes:
    solo la consulta (texto administrativo), la intención clasificada y las
    fuentes documentales usadas para responder.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existe = LOG_PATH.exists()

    with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(["timestamp", "consulta", "intencion", "fuentes_utilizadas"])
        writer.writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            pregunta,
            intencion,
            "; ".join(fuentes) if fuentes else "N/A",
        ])
