"""
memory.py
---------
Módulo de memoria del agente (IE3, IE4 de la EP2).

- Memoria de corto plazo: historial de mensajes de la conversación actual,
  usado por el agente para mantener coherencia dentro de un mismo intercambio
  (ej. "¿y con mi otro convenio?" recordando el convenio mencionado antes).

- Memoria de largo plazo: resúmenes persistentes por sesión/paciente,
  almacenados en SQLite, para dar continuidad a tareas que cruzan múltiples
  conversaciones (ej. el paciente vuelve días después y el agente puede
  recuperar que ya había consultado por un examen específico).

No se almacenan identificadores de pacientes (nombre, RUT, etc.), solo un
identificador de sesión anónimo, en línea con la Ley 19.628 (ver EP1).
"""

import sqlite3
import datetime
from pathlib import Path
from langchain.memory import ConversationBufferMemory

DB_PATH = Path(__file__).resolve().parent.parent / "memoria" / "memoria_largo_plazo.db"


class MemoriaCortoPlazo:
    """
    Envoltorio sobre ConversationBufferMemory de LangChain: mantiene el
    historial de la conversación actual en memoria volátil (no persiste
    entre ejecuciones del programa).
    """

    def __init__(self):
        self.buffer = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

    def agregar_turno(self, entrada: str, salida: str) -> None:
        self.buffer.save_context({"input": entrada}, {"output": salida})

    def obtener_historial(self):
        return self.buffer.load_memory_variables({})["chat_history"]

    def reiniciar(self) -> None:
        self.buffer.clear()


class MemoriaLargoPlazo:
    """
    Persistencia simple en SQLite de resúmenes de interacciones por sesión.
    Cada fila es un resumen breve, no la conversación completa, para evitar
    almacenar datos sensibles y mantener el contexto recuperable liviano.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._inicializar_tabla()

    def _inicializar_tabla(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS resumenes_sesion (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sesion_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    resumen TEXT NOT NULL
                )
                """
            )

    def guardar_resumen(self, sesion_id: str, resumen: str) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO resumenes_sesion (sesion_id, timestamp, resumen) VALUES (?, ?, ?)",
                (sesion_id, datetime.datetime.now().isoformat(timespec="seconds"), resumen),
            )

    def recuperar_historial_sesion(self, sesion_id: str, limite: int = 5) -> list[str]:
        """Devuelve los últimos `limite` resúmenes de una sesión, del más antiguo al más reciente."""
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute(
                """
                SELECT resumen FROM resumenes_sesion
                WHERE sesion_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (sesion_id, limite),
            )
            filas = cur.fetchall()
        return [fila[0] for fila in reversed(filas)]


if __name__ == "__main__":
    # Prueba manual rápida
    mlp = MemoriaLargoPlazo()
    mlp.guardar_resumen("sesion-demo-001", "Consultó preparación para examen de glicemia.")
    print(mlp.recuperar_historial_sesion("sesion-demo-001"))
