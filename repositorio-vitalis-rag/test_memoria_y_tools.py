"""
test_memoria_y_tools.py
-----------------------
Pruebas de los componentes nuevos de la EP2: memoria de corto/largo plazo
y las herramientas de razonamiento/escritura que no dependen del LLM real.

Ejecutar con:
    pytest tests/test_memoria_y_tools.py -v
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory import MemoriaCortoPlazo, MemoriaLargoPlazo
from src.tools import evaluar_urgencia, crear_ticket_derivacion, registrar_interaccion


# ---------------------- Memoria de corto plazo ----------------------

def test_memoria_corto_plazo_guarda_turnos():
    memoria = MemoriaCortoPlazo()
    memoria.agregar_turno("¿Cuál es el horario?", "El horario es de 08:00 a 20:00.")
    historial = memoria.obtener_historial()
    assert len(historial) == 2  # un mensaje humano + uno de respuesta


def test_memoria_corto_plazo_reiniciar():
    memoria = MemoriaCortoPlazo()
    memoria.agregar_turno("pregunta", "respuesta")
    memoria.reiniciar()
    assert memoria.obtener_historial() == []


# ---------------------- Memoria de largo plazo ----------------------

def test_memoria_largo_plazo_persiste_y_recupera():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_memoria.db"
        memoria = MemoriaLargoPlazo(db_path=db_path)
        memoria.guardar_resumen("sesion-1", "Consultó por glicemia.")
        memoria.guardar_resumen("sesion-1", "Consultó por cobertura Fonasa.")

        historial = memoria.recuperar_historial_sesion("sesion-1")
        assert historial == ["Consultó por glicemia.", "Consultó por cobertura Fonasa."]


def test_memoria_largo_plazo_aisla_sesiones():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_memoria.db"
        memoria = MemoriaLargoPlazo(db_path=db_path)
        memoria.guardar_resumen("sesion-A", "Dato de A.")
        memoria.guardar_resumen("sesion-B", "Dato de B.")

        assert memoria.recuperar_historial_sesion("sesion-A") == ["Dato de A."]
        assert memoria.recuperar_historial_sesion("sesion-B") == ["Dato de B."]


# ---------------------- Herramientas de razonamiento ----------------------

def test_evaluar_urgencia_detecta_clinico():
    resultado = evaluar_urgencia.invoke({"consulta_paciente": "me duele mucho la guatita"})
    assert "URGENTE" in resultado
    assert "NO_URGENTE" not in resultado


def test_evaluar_urgencia_permite_administrativo():
    resultado = evaluar_urgencia.invoke({"consulta_paciente": "¿cuál es el horario de atención?"})
    assert "NO_URGENTE" in resultado


# ---------------------- Herramientas de escritura ----------------------

def test_crear_ticket_derivacion_genera_id():
    resultado = crear_ticket_derivacion.invoke({"motivo": "Consulta con señales clínicas"})
    assert "TCK-" in resultado
    assert "Consulta con señales clínicas" in resultado


def test_registrar_interaccion_persiste(monkeypatch, tmp_path):
    # Redirige la memoria de largo plazo usada por la tool a un archivo temporal
    from src import tools as tools_module
    from src.memory import MemoriaLargoPlazo as MLP

    tools_module._memoria_largo_plazo = MLP(db_path=tmp_path / "test_tool_memoria.db")

    resultado = registrar_interaccion.invoke({
        "sesion_id": "sesion-test",
        "resumen": "Consultó preparación de examen.",
    })
    assert "sesion-test" in resultado
    assert tools_module._memoria_largo_plazo.recuperar_historial_sesion("sesion-test") == [
        "Consultó preparación de examen."
    ]
