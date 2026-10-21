"""
tools.py
--------
Herramientas (tools) del agente, categorizadas según lo exigido por la EP2:
consulta, escritura y razonamiento (IE1, IE2).

Cada función se expone como @tool de LangChain para que el agente (un
AgentExecutor con un LLM que razona en formato ReAct) pueda invocarlas de
forma autónoma, decidiendo cuáles usar y en qué orden según el caso.
"""

import datetime
import random

from langchain.tools import tool

from src.retriever import recuperar_contexto
from src.memory import MemoriaLargoPlazo

_memoria_largo_plazo = MemoriaLargoPlazo()

PALABRAS_URGENCIA = [
    "me duele", "síntoma", "diagnóstico", "qué tengo", "es grave",
    "tratamiento", "dosis", "receta", "sangrado", "no puedo respirar",
    "dolor fuerte", "desmayo",
]


# ---------------------- Herramientas de CONSULTA ----------------------

@tool
def buscar_protocolo(pregunta: str) -> str:
    """
    Busca información administrativa en los documentos oficiales del centro
    (protocolos de examen, convenios, horarios/FAQ) usando recuperación
    semántica (RAG). Úsala para cualquier pregunta sobre preparación de
    exámenes, cobertura o información general del centro.
    """
    chunks = recuperar_contexto(pregunta, k=3)
    if not chunks:
        return "No se encontró información relevante en los documentos disponibles."
    resultado = []
    for c in chunks:
        fuente = c.metadata.get("archivo_origen", "desconocida")
        resultado.append(f"[Fuente: {fuente}] {c.page_content}")
    return "\n\n".join(resultado)


@tool
def consultar_horario_disponible(especialidad: str) -> str:
    """
    Consulta (simulada) la disponibilidad de horas para una especialidad dada
    (ej. 'medicina general', 'laboratorio', 'imagenología'). Úsala cuando el
    paciente quiera saber si hay horas disponibles o quiera agendar.
    """
    # Simulación determinista para fines de evaluación/demo: en producción
    # esto llamaría a la API real del sistema de agendamiento del centro.
    horas_simuladas = random.Random(especialidad).sample(
        ["09:00", "10:30", "11:15", "15:00", "16:45"], k=2
    )
    return (
        f"Horas disponibles simuladas para {especialidad}: "
        f"{', '.join(horas_simuladas)} (próximos días hábiles)."
    )


# ---------------------- Herramientas de RAZONAMIENTO ----------------------

@tool
def evaluar_urgencia(consulta_paciente: str) -> str:
    """
    Evalúa si una consulta del paciente contiene señales de urgencia clínica
    (dolor, síntomas, solicitud de diagnóstico o tratamiento) que requieran
    derivación inmediata a un profesional de salud, en vez de una respuesta
    administrativa. Úsala SIEMPRE como primer paso antes de responder.
    Devuelve 'URGENTE' o 'NO_URGENTE'.
    """
    texto = consulta_paciente.lower()
    if any(palabra in texto for palabra in PALABRAS_URGENCIA):
        return "URGENTE: la consulta contiene señales de alcance clínico. Debe derivarse a un humano y no debe responderse con información administrativa."
    return "NO_URGENTE: la consulta es de carácter administrativo y puede resolverse con las herramientas de consulta disponibles."


# ---------------------- Herramientas de ESCRITURA ----------------------

@tool
def crear_ticket_derivacion(motivo: str) -> str:
    """
    Crea un ticket de derivación a un operador humano cuando la consulta
    requiere atención clínica o excede el alcance del agente. Úsala cuando
    evaluar_urgencia devuelva 'URGENTE' o cuando no exista información
    suficiente para responder con certeza.
    """
    ticket_id = f"TCK-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # En producción, esto escribiría en un sistema de tickets real (ej. Zendesk, Jira).
    return f"Ticket {ticket_id} creado. Motivo: '{motivo}'. Un operador humano será notificado."


@tool
def registrar_interaccion(sesion_id: str, resumen: str) -> str:
    """
    Registra un resumen breve de la interacción actual en la memoria de largo
    plazo del agente, para dar continuidad a futuras conversaciones de la
    misma sesión/paciente. Úsala al final de cada consulta resuelta.
    No incluyas datos identificables del paciente en el resumen.
    """
    _memoria_largo_plazo.guardar_resumen(sesion_id, resumen)
    return f"Interacción registrada en memoria de largo plazo para sesión '{sesion_id}'."


HERRAMIENTAS_DISPONIBLES = [
    buscar_protocolo,
    consultar_horario_disponible,
    evaluar_urgencia,
    crear_ticket_derivacion,
    registrar_interaccion,
]
