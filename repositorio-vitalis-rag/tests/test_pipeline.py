"""
test_pipeline.py
-----------------
Pruebas básicas del pipeline, como evidencia de testing exigida por la pauta.
No requieren API key: prueban las partes deterministas (carga, chunking,
guardrails, clasificación por palabras clave) sin llamar al LLM real.

Ejecutar con:
    pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import cargar_documentos, chunkear
from src.generate import aplicar_guardrails, construir_prompt_rag


def test_cargar_documentos_no_vacio():
    documentos = cargar_documentos()
    assert len(documentos) > 0, "Debe cargar al menos un documento desde /data"


def test_cargar_documentos_tiene_metadata_fuente():
    documentos = cargar_documentos()
    for doc in documentos:
        assert "archivo_origen" in doc.metadata
        assert "tipo_documento" in doc.metadata


def test_chunkear_respeta_tamano_maximo():
    documentos = cargar_documentos()
    chunks = chunkear(documentos)
    assert len(chunks) >= len(documentos), "El chunking no debería reducir el número de fragmentos"
    for chunk in chunks:
        assert len(chunk.page_content) <= 900  # chunk_size (800) + margen de separadores


def test_guardrail_bloquea_consulta_clinica():
    respuesta_original = "Aquí tienes información sobre tu síntoma."
    resultado = aplicar_guardrails("me duele mucho la guatita, ¿qué tengo?", respuesta_original)
    assert "profesional de salud" in resultado.lower() or "derivación" in resultado.lower() or "urgencia" in resultado.lower()
    assert resultado != respuesta_original, "El guardrail debe sobreescribir respuestas ante señales clínicas"


def test_guardrail_permite_consulta_administrativa():
    respuesta_original = "El horario de atención es de 08:00 a 20:00 horas."
    resultado = aplicar_guardrails("¿Cuál es el horario de atención?", respuesta_original)
    assert resultado == respuesta_original, "Consultas administrativas no deben ser bloqueadas por el guardrail"


def test_construir_prompt_rag_incluye_fuentes():
    class ChunkFalso:
        def __init__(self, contenido, meta):
            self.page_content = contenido
            self.metadata = meta

    chunks_falsos = [
        ChunkFalso("Ayuno de 8 horas.", {"archivo_origen": "protocolos_examenes.md", "titulo": "Glicemia"})
    ]
    prompt = construir_prompt_rag("¿Necesito ayuno?", chunks_falsos)
    assert "protocolos_examenes.md" in prompt
    assert "Ayuno de 8 horas." in prompt
