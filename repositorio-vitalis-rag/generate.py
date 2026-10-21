"""
generate.py
-----------
Etapas 6-8 del pipeline: augmentation (armado del prompt con contexto),
generación de la respuesta y guardrails post-generación.
"""

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from src.retriever import recuperar_contexto

MODEL_NAME = "gpt-4o-mini"  # Reemplazar por el modelo disponible en tu proveedor

PROMPT_SISTEMA = (
    "Eres un asistente administrativo del Centro Médico Vitalis. "
    "Respondes únicamente sobre preparación de exámenes, convenios/coberturas y horarios, "
    "usando exclusivamente la información recuperada de los documentos oficiales entregados "
    "en el contexto. Si la consulta implica diagnóstico, tratamiento o síntomas, indica que "
    "debes derivar a un profesional de salud y no respondas. Cita siempre el documento fuente."
)

# Palabras clave simples usadas por el guardrail para detectar posible contenido clínico.
# (En una versión productiva esto se reemplazaría por un clasificador dedicado.)
PALABRAS_ALCANCE_CLINICO = [
    "me duele", "síntoma", "diagnóstico", "qué tengo", "es grave",
    "tratamiento", "dosis", "receta", "tomar medicamento",
]


def construir_prompt_rag(pregunta: str, chunks: list) -> str:
    contexto = "\n\n".join(
        f"[Fuente: {c.metadata.get('archivo_origen')} — {c.metadata.get('titulo')}]\n{c.page_content}"
        for c in chunks
    )
    return (
        f"Contexto recuperado:\n{contexto}\n\n"
        f"Pregunta del usuario: {pregunta}\n\n"
        "Responde de forma clara y breve, citando el nombre del documento de origen entre "
        "paréntesis. Si el contexto no contiene la respuesta, indícalo explícitamente en "
        "vez de inventarla."
    )


def aplicar_guardrails(pregunta: str, respuesta: str) -> str:
    """
    Guardrail simple: si la pregunta original contiene señales de consulta clínica,
    se sobreescribe la respuesta indicando derivación a un profesional, sin importar
    lo que haya generado el modelo.
    """
    pregunta_lower = pregunta.lower()
    if any(palabra in pregunta_lower for palabra in PALABRAS_ALCANCE_CLINICO):
        return (
            "Esta consulta parece requerir evaluación clínica. No puedo entregar orientación "
            "médica por este medio: por favor agenda una hora con un profesional de salud o "
            "comunícate con el equipo de urgencia del centro."
        )
    return respuesta


def responder(pregunta: str, tipo_documento: str | None = None) -> dict:
    """Orquesta augmentation + generación + guardrails para una pregunta dada."""
    chunks = recuperar_contexto(pregunta, tipo_documento=tipo_documento)

    if not chunks:
        return {
            "respuesta": "No encontré información suficiente en los documentos disponibles para responder esto con certeza.",
            "fuentes": [],
        }

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    prompt_usuario = construir_prompt_rag(pregunta, chunks)

    respuesta_llm = llm.invoke(
        [SystemMessage(content=PROMPT_SISTEMA), HumanMessage(content=prompt_usuario)]
    ).content

    respuesta_final = aplicar_guardrails(pregunta, respuesta_llm)

    return {
        "respuesta": respuesta_final,
        "fuentes": [c.metadata.get("archivo_origen") for c in chunks],
    }


if __name__ == "__main__":
    ejemplo = responder("¿Qué preparación necesito para un examen de glicemia?")
    print(ejemplo["respuesta"])
    print("Fuentes:", ejemplo["fuentes"])
