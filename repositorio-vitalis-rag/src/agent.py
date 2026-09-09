"""
agent.py
--------
Agente orquestador: clasifica la intención de la consulta y decide la ruta
(RAG, herramienta externa simulada, o derivación humana), según la
arquitectura descrita en el informe técnico (sección 4).

Uso interactivo:
    python src/agent.py
"""

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from src.generate import responder

MODEL_NAME = "gpt-4o-mini"

CATEGORIAS = [
    "informacion_general",
    "agendamiento",
    "cobertura_convenio",
    "requiere_derivacion_humana",
]

PROMPT_ENRUTAMIENTO = (
    "Clasifica la siguiente consulta en una de estas categorías: "
    f"{CATEGORIAS}. Responde solo con el nombre exacto de la categoría, sin explicación. "
    "Consulta: {pregunta}"
)


def clasificar_intencion(pregunta: str) -> str:
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    resultado = llm.invoke(
        [HumanMessage(content=PROMPT_ENRUTAMIENTO.format(pregunta=pregunta))]
    ).content.strip().lower()

    for categoria in CATEGORIAS:
        if categoria in resultado:
            return categoria
    return "informacion_general"  # fallback conservador


def herramienta_agendamiento(pregunta: str) -> str:
    """Simulación de integración con sistema de agendamiento (fuera del alcance del pipeline RAG)."""
    return (
        "Para agendar, reagendar o cancelar una hora puedes hacerlo a través del portal "
        "web del centro o llamando al call center. [Esta respuesta simula una integración "
        "real con el sistema de agendamiento; no fue generada con RAG]."
    )


def derivar_a_humano(pregunta: str) -> str:
    return (
        "Esta consulta requiere ser atendida por un profesional o por el equipo de counter. "
        "Un operador humano se pondrá en contacto contigo a la brevedad."
    )


def procesar_consulta(pregunta: str) -> dict:
    """Punto de entrada del agente: enruta la consulta y devuelve la respuesta final."""
    intencion = clasificar_intencion(pregunta)

    if intencion == "agendamiento":
        return {"intencion": intencion, "respuesta": herramienta_agendamiento(pregunta), "fuentes": []}

    if intencion == "requiere_derivacion_humana":
        return {"intencion": intencion, "respuesta": derivar_a_humano(pregunta), "fuentes": []}

    if intencion == "cobertura_convenio":
        resultado = responder(pregunta, tipo_documento="convenio")
    else:  # informacion_general -> protocolo o faq, sin restringir tipo_documento
        resultado = responder(pregunta)

    return {"intencion": intencion, "respuesta": resultado["respuesta"], "fuentes": resultado["fuentes"]}


def main():
    print("Agente Vitalis — escribe 'salir' para terminar.\n")
    while True:
        pregunta = input("Consulta: ").strip()
        if pregunta.lower() in {"salir", "exit", "quit"}:
            break
        resultado = procesar_consulta(pregunta)
        print(f"\n[Intención detectada: {resultado['intencion']}]")
        print(f"Respuesta: {resultado['respuesta']}")
        if resultado["fuentes"]:
            print(f"Fuentes: {resultado['fuentes']}")
        print()


if __name__ == "__main__":
    main()
