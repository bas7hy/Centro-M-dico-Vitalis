"""
agent_v2.py
-----------
Agente funcional de la EP2 (IL2.1, IL2.3): reemplaza el router de intención
fijo de la EP1 por un AgentExecutor de LangChain que PLANIFICA sus propios
pasos (patrón ReAct: piensa → actúa con una herramienta → observa el
resultado → decide el siguiente paso), permitiendo ajustar el comportamiento
si las condiciones cambian a mitad de la conversación (IE5, IE7).

Uso interactivo:
    python src/agent_v2.py
"""

import os
import uuid

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.tools import HERRAMIENTAS_DISPONIBLES
from src.memory import MemoriaCortoPlazo, MemoriaLargoPlazo

MODEL_NAME = "gpt-4o-mini"

PROMPT_SISTEMA = """Eres un asistente administrativo del Centro Médico Vitalis.

Tu trabajo es resolver consultas administrativas (preparación de exámenes,
convenios/coberturas, horarios) planificando tú mismo qué herramientas usar
y en qué orden, según cada caso:

1. SIEMPRE evalúa primero la urgencia de la consulta con evaluar_urgencia.
2. Si es URGENTE, usa crear_ticket_derivacion y responde solo indicando que
   se derivó a un profesional humano. No intentes responder la parte clínica.
3. Si NO es urgente, usa buscar_protocolo (y consultar_horario_disponible si
   corresponde) para construir una respuesta basada en el contexto recuperado.
4. Cita siempre el documento fuente cuando uses buscar_protocolo.
5. Al final de cada consulta resuelta, usa registrar_interaccion para dejar
   evidencia en la memoria de largo plazo (sin datos identificables).
6. Si durante la conversación el paciente cambia las condiciones (por ejemplo,
   agrega síntomas a una consulta que parecía administrativa), vuelve a
   evaluar la urgencia y ajusta tu plan en consecuencia, incluso si ya habías
   empezado a responder de otra forma.

Si el contexto recuperado no contiene la respuesta, dilo explícitamente en
vez de inventarla.
"""


class AgenteVitalis:
    """
    Encapsula el AgentExecutor junto con las dos memorias (corto y largo
    plazo), exponiendo un único punto de entrada: procesar_consulta().
    """

    def __init__(self, sesion_id: str | None = None):
        self.sesion_id = sesion_id or str(uuid.uuid4())
        self.memoria_corto_plazo = MemoriaCortoPlazo()
        self.memoria_largo_plazo = MemoriaLargoPlazo()

        llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_SISTEMA),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agente = create_tool_calling_agent(llm, HERRAMIENTAS_DISPONIBLES, prompt)
        self.executor = AgentExecutor(
            agent=agente,
            tools=HERRAMIENTAS_DISPONIBLES,
            verbose=True,
            max_iterations=6,  # límite de pasos de planificación por consulta
            handle_parsing_errors=True,
        )

    def _contexto_largo_plazo(self) -> str:
        historial = self.memoria_largo_plazo.recuperar_historial_sesion(self.sesion_id)
        if not historial:
            return ""
        resumen = "\n".join(f"- {h}" for h in historial)
        return f"\n\n[Contexto de interacciones previas de esta sesión]\n{resumen}"

    def procesar_consulta(self, pregunta: str) -> dict:
        entrada_completa = pregunta + self._contexto_largo_plazo()

        resultado = self.executor.invoke({
            "input": entrada_completa,
            "chat_history": self.memoria_corto_plazo.obtener_historial(),
        })

        respuesta = resultado["output"]
        self.memoria_corto_plazo.agregar_turno(pregunta, respuesta)

        pasos = [
            {"herramienta": accion.tool, "entrada": accion.tool_input, "observacion": observacion}
            for accion, observacion in resultado.get("intermediate_steps", [])
        ]

        return {
            "respuesta": respuesta,
            "pasos_ejecutados": pasos,
            "sesion_id": self.sesion_id,
        }


def main():
    print("Agente Vitalis v2 (con memoria y planificación) — escribe 'salir' para terminar.\n")
    agente = AgenteVitalis()
    print(f"[Sesión: {agente.sesion_id}]\n")

    while True:
        pregunta = input("Consulta: ").strip()
        if pregunta.lower() in {"salir", "exit", "quit"}:
            break

        resultado = agente.procesar_consulta(pregunta)
        print(f"\nRespuesta: {resultado['respuesta']}")
        print(f"Pasos ejecutados: {[p['herramienta'] for p in resultado['pasos_ejecutados']]}\n")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Falta la variable de entorno OPENAI_API_KEY. "
            "Copia .env.example a .env y completa tu clave antes de ejecutar."
        )
    main()
