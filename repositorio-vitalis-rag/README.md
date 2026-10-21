# Agente RAG — Centro Médico Vitalis

Ingeniería de Soluciones con IA (ISY0101) — Evaluación Parcial N°1 y N°2

Agente conversacional para un centro de salud ambulatorio, que resuelve consultas
administrativas frecuentes (preparación de exámenes, cobertura por convenio,
horarios) combinando recuperación aumentada (RAG), memoria y planificación
autónoma.

> - EP1: `Informe_EP1_ISY0101_Vitalis.docx` — análisis del caso, pipeline RAG y arquitectura inicial.
> - EP2: `Informe_EP2_ISY0101_Vitalis.docx` — agente funcional, memoria y orquestación (este README cubre ambas entregas).

## Qué hace el proyecto

**Versión EP1 (`src/agent.py`)** — pipeline RAG con enrutamiento fijo de intención:
clasifica la consulta y decide una ruta predefinida (RAG, herramienta simulada o
derivación humana), sin memoria entre turnos.

**Versión EP2 (`src/agent_v2.py`, recomendada)** — agente funcional con un
`AgentExecutor` (LangChain) que **planifica sus propios pasos** en vez de seguir
una ruta fija:

1. Evalúa la urgencia de la consulta (herramienta de razonamiento).
2. Decide qué herramientas de consulta usar (buscar protocolo, consultar horario).
3. Si la situación cambia a mitad de conversación (ej. aparecen síntomas),
   **replanifica** y deriva a un humano en vez de continuar el plan original.
4. Registra cada interacción resuelta en memoria de largo plazo.
5. Mantiene memoria de corto plazo (historial de la conversación activa) para
   resolver referencias como "¿y con mi otro convenio?".

## Estructura del repositorio

```
.
├── data/                        # Documentos de ejemplo (protocolos, convenios, FAQ)
├── docs/
│   ├── arquitectura.png          # Diagrama de arquitectura RAG (EP1)
│   └── orquestacion.png          # Diagrama de orquestación del agente (EP2)
├── memoria/                     # Base SQLite de memoria de largo plazo (se crea al ejecutar)
├── src/
│   ├── ingest.py                  # Etapas 1-4 del pipeline: carga, chunking, embeddings, vector store
│   ├── retriever.py                # Recuperación semántica + filtro de metadata
│   ├── generate.py                 # Augmentation, generación y guardrails (usado por agent.py v1)
│   ├── agent.py                     # [EP1] Agente con enrutamiento fijo de intención
│   ├── agent_v2.py                  # [EP2] Agente funcional con planificación ReAct
│   ├── tools.py                     # [EP2] Herramientas: consulta, escritura y razonamiento
│   ├── memory.py                    # [EP2] Memoria de corto y largo plazo
│   └── logger.py                    # Registro de trazabilidad (EP1)
├── tests/
│   ├── test_pipeline.py            # Pruebas del pipeline RAG (EP1)
│   └── test_memoria_y_tools.py      # [EP2] Pruebas de memoria y herramientas
├── requirements.txt
├── .env.example
└── README.md
```

## Requisitos previos

- Python 3.10 o superior
- Una cuenta y API key de OpenAI (o adaptar `src/agent_v2.py`, `src/generate.py`
  y `src/agent.py` a otro proveedor compatible con LangChain, ej. Anthropic)

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/ep-isy0101-vitalis-agente.git
cd ep-isy0101-vitalis-agente

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar la API key
cp .env.example .env
# Editar .env y completar OPENAI_API_KEY=tu_clave_real
```

## Ejecución

```bash
# Paso 1: indexar los documentos (crea la base vectorial en /chroma_db)
python src/ingest.py

# Paso 2 (EP2 — recomendado): ejecutar el agente funcional con memoria y planificación
python src/agent_v2.py

# Alternativa (EP1): ejecutar el agente con enrutamiento fijo de intención
python src/agent.py
```

Ejemplo de uso interactivo (`agent_v2.py`):

```
Agente Vitalis v2 (con memoria y planificación) — escribe 'salir' para terminar.
[Sesión: 3f1b2c4d-...]

Consulta: ¿Qué preparación necesito para un examen de glicemia?

Respuesta: Debes mantener ayuno estricto de 8 a 12 horas. Se permite tomar
agua. No es necesario suspender medicamentos habituales salvo indicación
médica expresa. (protocolos_examenes.md)
Pasos ejecutados: ['evaluar_urgencia', 'buscar_protocolo', 'registrar_interaccion']

Consulta: en realidad también me duele mucho el estómago

Respuesta: Esta consulta parece requerir evaluación clínica. Se ha derivado
a un profesional de salud (Ticket TCK-20261020-...).
Pasos ejecutados: ['evaluar_urgencia', 'crear_ticket_derivacion']
```

Este segundo ejemplo muestra la **toma de decisiones adaptativa** (IE5, IE7):
el agente detecta el cambio de condiciones dentro de la misma conversación y
ajusta su plan, derivando en lugar de continuar con una respuesta administrativa.

## Pruebas

```bash
pytest tests/ -v
```

- `test_pipeline.py` (EP1): carga de documentos, chunking, guardrails del pipeline RAG.
- `test_memoria_y_tools.py` (EP2): persistencia de memoria de corto/largo plazo,
  aislamiento entre sesiones, y comportamiento de las herramientas de
  razonamiento (`evaluar_urgencia`) y escritura (`crear_ticket_derivacion`,
  `registrar_interaccion`).

Ninguna prueba requiere API key: no invocan al LLM real, solo los componentes
deterministas del sistema.

## Arquitectura y orquestación

- `docs/arquitectura.png` — arquitectura general del pipeline RAG (EP1).
- `docs/orquestacion.png` — orquestación del agente funcional (EP2): planificador
  ReAct, las cinco herramientas (consulta, escritura, razonamiento), y los dos
  tipos de memoria.

## Limitaciones conocidas

- La planificación y clasificación de urgencia dependen de un LLM externo (no incluido).
- Los datos en `/data` son de ejemplo/simulados, no información real de pacientes.
- `evaluar_urgencia` combina razonamiento del LLM con un filtro de palabras
  clave como respaldo; en producción se recomendaría un clasificador dedicado
  y validado clínicamente.
- La memoria de largo plazo usa un identificador de sesión anónimo (no datos
  de pacientes), conforme a la Ley 19.628 — ver informe técnico EP1, sección 1.5.
- `consultar_horario_disponible` es una simulación; una integración real
  requeriría conectar la API del sistema de agendamiento del centro.

## Declaración de uso de IA

Se utilizó Gemini como apoyo para estructurar la propuesta inicial del caso
organizacional, así como para apoyar la redacción de la información buscada
desde las páginas elegidas, y también como ayuda para generar algunos
diagramas de arquitectura (EP1).

Para la Evaluación Parcial 2, se utilizó adicionalmente Claude (Anthropic)
como apoyo para estructurar el código del agente funcional (módulos de
memoria, herramientas y orquestación), el diagrama de orquestación y este
README. Las decisiones de diseño, la lógica de negocio y la validación del
funcionamiento fueron revisadas por el equipo. [Ajustar esta sección según el
uso real dado por el equipo, tal como exige la pauta de la evaluación.]
