# Agente RAG — Centro Médico Vitalis

Evaluación Parcial N°1 — Ingeniería de Soluciones con IA (ISY0101)

Agente conversacional con arquitectura RAG (Retrieval-Augmented Generation) para
responder consultas administrativas frecuentes de un centro de salud ambulatorio
(preparación de exámenes, cobertura por convenio, horarios), con enrutamiento de
intención, guardrails de alcance clínico y trazabilidad de fuentes.

> Ver el informe técnico completo (`Informe_EP1_ISY0101_Vitalis.docx`) para el
> análisis del caso, la justificación de decisiones de diseño y el detalle de
> la arquitectura.

## Qué hace el proyecto

1. Indexa documentos internos (protocolos, convenios, FAQ) en una base vectorial (Chroma).
2. Ante una consulta, un **agente orquestador** clasifica la intención y decide la ruta:
   - `informacion_general` / `cobertura_convenio` → pipeline RAG (recuperación + generación).
   - `agendamiento` → herramienta externa simulada.
   - `requiere_derivacion_humana` → deriva a un operador.
3. Aplica **guardrails** para bloquear respuestas de alcance clínico.
4. Registra cada interacción (sin datos de pacientes) en un log de trazabilidad.

## Estructura del repositorio

```
.
├── data/                     # Documentos de ejemplo (protocolos, convenios, FAQ)
├── docs/                     # Diagrama de arquitectura y log de trazabilidad generado
├── src/
│   ├── ingest.py              # Etapas 1-4: carga, chunking, embeddings, vector store
│   ├── retriever.py            # Etapa 5: recuperación semántica + filtro de metadata
│   ├── generate.py             # Etapas 6-8: augmentation, generación, guardrails
│   ├── agent.py                # Agente orquestador con clasificador de intención
│   └── logger.py               # Registro de trazabilidad
├── tests/
│   └── test_pipeline.py        # Pruebas unitarias (no requieren API key)
├── requirements.txt
├── .env.example
└── README.md
```

## Requisitos previos

- Python 3.10 o superior
- Una cuenta y API key de OpenAI (o adaptar `src/generate.py` y `src/agent.py`
  a otro proveedor compatible con LangChain, ej. Anthropic, si se prefiere)

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/ep1-isy0101-vitalis-rag.git
cd ep1-isy0101-vitalis-rag

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

# Paso 2: probar el retriever de forma aislada (opcional)
python src/retriever.py

# Paso 3: ejecutar el agente en modo interactivo
python src/agent.py
```

Ejemplo de uso interactivo:

```
Consulta: ¿Qué preparación necesito para un examen de glicemia?

[Intención detectada: informacion_general]
Respuesta: Debes mantener ayuno estricto de 8 a 12 horas. Se permite tomar agua...
(protocolos_examenes.md)
Fuentes: ['protocolos_examenes.md']
```

## Pruebas

```bash
pytest tests/ -v
```

Las pruebas cubren: carga de documentos, metadata de trazabilidad, chunking,
y el comportamiento del guardrail ante consultas de alcance clínico vs.
administrativo. No requieren API key porque no invocan al LLM real.

## Limitaciones conocidas

- El clasificador de intención y la generación dependen de un LLM externo (no incluido).
- Los datos en `/data` son de ejemplo/simulados, no información real de pacientes.
- El guardrail de alcance clínico usa un filtro simple por palabras clave; en un
  entorno productivo debería reemplazarse por un clasificador dedicado.

## Declaración de uso de IA

Se utilizó Claude (Anthropic) como apoyo para generar el andamiaje inicial del código (estructura de módulos, boilerplate de LangChain/Chroma) y el README. Se utilizó Gemini como apoyo para estructurar la propuesta inicial del caso organizacional, así apoyar la redacción de la información buscada desde las páginas elegidas, y así también tener una ayuda para generar algunos diagramas de arquitectura. La lógica de negocio, las decisiones de diseño y la validación del funcionamiento fueron revisadas por el equipo. [Ajustar esta sección según el uso real dado]
