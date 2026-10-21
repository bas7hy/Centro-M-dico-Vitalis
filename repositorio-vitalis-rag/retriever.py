"""
retriever.py
------------
Etapa 5 del pipeline: recuperación híbrida (semántica + filtro de metadata)
sobre el vector store construido por ingest.py.
"""

from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

PERSIST_DIR = Path(__file__).resolve().parent.parent / "chroma_db"


def cargar_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name="vitalis_knowledge_base",
    )


def recuperar_contexto(pregunta: str, tipo_documento: str | None = None, k: int = 4):
    """
    Busca los k chunks más relevantes para la pregunta.
    Si se especifica tipo_documento ('protocolo', 'convenio', 'faq'),
    filtra por metadata antes de la búsqueda semántica.
    """
    vectordb = cargar_vector_store()
    filtro = {"tipo_documento": tipo_documento} if tipo_documento else None
    resultados = vectordb.similarity_search(pregunta, k=k, filter=filtro)
    return resultados


if __name__ == "__main__":
    # Prueba rápida manual del retriever
    pregunta_ejemplo = "¿Necesito ayuno para la ecografía abdominal?"
    resultados = recuperar_contexto(pregunta_ejemplo)
    for i, doc in enumerate(resultados, 1):
        print(f"\n--- Resultado {i} (fuente: {doc.metadata.get('archivo_origen')}) ---")
        print(doc.page_content[:200])
