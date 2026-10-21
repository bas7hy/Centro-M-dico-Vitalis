"""
ingest.py
---------
Etapas 1-4 del pipeline RAG: ingesta de documentos, chunking,
generación de embeddings y almacenamiento en el vector store (Chroma).

Uso:
    python src/ingest.py
"""

import os
import re
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PERSIST_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

# Mapeo de archivo -> metadata de tipo de documento (usado luego para filtrar en el retriever)
DOC_METADATA = {
    "protocolos_examenes.md": {"tipo_documento": "protocolo", "fuente": "interna"},
    "convenios_cobertura.md": {"tipo_documento": "convenio", "fuente": "interna"},
    "faq_recepcion.md": {"tipo_documento": "faq", "fuente": "interna"},
}


def cargar_documentos() -> list[Document]:
    """Carga los archivos markdown de /data y separa cada entrada (## encabezado) en un Document."""
    documentos = []
    for filename, meta in DOC_METADATA.items():
        filepath = DATA_DIR / filename
        if not filepath.exists():
            continue
        texto = filepath.read_text(encoding="utf-8")
        # Cada sección "## Titulo ... " se separa como un documento independiente,
        # preservando el nombre del archivo de origen como fuente citable.
        secciones = re.split(r"\n(?=## )", texto)
        for seccion in secciones:
            if not seccion.strip().startswith("##"):
                continue
            titulo = seccion.splitlines()[0].replace("## ", "").strip()
            documentos.append(
                Document(
                    page_content=seccion.strip(),
                    metadata={
                        **meta,
                        "titulo": titulo,
                        "archivo_origen": filename,
                    },
                )
            )
    return documentos


def chunkear(documentos: list[Document]) -> list[Document]:
    """Etapa 2: chunking con solapamiento, según lo definido en el informe técnico (500-800 tokens aprox, overlap 100)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(documentos)


def construir_vector_store(chunks: list[Document]) -> Chroma:
    """Etapas 3-4: genera embeddings y persiste en Chroma."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_name="vitalis_knowledge_base",
    )
    vectordb.persist()
    return vectordb


def main():
    print("Cargando documentos desde /data ...")
    documentos = cargar_documentos()
    print(f"  {len(documentos)} documentos cargados.")

    print("Generando chunks ...")
    chunks = chunkear(documentos)
    print(f"  {len(chunks)} chunks generados.")

    print("Construyendo vector store (Chroma) ...")
    construir_vector_store(chunks)
    print(f"Listo. Base vectorial persistida en: {PERSIST_DIR}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Falta la variable de entorno OPENAI_API_KEY. "
            "Copia .env.example a .env y completa tu clave antes de ejecutar."
        )
    main()
