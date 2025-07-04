import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


def cargar_documentos(ruta_archivo):
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"El archivo {ruta_archivo} no existe.")
    
    if not ruta_archivo.endswith('.xlsx'):
        raise ValueError("Solo se admiten archivos Excel (.xlsx)")
    
    # Cargar archivo Excel
    df = pd.read_excel(ruta_archivo)
    
    # Convertir cada fila del DataFrame en un documento
    documentos = []
    for index, row in df.iterrows():
        # Crear contenido de texto combinando todas las columnas
        contenido = ""
        for col in df.columns:
            if pd.notna(row[col]):  # Solo agregar valores no nulos
                contenido += f"{col}: {row[col]}\n"
        
        # Crear documento con metadatos
        doc = Document(
            page_content=contenido,
            metadata={
                "file_path": ruta_archivo,
                "row": index
            }
        )
        documentos.append(doc)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    docs = text_splitter.split_documents(documentos)
    return docs

def crear_vectorstore(docs):
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory="chroma_db_dir",
        collection_name="stanford_report_data"
    )
    return vectorstore