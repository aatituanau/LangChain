from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from upload_data import cargar_documentos, crear_vectorstore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Configuración inicial
ruta_archivo = "salert.xlsx"
GOOGLE_API_KEY = "AIzaSyACorD2ho1LR1a4uY92K06h6NXXflKZg5E"

def iniciar_chat(ruta_archivo):
    # Inicializar modelos
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )
    
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Cargar o crear vectorstore
    vectorstore = Chroma(
        embedding_function=embed_model,
        persist_directory="chroma_db_dir",
        collection_name="stanford_report_data"
    )
    
    if len(vectorstore.get()['ids']) == 0:
        docs = cargar_documentos(ruta_archivo)
        vectorstore = crear_vectorstore(docs)

    # Configurar QA chain
    prompt = PromptTemplate(
        template="""Responde basándote únicamente en el contexto proporcionado.
Contexto: {context}
Pregunta: {question}
Respuesta concisa en español:""",
        input_variables=['context', 'question']
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 4}),
        chain_type_kwargs={"prompt": prompt}
    )

    # Interfaz de chat simple
    print("Sistema listo. Escribe 'salir' para terminar.")
    while True:
        pregunta = input("Tú: ")
        if pregunta.lower() == 'salir':
            break
            
        respuesta = qa.invoke({"query": pregunta})
        print("Asistente:", respuesta['result'])

if __name__ == "__main__":
    iniciar_chat(ruta_archivo)