import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

def get_llm():
    return ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0.3,
        max_retries=6,      # waits and retries on 429
        max_tokens=400,
    )

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(transcript : str):
    vector_store = build_vector_store(transcript)

    retriever = get_retriever(vector_store)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
    You are an expert meeting assistant. Answer the user's questions based only on the meeting transcript context provided below.

    If the answer is not found in the context,  say : " I could not find this information in the meeting transcript ." 
    
    Always be concise and precise, if quoting someone , mention it properly.

    Context from meeting transcript : {context}
"""),
("human", "{question}")
    ])


    # Full LCEL RAG pipeline

    rag_chain = (
        {"context" : retriever | RunnableLambda(format_docs),
         "question" : RunnablePassthrough()
        } | prompt | llm | StrOutputParser()
    )

    return rag_chain



def load_rag_chain(transcript : str):
    vector_store = load_vector_store(transcript)

    retriever = get_retriever(vector_store)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
    You are an expert meeting assistant. Answer the user's questions based only on the meeting transcript context provided below.

    If the answer is not found in the context,  say : " I could not find this information in the meeting transcript ." 
    
    Always be concise and precise, if quoting someone , mention it properly.

    Context from meeting transcript : {context}
"""),
("human", "{question}")
    ])

    rag_chain = (
            {"context" : retriever | RunnableLambda(format_docs),
             "question" : RunnablePassthrough()
            } | prompt | llm | StrOutputParser()
        )
    
    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer : {answer}")
    return answer