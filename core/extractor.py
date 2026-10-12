# Actionable items, decision, questions

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os


def get_llm():
    return ChatGroq(model = "qwen/qwen3.8-27b", temperature=0.3, max_retries=6,      # waits and retries on 429
        max_tokens=400)


def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |
        ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}")
        ]) | llm | StrOutputParser()
    )


def extract_action_items(transcript : str) -> str:
    chain = build_chain(
        """
        You are an expert meeting analyst. From the meeting transcript, extract all action items. For each provide: \n
        - Task description \n
        - Owner ( who is responsible ) \n
        - Deadline (if mentioned, else write "not specified") \n
        Format as a numbered list, If none found say "action items not found."
        """
    )

    return chain.invoke(transcript)

def extract_key_decisions(transcript : str) -> str:
    chain = build_chain(
        """
    You are an expert meeting analyst. From the meeting transcript, extract all key decisions made.
    Format as a numbered list, If none found say "No key decisions found."
        """
    )

    return chain.invoke(transcript)


def extract_questions(transcript : str) -> str:
    chain = build_chain(
        """
        From the meeting transcript, extract all unresolved questions or topics needing follow-up.
        Format as a numbered list, If none found say "No open questions found."

        """
    )

    return chain.invoke(transcript)