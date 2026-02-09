import streamlit as st
import json
import os
from langchain_core.documents import Document

HISTORY_FILE = "chat_history.json"

def serialize_docs(docs):
    """Превращает объекты Document в словари для JSON."""
    if not docs:
        return None
    return [
        {"page_content": d.page_content, "metadata": d.metadata} 
        for d in docs
    ]

def deserialize_docs(doc_dicts):
    """Превращает словари обратно в объекты Document."""
    if not doc_dicts:
        return None
    return [
        Document(page_content=d["page_content"], metadata=d["metadata"]) 
        for d in doc_dicts
    ]

def save_history():
    """Сохраняет историю чата в файл."""
    # Создаем копию истории для сохранения (сериализуем документы)
    serializable_messages = []
    for msg in st.session_state.messages:
        msg_copy = msg.copy()
        if "sources" in msg_copy and msg_copy["sources"]:
            msg_copy["sources"] = serialize_docs(msg_copy["sources"])
        serializable_messages.append(msg_copy)
        
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable_messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Ошибка сохранения истории: {e}")

def load_history():
    """Загружает историю из файла при старте."""
    if not os.path.exists(HISTORY_FILE):
        return []
    
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Восстанавливаем объекты Document
        for msg in data:
            if "sources" in msg and msg["sources"]:
                msg["sources"] = deserialize_docs(msg["sources"])
        return data
    except Exception:
        return []

def init_session_state():
    """
    Инициализирует переменные сессии.
    Пытается загрузить историю с диска.
    """
    if "messages" not in st.session_state:
        # Сначала пробуем загрузить с диска
        saved_history = load_history()
        
        if saved_history:
            st.session_state.messages = saved_history
        else:
            # Если пусто, создаем приветствие
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Привет! Я научный ассистент BIOCAD. История диалога сохраняется локально."
            }]

    if "last_sources" not in st.session_state:
        st.session_state.last_sources = None