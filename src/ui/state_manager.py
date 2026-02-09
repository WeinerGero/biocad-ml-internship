import streamlit as st


def init_session_state():
    """
    Инициализирует переменные сессии при первом запуске.
    """
    # История сообщений
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
        # Приветственное сообщение от ассистента
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Привет! Я научный ассистент BIOCAD. Я помогу найти информацию о болезни Альцгеймера, используя гибридный поиск по научным статьям. О чем вы хотите узнать?"
        })

    # Хранилище для найденных источников (чтобы не пропадали при ререндере)
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = None
