import streamlit as st
import time

from src.generator.control_generator import RAGPipeline
from src.ui.state_manager import init_session_state, save_history
from src.ui.components import render_sidebar, render_sources, render_message


# Конфигурация страницы Streamlit
st.set_page_config(
    page_title="BioCAD RAG Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стилизация для оптимальной читаемости (макс. ширина, скругление сообщений)
st.markdown(
    """
    <style>
    .block-container {
        max_width: 850px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        margin: auto; 
    }
    
    .stChatMessage {
        border-radius: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Кешируем загрузку RAGPipeline, с помощью cache_resource,
# чтобы не перезагружать модель при каждом взаимодействии
@st.cache_resource(show_spinner="Загрузка нейросети и базы знаний...")
def load_rag_pipeline():
    """
    Инициализирует RAG один раз.
    Использует кэш Streamlit, чтобы не перезагружать модель при каждом клике.
    """
    return RAGPipeline(db_path="./db", model_name="mistral-nemo")

try:
    rag = load_rag_pipeline()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

# Инициализация состояния сессии для хранения истории сообщений
init_session_state()

# Слайдер для настройки количества возвращаемых документов (k) в поиске
k_value = render_sidebar()

# Интерфейс чата
st.title("АА – Альцгеймер-Ассистент")
st.caption("Поможет вспомнить статью, которую вы читали на днях")

# Отрисовка истории (чтобы сообщения не пропадали)
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg.get("sources"))

# Обработка нового ввода пользователя
if prompt := st.chat_input("Введите ваш вопрос об исследовании..."):
    # Отображаем вопрос пользователя сразу
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message("user", prompt)

    # Генерация ответа
    with st.chat_message("assistant"):
        # Контейнер статуса показывает, что происходит "под капотом"
        with st.status("Анализ научных статей...", expanded=True) as status:
            
            st.write("Генерация поисковых стратегий...")
            start_time = time.time()
            
            # Запуск RAG
            result = rag.run(prompt, k=k_value)
            
            duration = time.time() - start_time
            st.write(f"Поиск и генерация заняли: {duration:.2f} сек")
            
            # Показываем, что нашли (для прозрачности)
            if "strategies" in result:
                st.info(f"Стратегии поиска:\n" + "\n".join([f"- {s}" for s in result["strategies"]]))
            
            status.update(label="Готово!", state="complete", expanded=False)

        # Вывод ответа
        st.markdown(result["answer"])
        
        # Вывод источников
        if result.get("source_documents"):
            render_sources(result["source_documents"])

        # Сохранение в историю
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
        })
        
        save_history() 