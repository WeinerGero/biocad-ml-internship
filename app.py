import asyncio
import streamlit as st

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
if prompt := st.chat_input("Введите ваш вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message("user", prompt)
    save_history()

    with st.chat_message("assistant", avatar="🧬"):
        # Плейсхолдер для стриминга текста
        response_placeholder = st.empty()
        
        # Контейнер-мост для данных между async и sync кодом
        stream_data = {
            "full_response": "",
            "sources": None
        }
        
        with st.status("Поиск статей...", expanded=True) as status:
            
            async def process_stream():
                # Асинхронный цикл для обработки стрима от RAGPipeline
                async for chunk in rag.astream_run(prompt, k=k_value):
                    # Обработка разных этапов генерации и обновление UI
                    if chunk["status"] == "searching":
                        st.write("🔍 Разделяю запрос:")
                        # Выводим стратегии поиска, которые RAGPipeline использует для поиска в базе
                        for s in chunk["strategies"]:
                            st.write(f"- {s}")
                        status.update(label="Поиск в базе PubMed...")
                        
                    # Когда начинается генерация ответа, мы получаем список уникальных статей и обновляем статус
                    elif chunk["status"] == "generating":
                        stream_data["sources"] = chunk["sources"]
                        st.write(f"📚 Найдено уникальных статей: {len(stream_data['sources'])}")
                        status.update(label="Ответ готовится...")
                        status.update(state="complete", expanded=False)
                     
                    # Во время генерации мы получаем чанки текста и обновляем плейсхолдер, добавляя каретку в конце для эффекта печати    
                    elif chunk["status"] == "streaming":
                        stream_data["full_response"] += chunk["answer_chunk"]
                        # Обновляем UI по мере поступления токенов
                        response_placeholder.markdown(stream_data["full_response"] + "▌")

            # Запуск асинхронного цикла
            try:
                asyncio.run(process_stream())
            except Exception as e:
                st.error(f"Ошибка при генерации: {e}")

        # Финальное отображение (убираем каретку ▌)
        response_placeholder.markdown(stream_data["full_response"])
        
        # Отрисовка источников
        if stream_data["sources"]:
            render_sources(stream_data["sources"])

        # Сохранение в историю сессии
        st.session_state.messages.append({
            "role": "assistant",
            "content": stream_data["full_response"],
            "sources": stream_data["sources"]
        })
        
        save_history()
