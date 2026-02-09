import streamlit as st
import time

from src.generator.control_generator import RAGPipeline

####### УБРАТЬ ЭТОТ БЛОК ПОСЛЕ СОЗДАНИЯ UI #######
try:
    from src.ui.state_manager import init_session_state
    from src.ui.components import render_sidebar, render_sources, render_message
except ImportError:
    # Заглушка, пока ты не создал файлы UI
    def init_session_state(): 
        if "messages" not in st.session_state: st.session_state.messages = []
    def render_sidebar(): return 15
    def render_sources(docs): st.write(docs)
    def render_message(role, content, sources=None):
        with st.chat_message(role):
            st.markdown(content)
            if sources: render_sources(sources)

# Конфигурация страницы Streamlit
st.set_page_config(
    page_title="BioCAD RAG Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
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
st.caption("Задайте вопрос о болезни Альцгеймера, и я найду ответы в последних научных статьях")

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
            "sources": result["source_documents"] # Сохраняем источники, чтобы они остались при перезагрузке
        })