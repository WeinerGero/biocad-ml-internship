import streamlit as st


def render_sidebar():
    """
    Отрисовывает боковую панель с настройками.
    Возвращает выбранное значение k.
    """
    with st.sidebar:
        st.header("Настройки поиска")
        
        # Настройка глубины поиска
        k_value = st.slider(
            "Количество статей (k)", 
            min_value=3, 
            max_value=20, 
            value=15,
            help="Сколько уникальных статей анализировать для ответа."
        )
        
        st.divider()
        
        # Кнопка очистки истории
        if st.button("Очистить диалог", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_sources = None
            st.rerun() # Перезагрузка страницы
            
        st.markdown("---")
        st.caption("Архитектура: Multi-Query -> Hybrid Search (BM25+Vector) -> RRF -> Mistral-Nemo")
        
    return k_value

def render_sources(docs):
    """
    Красиво отрисовывает список источников в сворачиваемом блоке.
    """
    if not docs:
        return

    # Группируем чанки по PMID, чтобы не выводить одну статью 3 раза
    unique_articles = {}
    for doc in docs:
        pmid = doc.metadata.get("pmid", "N/A")
        if pmid not in unique_articles:
            unique_articles[pmid] = {
                "year": doc.metadata.get("year", "N/A"),
                "text": doc.page_content
            }

    count = len(unique_articles)
    
    # Контейнер-экспандер
    with st.expander(f"Использовано источников: {count}", expanded=False):
        for pmid, data in unique_articles.items():
            year = data["year"]
            
            # Формируем ссылку на PubMed
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            
            st.markdown(f"**PMID: [{pmid}]({link}) ({year})**")
            st.caption(data["text"][:300] + "...") # Показываем начало текста
            st.divider()

def render_message(role, content, sources=None):
    """
    Отрисовывает одно сообщение чата.
    """
    # Определяем аватарки
    avatar = "🧬" if role == "assistant" else "👤"
    
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        
        # Если есть источники к этому сообщению - показываем их
        if sources:
            render_sources(sources)
