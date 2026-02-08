import re
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.db.vector_db import VectorDB


class RAGPipeline:
    def __init__(self, db_path="./db", model_name="llama3"):
        # Инициализация векторной базы данных и LLM модели
        self.vector_db = VectorDB(persist_dir=db_path)
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.1,
            num_ctx=4096,
            repeat_penalty=1.1, # Наказываем за повторения
            top_p=0.9,
            num_gpu=999, # Используем все доступные GPU
        )
        
        # Промт для перевода вопроса в поисковый запрос
        self.translate_prompt = PromptTemplate.from_template(
            """
            Task: You are a BioMedical search expert. Transform the user's conversational Russian question into a precise English keyword-rich query for a vector database.

            Rules:
            1. Extract key biological entities (proteins, genes, pathways).
            2. Focus on the core question.
            3. Output ONLY the optimized English query.

            Example:
            Russian Question: Слушай, а как наши бактерии в животе вообще могут влиять на то, что происходит в мозгу при Альцгеймере? Есть какие-то конкретные вещества, которые они выделяют, чтобы уменьшить вредный тау-белок?
            English Query: gut microbiota metabolites effect on tau protein reduction Alzheimer's disease
            ---

            Russian Question: {question}
            English Query:
            """
        )

        # Системный промт
        self.rag_prompt = PromptTemplate.from_template(
            """<|begin_of_text|>[INST]
            ### SYSTEM PROMPT ###
            You are a Senior BioMedical Analyst. Your task is to provide a structured scientific report in RUSSIAN, using ONLY the provided CONTEXT.

            **Critical Instructions:**
            1.  **Language:** Respond ONLY in RUSSIAN.
            2.  **Terminology:** Keep all biological terms (Genes like PLEC, Proteins like Amyloid-beta, Cells like Astrocytes) in ENGLISH. Do not invent Russian terms.
            3.  **Accuracy:** If the CONTEXT does not contain an answer, you MUST respond with: "В предоставленных статьях нет информации по этому вопросу".
            4.  **Structure:** Follow the response structure EXACTLY as shown below.
            5.  **SELF-CORRECTION:** Before answering, check if the retrieved CONTEXT specifically mentions the key entities from the USER QUESTION (e.g., "tau protein"). If it doesn't, start your answer with "В предоставленных статьях нет точной информации о...".

            **Response Structure:**
            **1. РЕЗЮМЕ**
            (A concise 2-3 sentence summary).
            **2. МОЛЕКУЛЯРНЫЙ МЕХАНИЗМ**
            (Describe the biological pathways).
            **3. БЕЗОПАСНОСТЬ И РИСКИ**
            (Information on side effects or a statement of its absence in the text).
            **4. ИСТОЧНИКИ**
            (List of [PMID: ...]).

            ---
            ### USER PROMPT ###

            **CONTEXT:**
            {context}

            **QUESTION:**
            {question}
            [/INST]
            """
            ) 
            
    def _format_context(self, docs):
        """
        Форматирует контекст из найденных документов.
        
        :param docs list[Document]: Список найденных документов
        :return formated_text str: Отформатированный текст для LLM
        """
        unique_articles = {}
        for doc in docs:
            pmid = doc.metadata.get('pmid', 'N/A')
            if pmid not in unique_articles:
                unique_articles[pmid] = {
                    "year": doc.metadata.get('year', 'N/A'),
                    "url": doc.metadata.get('url', 'N/A'),
                    "text": doc.page_content.strip()[:1500] # Берем меньше текста, но больше сути
                }
        
        formatted = []
        for pmid, data in unique_articles.items():
            # МЫ САМИ ГОТОВИМ СТРОКУ ИСТОЧНИКА ДЛЯ МОДЕЛИ
            citation_hint = f"[Источник: {data['url']}, Год: {data['year']}]"
            formatted.append(f"ARTICLE PMID: {pmid} | CITATION_STR: {citation_hint}\nCONTENT: {data['text']}")
        
        return "\n\n".join(formatted)
    
    def get_multi_query_docs(self, query: str, k: int = 5):
        # Промпт для генерации вариаций вопроса на английском
        mq_prompt = PromptTemplate.from_template(
            "Generate 3 different scientific search queries in English "
            "for the following Russian question to find research papers. "
            "Separate queries by new lines. Question: {question}"
        )
        
        # Генерируем вопросы
        chain = mq_prompt | self.llm | StrOutputParser()
        queries_raw = chain.invoke({"question": query})
        # Очищаем и создаем список запросов
        queries = [q.strip() for q in queries_raw.split("\n") if q.strip()]
        queries.append(query) # Добавляем оригинальный запрос
        
        # Собираем документы по всем запросам
        all_docs = []
        for q in queries:
            # Используем твой текущий векторный поиск
            docs_with_scores = self.vector_db.search(q, k=k)
            all_docs.extend([doc for doc, _ in docs_with_scores])
        
        # Дедупликация по PMID
        unique_docs = {}
        for doc in all_docs:
            pmid = doc.metadata.get('pmid')
            if pmid not in unique_docs:
                unique_docs[pmid] = doc
                
        return list(unique_docs.values())

        
    def run(self, query: str, k: int=10) -> dict:
        """
        Запускает RAG pipeline: поиск в векторной БД и генерация ответа.
        
        :param query str: Вопрос пользователя
        :param k int: Количество документов для извлечения
        :return answer str: Сгенерированный ответ
        """
        # Переводим вопрос в поисковый запрос на английском
        translation_prompt = self.translate_prompt.format(question=query)
        english_query = self.llm.invoke(translation_prompt).strip()

         # Чистим от возможной болтовни модели (если она скажет "Here is translation:")
        english_query = re.sub(r'^Here is.*?:\s*', '', english_query, flags=re.IGNORECASE).strip()
        english_query = english_query.replace('"', '') # Убираем кавычки
        
        # Поиск релевантных документов
        retrieved_docs = self.get_multi_query_docs(query, k=k)
        
        # Форматирование контекста для LLM
        context_text = self._format_context(retrieved_docs)
        
        # Заполняем шаблон вопросом и контекстом
        final_prompt = self.rag_prompt.format(context=context_text, question=query)
        
        # Генерация ответа с помощью LLM
        response = self.llm.invoke(final_prompt)

        return {
            "query": query,
            "translated_query": english_query,
            "answer": response,
            "source_documents": retrieved_docs,
        }

            
if __name__ == "__main__":
    pass