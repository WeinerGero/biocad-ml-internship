from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from src.db.vector_db import VectorDB


class RAGPipeline:
    def __init__(self, db_path="./db", model_name="llama3"):
        # Инициализация векторной базы данных и LLM модели
        self.vector_db = VectorDB(persist_dir=db_path)
        self.llm = OllamaLLM(model=model_name, temperature=0.0)
        
        # Системный промт
        self.prompt_template = PromptTemplate.from_template(
            """
            ### SYSTEM INSTRUCTIONS:
            You are an expert biomedical research assistant at BIOCAD. 
            Your goal is to answer the researcher's question using ONLY the provided Context.
            
            ### RULES:
            1. Use ONLY the provided Context. Do not use outside knowledge.
            2. If the answer is not in the Context, state clearly that information is missing.
            3. Cite sources (PMID or Title) for every key fact.
            4. Keep the tone scientific and professional.
            
            ### CRITICAL OUTPUT CONSTRAINT:
            **YOU MUST ANSWER IN RUSSIAN LANGUAGE ONLY.** 
            Translate technical terms where appropriate, but keep specific protein names (like TREM2, APOE) in English if standard.

            ### CONTEXT:
            {context}

            ### QUESTION: 
            {question}

            ### ANSWER (in Russian):
            """
        )
        
    def _format_context(self, docs):
        """
        Форматирует контекст из найденных документов.
        
        :param docs list[Document]: Список найденных документов
        :return formated_text str: Отформатированный текст для LLM
        """
        formated_text = ""
        for doc in docs:
            pmid = doc.metadata.get("url", "N/A")
            year = doc.metadata.get("year", "N/A")
            formated_text += f"[Источник: https://pubmed.ncbi.nlm.nih.gov/{pmid}, Год: {year}]\n{doc.page_content}\n---\n"
        return formated_text
    
    def run(self, query: str, k: int=3):
        """
        Запускает RAG pipeline: поиск в векторной БД и генерация ответа.
        
        :param query str: Вопрос пользователя
        :param k int: Количество документов для извлечения
        :return answer str: Сгенерированный ответ
        """
        # Поиск релевантных документов
        docs_with_scores = self.vector_db.search(query, k=k)
        
        # Разделяем документы и оценки (scores)
        retrieved_docs = [doc for doc, _ in docs_with_scores]
        scores = [score for _, score in docs_with_scores]
        
        # Форматирование контекста для LLM
        context_text = self._format_context(retrieved_docs)
        
        # Заполняем шаблон вопросом и контекстом
        prompt = self.prompt_template.format(context=context_text, question=query)
        
        # Генерация ответа с помощью LLM
        response = self.llm.invoke(prompt)

        return {
            "query": query,
            "answer": response,
            "source_documents": retrieved_docs,
            "retrieval_scores": scores
        }

            
if __name__ == "__main__":
    pass