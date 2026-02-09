import re
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from src.db.vector_db import VectorDB


class RAGPipeline:
    def __init__(self, db_path="./db", model_name="llama3"):
        # Инициализация векторной базы данных и LLM модели
        self.vector_db = VectorDB(persist_dir=db_path)
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.2,
            num_ctx=12_288,
            repeat_penalty=1.1, # Наказываем за повторения
            top_p=0.9,
            num_gpu=999, # Используем все доступные GPU
        )
        # Промпт для ПЕРЕВОДА запроса (RU -> EN Keywords)
        self.translate_prompt = PromptTemplate.from_template(
            """
            Role: BioMedical Search Expert.
            Task: Transform the user's Russian question into a Keyword-Rich English Search Query.
            
            STRATEGY:
            1.  **Extract Entities:** Pull out gene names (PLEC, TREM2), proteins, and diseases.
            2.  **Translate & Expand:** Convert Russian terms to medical English (e.g., "сердце" -> "heart cardiac").
            3.  **Keywords Only:** Output a string of keywords optimized for BM25 search.
            
            User Question: {question}
            
            Optimized English Query:
            """
        )
        
        # Системный промт
        self.rag_prompt = PromptTemplate.from_template(
            """<|begin_of_text|>[INST]
            ### SYSTEM ROLE ###
            You are a Senior BioMedical Analyst. Your goal is to answer the user's question in RUSSIAN based on the provided CONTEXT.

            ### CRITICAL INSTRUCTIONS ###
            1.  **Analyze Step-by-Step:** Before answering, look for keywords from the question in the context. Study each source carefully.
            2.  **Terminology:** Keep genes/proteins (e.g. TREM2, PILRA, AlphaFold) in ENGLISH.
            3.  **No Negative Hallucinations:** If the context mentions the specific proteins asked about (e.g. TREM2), USE THAT INFORMATION. Do not say "no info" if the keywords are present.
            4.  **Language:** Final output in RUSSIAN.
            5.  **SELF-CORRECTION:** Before answering, check if the retrieved CONTEXT specifically mentions the key entities from the USER QUESTION (e.g., "tau protein"). If it doesn't, start your answer with "В предоставленных статьях нет точной информации о...".
            6.  **Structure:** The answer must be well-organized (intro, body, conclusion, citations).
            
            ### RESPONSE STRUCTURE ###
            **1. РЕЗЮМЕ**
            (Direct answer. Name the specific receptors found).
            **2. ДЕТАЛЬНЫЙ АНАЛИЗ**
            (How were they found? Mention WES and AlphaFold if present).
            **3. БЕЗОПАСНОСТЬ**
            (Safety data).
            **4. ИСТОЧНИКИ**
            ([PMID: ...]).

            ---
            ### CONTEXT ###
            {context}

            ### USER QUESTION ###
            {question}
            [/INST]
            """
        )
        
        # Выгружаем ВСЕ документы из ChromaDB, чтобы построить индекс
        all_docs_data = self.vector_db.vector_store.get() 
        
        self.bm25_docs = []
        tokenized_corpus = []
        
        # Собираем документы и токенизируем их для BM25
        if all_docs_data['documents']:
            for i, text in enumerate(all_docs_data['documents']):
                metadata = all_docs_data['metadatas'][i] if all_docs_data['metadatas'] else {}
                # Создаем объект Document для удобства
                doc = Document(page_content=text, metadata=metadata)
                self.bm25_docs.append(doc)
                tokenized_corpus.append(text.lower().split())
            
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"BM25 Index built with {len(self.bm25_docs)} documents.")
        else:
            print("WARNING: VectorDB is empty, BM25 skipped.")
            self.bm25 = None
            
    def hybrid_search(self, query: str, k: int = 15):
        """
        Выполняет гибридный поиск (Vector + BM25) с агрегацией по PMID.
        Используется как поставщик кандидатов для глобального слияния.
        """
        # Очистка запроса от лишних символов
        clean_query = query.replace('"', '').replace("'", "").strip()
    
        if not self.bm25:
            vector_results = self.vector_db.search(clean_query, k=k)
            return [doc for doc, _ in vector_results]

        fetch_k = 100 
        vector_results = self.vector_db.search(clean_query, k=fetch_k)
        vector_docs = [doc for doc, _ in vector_results]

        tokenized_query = clean_query.lower().split()
        bm25_docs = self.bm25.get_top_n(tokenized_query, self.bm25_docs, n=fetch_k)

        rank_fusion = {}
        c = 60

        def add_to_fusion(docs_list):
            for rank, doc in enumerate(docs_list):
                pmid = str(doc.metadata.get('pmid', 'N/A'))
                if pmid == 'N/A': continue
                
                if pmid not in rank_fusion:
                    # Изменено: храним список docs вместо одного doc
                    rank_fusion[pmid] = {'docs': [], 'score': 0.0}
                
                # Добавляем чанк в список, если их меньше 3-х (самые релевантные)
                if len(rank_fusion[pmid]['docs']) < 3:
                    # Проверка на дубликаты контента внутри одной статьи
                    if doc.page_content not in [d.page_content for d in rank_fusion[pmid]['docs']]:
                        rank_fusion[pmid]['docs'].append(doc)
                
                rank_fusion[pmid]['score'] += 1.0 / (rank + c)

        add_to_fusion(vector_docs)
        add_to_fusion(bm25_docs)

        sorted_articles = sorted(
            rank_fusion.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        # Собираем все чанки из k лучших статей в один плоский список
        final_docs = []
        for item in sorted_articles[:k]:
            final_docs.extend(item['docs'])
        
        return final_docs
                
    def _format_context(self, docs):
        """
        Форматирует контекст из найденных документов.
        
        :param docs list[Document]: Список найденных документов
        :return formated_text str: Отформатированный текст для LLM
        """
        articles = {}
        for doc in docs:
            pmid = doc.metadata.get('pmid', 'N/A')
            if pmid not in articles:
                articles[pmid] = {
                    "year": doc.metadata.get('year', 'N/A'),
                    "chunks": []
                }
            articles[pmid]["chunks"].append(doc.page_content.strip())
        
        formatted = []
        for pmid, data in articles.items():
            # Склеиваем все чанки одной статьи
            full_content = "\n[ФРАГМЕНТ ТЕКСТА]: ".join(data['chunks'])
            formatted.append(f"ИСТОЧНИК [PMID: {pmid}, Год: {data['year']}]\nСОДЕРЖАНИЕ:\n{full_content}")
        
        return "\n\n" + "\n\n".join(formatted)
    
    def get_multi_query_docs(self, query: str, k: int = 7):
        # Промпт для генерации вариаций вопроса на английском
        mq_prompt = PromptTemplate.from_template(
            """
            Role: BioMedical Search Expert.
            Task: You have a user question in RUSSIAN. Generate 3 specific search queries in ENGLISH to find relevant scientific papers.
            
            STRATEGY FOR GENERATION:
            1. Query 1 (Direct): Literal translation of the user's question into scientific English.
            2. Query 2 (Expanded): Add synonyms and related terms. 
               - If asking about "mechanisms", add: "molecular pathway", "signaling", "interaction".
               - If asking about "genes/proteins", add: "expression levels", "isoforms", "accumulation".
            3. Query 3 (Specifics): Add potential specific details likely to be in the papers.
               - Keywords to include if relevant: "genetic variants", "SNPs", "polymorphisms", "tissue-specific", "mutations".
            
            INPUT (Russian): {question}
            
            OUTPUT (3 English queries, one per line, no numbering):
            """
        )
        
        # Генерируем вопросы
        chain = mq_prompt | self.llm | StrOutputParser()
        queries_raw = chain.invoke({"question": query})
        # Очищаем и создаем список запросов
        queries = [q.strip() for q in queries_raw.split("\n") if q.strip() and not q.startswith("Here")]
        # queries.append(query) # Добавляем оригинальный запрос
        
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

    def debug_hybrid_search(self, query: str, k: int = 10, target_pmid: str = "37349091"):
        """
        Универсальный дебаггер с широкой воронкой поиска.
        Реализует паттерн: Wide Retrieval -> Global Fusion -> Final Slice.
        """
        print(f"\n{'='*70}")
        print(f"СТАРТ ДЕБАГА (Wide Funnel Strategy): {query}")
        print(f"{'='*70}")

        # 1. Генерация стратегий (Универсальный промпт)
        mq_prompt = PromptTemplate.from_template(
            """
            Role: Senior Scientific Information Architect.
            Task: Break down the biological question into 3 search strategies with different zoom levels.
            
            STRATEGIES:
            1. MACRO: High-level overview (Disease name, general symptoms, affected organs).
            2. MICRO: Deep molecular level (Search for specific genes, proteins, SNPs, variants, alleles, and mutations).
            3. LINKAGE: Focus on connections (How X affects Y, signaling pathways, metabolic links, tissue-specific interactions).
            
            RULES:
            - Keep queries distinct. Do NOT repeat the same keywords in all three.
            - Preserve all technical codes and chemical symbols.
            - Output ONLY 3 English queries, one per line.
            
            User Question: {question}
            """
        )
        
        chain = mq_prompt | self.llm | StrOutputParser()
        raw_output = chain.invoke({"question": query}).strip()
        
        english_queries = []
        for line in raw_output.split("\n"):
            clean_line = re.sub(r'^\d+[\.\s\-/]+', '', line).strip()
            if len(clean_line) > 5:
                english_queries.append(clean_line)

        print(f"[*] Сгенерировано стратегий: {len(english_queries)}\n")

        # СЛОВАРЬ ДЛЯ ГЛОБАЛЬНОГО RRF (Суммируем веса всех чанков всех запросов)
        global_article_scores = {}
        c = 60 # Константа RRF
        
        # ПАРАМЕТР: Сколько брать из каждого запроса?
        # Твоя статья была на 50 месте, значит берем минимум 60.
        intermediate_k = 100

        # 2. Поиск по каждой стратегии
        for i, eng_q in enumerate(english_queries, 1):
            print(f"--- ЗАПРОС №{i}: '{eng_q}' ---")
            
            # Вызываем гибридный поиск, НО просим вернуть много результатов (intermediate_k)
            # Это не дает статье отсечься на раннем этапе
            candidate_docs = self.hybrid_search(eng_q, k=intermediate_k)
            
            for rank, doc in enumerate(candidate_docs):
                pmid = str(doc.metadata.get('pmid'))
                if pmid not in global_article_scores:
                    global_article_scores[pmid] = {'doc': doc, 'score': 0.0}
                
                # Накопительный эффект: статья может быть в результатах всех 3-х запросов
                # Чем выше она была в каждом из них, тем выше итоговый балл
                global_article_scores[pmid]['score'] += 1.0 / (rank + c)
            
            # Дебаг проверка наличия в текущем окне
            current_pmids = [str(d.metadata.get('pmid')) for d in candidate_docs]
            if target_pmid in current_pmids:
                print(f"  [+] Target {target_pmid} пойман в окно! Ранг в окне: {current_pmids.index(target_pmid)}")
            else:
                print(f"  [-] Target {target_pmid} не попал в топ-{intermediate_k}")
            print("\n")

        # 3. Финальная сортировка всего пула кандидатов
        sorted_global = sorted(
            global_article_scores.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        final_pmids = [str(item['doc'].metadata.get('pmid')) for item in sorted_global[:k]]

        print(f"{'='*70}")
        print(f"ФИНАЛЬНЫЙ КОНТЕКСТ ДЛЯ LLM (Топ-{k} из пула {len(global_article_scores)}):")
        print(final_pmids)

        if target_pmid in final_pmids:
            # Считаем итоговый ранг для отчета
            final_rank = final_pmids.index(target_pmid)
            print(f"\n[!!!] SUCCESS! Статья {target_pmid} вышла на {final_rank} место.")
        else:
            print(f"\n[XXX] FAILURE. Статья все еще за пределами топ-{k}.")
        print(f"{'='*70}\n")
    
    def run(self, query: str, k: int = 15) -> dict:
        """
        Запускает мульти-стратегический RAG pipeline:
        1. Генерация 3-х образов поиска (Macro, Micro, Relational).
        2. Гибридный поиск по каждой стратегии.
        3. Глобальная дедупликация и ранжирование (Global Fusion).
        4. Синтез ответа на основе расширенного контекста.
        """
        # 1. УНИВЕРСАЛЬНАЯ ГЕНЕРАЦИЯ СТРАТЕГИЙ (Zoom Strategy)
        mq_prompt = PromptTemplate.from_template(
            """
            Role: Senior Scientific Search Expert.
            Task: Break down the biological question into 3 search strategies with different zoom levels.
            
            STRATEGIES:
            1. MACRO: High-level overview (Disease, general symptoms, affected organs).
            2. MICRO: Deep molecular level (Specific genes, proteins, SNPs, variants, alleles, mutations).
            3. LINKAGE: Focus on connections (How X affects Y, pathways, tissue-specific interactions).
            
            RULES:
            - Output ONLY 3 English queries, one per line. No numbers, no intro.
            - Preserve all technical codes (rsID, Gene names) exactly.
            - DE-METAPHORIZE: Convert layman terms like "ломать чтение" or "портить" into scientific ones like "splicing defects", "transcription errors", "expression dysregulation".
            - Generate queries focused on RNA processing and protein-DNA interactions.

            User Question: {question}
            """
        )
        
        # Генерация англоязычных стратегий
        chain = mq_prompt | self.llm | StrOutputParser()
        raw_output = chain.invoke({"question": query})
        
        # Очистка и фильтрация строк
        english_queries = []
        for line in raw_output.split("\n"):
            clean_line = re.sub(r'^\d+[\.\s\-/]+', '', line).strip()
            if len(clean_line) > 5:
                english_queries.append(clean_line)
        
        # print(f"DEBUG: Generated Strategies for Search:\n" + "\n".join(english_queries))

        # 2. СБОР КАНДИДАТОВ (Wide Funnel)
        # Берем по 60 результатов на каждую стратегию, чтобы не потерять редкие статьи
        intermediate_k = 60
        global_article_scores = {}
        c = 60

        for eq in english_queries:
            candidates = self.hybrid_search(eq, k=intermediate_k)
            
            for rank, doc in enumerate(candidates):
                pmid = str(doc.metadata.get('pmid', 'N/A'))
                if pmid not in global_article_scores:
                    # Храним расширенный список фрагментов
                    global_article_scores[pmid] = {'docs': [], 'score': 0.0}
                
                if len(global_article_scores[pmid]['docs']) < 3:
                    if doc.page_content not in [d.page_content for d in global_article_scores[pmid]['docs']]:
                        global_article_scores[pmid]['docs'].append(doc)
                
                global_article_scores[pmid]['score'] += 1.0 / (rank + c)

        sorted_global = sorted(
            global_article_scores.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        # Извлекаем все чанки для топ-k статей
        retrieved_docs = []
        for item in sorted_global[:k]:
            retrieved_docs.extend(item['docs'])
        
        # ГЕНЕРАЦИЯ ОТВЕТА
        context_text = self._format_context(retrieved_docs)
        final_prompt = self.rag_prompt.format(context=context_text, question=query)
        response = self.llm.invoke(final_prompt)

        return {
            "query": query,
            "answer": response,
            "source_documents": retrieved_docs
        }

 
if __name__ == "__main__":
    pass