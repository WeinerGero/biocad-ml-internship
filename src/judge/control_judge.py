import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings


class RAGJudge:
    def __init__(
        self,
        llm_model_name="llama3",
        embedding_model_name="pritamdeka/S-PubMedBert-MS-MARCO"
        ):
        """
        Судья для оценки качества ответов RAG агента. Использует LLM для анализа и HuggingFaceEmbeddings для оценки релевантности.
        
        :param llm_model_name str: Модель для качественной оценки
        :param embedding_model_name str: Модель для количественной оценки
        """
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        
        # LLM для качественной оценки
        self.judge_llm = OllamaLLM(model=self.llm_model_name, temperature=0)
        
        # Эмбеддинг модель для количественной оценки
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )
        
        # Промт для качественной оценки
        self.judge_template = """
        You are an impartial expert evaluator for a BioMedical RAG system.
        Your task is to evaluate the quality of the 'Generated Answer' based on the 'User Question' and the 'Ground Truth'.

        Evaluate based on these 4 metrics (Score 1 to 5):
        1. Relevance: Does the answer directly address the user's question? (5 = Direct answer, 1 = Irrelevant)
        2. Depth: Is the answer detailed and scientific? Does it explain mechanisms? (5 = Deep/Comprehensive, 1 = Superficial)
        3. Clarity: Is the language clear and professional in Russian? (5 = Perfect clarity, 1 = Confusing)
        4. Structure: Is the answer well-organized (intro, body, conclusion, citations)? (5 = Well structured, 1 = Messy)

        User Question: {question}
        Ground Truth (Expected Answer): {ground_truth}
        Generated Answer: {answer}

        OUTPUT FORMAT:
        Return ONLY a text block with the scores in this exact format:
        Relevance: <score>
        Depth: <score>
        Clarity: <score>
        Structure: <score>
        """
        self.judge_prompt = PromptTemplate.from_template(self.judge_template)
        
    def _get_llm_scores(self, 
            question: str, 
            answer: str, 
            ground_truth: str
        ) -> dict:
        """
        Получает оценки (1-5) от LLM судьи.
        
        :param question str: Вопрос пользователя
        :param answer str: Сгенерированный ответ
        :param ground_truth str: Ожидаемый ответ (Ground Truth)
        :return scores dict: Словарь с оценками по каждому метрику
        """
        try:
            prompt_text = self.judge_prompt.format(question=question, ground_truth=ground_truth, answer=answer)
            response = self.judge_llm.invoke(prompt_text)
            
            scores = {}
            for metric in ["Relevance", "Depth", "Clarity", "Structure"]:
                match = re.search(f"{metric}:\\s*(\\d+)", response)
                scores[metric] = int(match.group(1)) if match else 0
            return scores
        except Exception as e:
            print(f"Ошибка при оценке LLM: {e}")
            return {"Relevance": 0, "Depth": 0, "Clarity": 0, "Structure": 0}
        
    def _get_semantic_similarity(self,
            ground_truth: str, 
            generated_answer: str
        ) -> float:
        """
        Считает косинусное сходство между эталоном и генерацией.
        
        :param ground_truth str: Ожидаемый ответ (Ground Truth)
        :param generated_answer str: Сгенерированный ответ
        """
        try:
            vec_gt = self.embed_model.embed_query(ground_truth)
            vec_gen = self.embed_model.embed_query(generated_answer)
            
            similarity = cosine_similarity([vec_gt], [vec_gen])[0][0]
            return round(float(similarity), 4)
        except Exception as e:
            print(f"Ошибка при расчете семантического сходства: {e}")
            return 0.0

    def evaluate(self, 
            rag_output: dict,
            ground_truth: str
        ) -> dict:
        """
        Проводит полную оценку одного ответа от RAG.
        
        :param rag_output: Словарь, который вернул RAGPipeline.run().
        :param ground_truth: Эталонный ответ из Golden Set.
        """
        question = rag_output['query']
        generated_answer = rag_output['answer']
        
        # Считает векторное сходство между генерацией и эталоном
        similarity_score = self._get_semantic_similarity(ground_truth, generated_answer)
        
        # Считает качественные метрики
        llm_scores = self._get_llm_scores(question, generated_answer, ground_truth)
        
        # Собираем все в один результат
        evaluation_results = {
            "Question": question,
            "Generated Answer": generated_answer,
            "Ground Truth": ground_truth,
            "Sources (PMID)": [d.metadata.get('pmid') for d in rag_output['source_documents']],
            "Cosine Similarity": similarity_score,
            **llm_scores
        }
        return evaluation_results

    def run_evaluation_loop(self, 
            golden_set: list, 
            rag_pipeline
        ) -> pd.DataFrame:
        """
        Запускает полный цикл оценки по всему тестовому набору.
        
        :param golden_set: Список словарей [{'question': ..., 'ground_truth': ...}]
        :param rag_pipeline: Экземпляр твоего класса RAGPipeline
        :return: Pandas DataFrame с результатами.
        """
        results = []
        for i, item in enumerate(golden_set):
            print(f"   Evaluating case {i+1}/{len(golden_set)}...")
            rag_output = rag_pipeline.run(item['question'], k=3)
            eval_result = self.evaluate(rag_output, item['ground_truth'])
            results.append(eval_result)
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    pass