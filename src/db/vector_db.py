import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class VectorDB:
    """Векторная база данных для хранения и поиска эмбеддингов"""
    def __init__(self, collection_name="alzheimer_docs", persist_dir="./db"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Эмбеддинг модель для медицинских текстов
        self.embedding_model = HuggingFaceEmbeddings(
                model_name="pritamdeka/S-PubMedBert-MS-MARCO",
                model_kwargs={'device': 'cpu'}, 
                encode_kwargs={'normalize_embeddings': True} 
            )
        
        # Векторное хранилище Chroma
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=persist_dir
        )
        
    def add_documents(self, batch_size: int=100, documents: list[Document]=None):
        """
        Добавляет чанки в векторную базу данных пакетами.
        
        :param batch_size int: Размер пакета для добавления
        :param documents list[Document]: Список документов для добавления
        """
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            self.vector_store.add_documents(batch)
            
    def search(self, query: str, k: int=3):
        """
        Выполняет поиск по векторной базе данных.
        
        :param query str: Запрос для поиска
        :param k int: Количество возвращаемых результатов
        :return list[Document]: Список найденных документов
        """
        return self.vector_store.similarity_search_with_score(query, k=k)


if __name__ == "__main__":
    VectorDB()