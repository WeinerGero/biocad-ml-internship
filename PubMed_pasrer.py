import json
import os
from Bio import Entrez
from dotenv import load_dotenv


# Загрузка переменных окружения из файла .env
load_dotenv()

# Установка email и API ключа для Entrez
Entrez.email = os.getenv("EMAIL")
Entrez.api_key = os.getenv("NCBI_API_KEY")


class Parser():
    """Класс для парсинга статей из PubMed."""
    def __init__(self, query=None, max_count=10):
        # Проверка наличия запроса
        if query is None:
            raise ValueError("Query must be provided.")
        
        self.query = query
        self.max_count = max_count      
    
    def search_pubmed_articles(self) -> list[str]:
        # Поиск статей в PubMed по заданному запросу
        handle = Entrez.esearch(
            db="pubmed", 
            term=self.query, 
            retmax=self.max_count,
            sort="relevance",
        )
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]

    def fetch_details(self, id_list: list[str]):
        # Получение подробной информации о статьях по списку ID
        ids = ",".join(id_list)
        handle = Entrez.efetch(
            db="pubmed",
            id=ids,
            retmode="xml",
        )
        records = Entrez.read(handle)
        handle.close()
        articles_list = records.get('PubmedArticle', [])
        return articles_list

if __name__ == "__main__":
    query = "(Alzheimer's disease[Title/Abstract]) AND (therapeutic targets[Title/Abstract] OR drug targets[Title/Abstract])"
    
    parser = Parser(query=query, max_count=10)
    id_list = parser.search_pubmed_articles()
    
    if id_list:
        articles  = parser.fetch_details(id_list)
        print("Count of article IDs fetched:", len(id_list))
        print("Article IDs:", id_list)
        print("Fetched records:", len(articles ))
        
        for i, article in enumerate(articles[:2]):
            title = article['MedlineCitation']['Article']['ArticleTitle']
            print(f"{i+1}. {title}")
            
        print("Keys in the first article:", list(articles[0].keys()) if articles else "No articles to show keys for")
    else:
        print("No articles found.")
        
