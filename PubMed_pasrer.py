import json
import os
import logging
from dotenv import load_dotenv

from Bio import Entrez


# Настройка логирования
logging.basicConfig(level=logging.INFO)

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

    def parse_article(self, article_data: dict):
        # Извлечение необходимых данных из статьи
        try:
            article = article_data['MedlineCitation']['Article']
            
            # Название статьи
            title = article.get('ArticleTitle', 'N/A')
            
            # Ссылка на статью в PubMed
            abstract_list = article.get('Abstract', {}).get('AbstractText', [])
            abstract = " ".join([str(part) for part in abstract_list]) if abstract_list else 'N/A'
            
            # Год публикации
            pub_date = article['Journal']['JournalIssue']['PubDate'].get('Year', 'N/A')
            
            # Ссылка
            pmid = article_data['MedlineCitation']['PMID']
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            
            if not abstract:
                return None # Пропустить статьи без абстракта
            
            return {
                "title": title,
                "abstract": abstract,
                "year": pub_date,
                "url": url,
                "pmid": str(pmid)
            }
        except KeyError as e:
            logging.error(f"KeyError: {e} in article ID {article.get('MedlineCitation', {}).get('PMID', 'Unknown')}")
            return None

if __name__ == "__main__":
    query = "(Alzheimer's disease[Title/Abstract]) AND (therapeutic targets[Title/Abstract] OR drug targets[Title/Abstract])"
    
    # Инициализация парсера и поиск статей
    parser = Parser(query=query, max_count=10)
    id_list = parser.search_pubmed_articles()
    
    # Получение подробной информации о статьях
    if id_list:
        articles  = parser.fetch_details(id_list)
        print("Count of article IDs fetched:", len(id_list))
        print("Article IDs:", id_list)
        print("Fetched records:", len(articles ))
    else:
        print("No articles found.")
        
    # Извлечение необходимых данных из статьи
    parsed_article = parser.parse_article(articles[0])
    print("Parsed article:", parsed_article)