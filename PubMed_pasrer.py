import json
import os
import logging
from dotenv import load_dotenv
import re

from Bio import Entrez


# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Загрузка переменных окружения из файла .env
load_dotenv()

# Установка email и API ключа для Entrez
Entrez.email = os.getenv("EMAIL")
Entrez.api_key = os.getenv("NCBI_API_KEY")


class Parser():
    """
    Класс для парсинга статей из PubMed
    
    :param query: строка запроса для поиска статей
    :max_count: максимальное количество статей для поиска
    """
    def __init__(self, query=None, max_count=10):
        # Проверка наличия запроса
        if query is None:
            raise ValueError("Query must be provided.")
        
        self.query = query
        self.max_count = max_count      
    
    def search_pubmed_articles(self) -> list[str]:
        """
        Поиск статей в PubMed по заданному запросу
        
        :return: список ID статей PubMed
        """
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
        """
        Получение подробной информации о статьях по списку ID
        
        :param id_list: список ID статей PubMed
        :return: список словарей с данными статей
        """
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
        """
        Парсинг данных о статье из PubMed
        
        :param article_data: словарь с данными статьи из PubMed
        :return: словарь с извлечёнными данными или None, если статья мусорная
        """
        try:
            # Извлечение корневого элемента
            cit = article_data.get('MedlineCitation', {})
            article = cit.get('Article', {})
            
            # Если нет заголовка или PMID - статья мусорная, пропускаем
            pmid = cit.get('PMID')
            if not pmid:
                return None

            # Заголовок
            title = self._clean_text(article.get('ArticleTitle', ''))

            # Обработка абстракта
            abstract_list = article.get('Abstract', {}).get('AbstractText', [])
            abstract_parts = []
            
            if abstract_list:
                for part in abstract_list:
                    # Проверяем, есть ли метка
                    if hasattr(part, 'attributes') and 'Label' in part.attributes:
                        label = part.attributes['Label']
                        text = str(part)
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(str(part))
                
                raw_abstract = " ".join(abstract_parts)
                abstract = self._clean_text(raw_abstract)
            else:
                abstract = ""

            # Если абстракт пустой - минусуем статью
            if not abstract:
                return None

            # Дата
            journal_issue = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
            year = journal_issue.get('Year')
            if not year:
                medline_date = journal_issue.get('MedlineDate', '')
                year_match = re.search(r'\d{4}', medline_date)
                year = year_match.group(0) if year_match else 'N/A'

            # Сборка результата
            return {
                "pmid": str(pmid),
                "title": title,
                "abstract": abstract,
                "year": year,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }

        except Exception as e:
            # Логируем ID, чтобы знать, на какой статье упало
            error_pmid = article_data.get('MedlineCitation', {}).get('PMID', 'Unknown')
            logging.error(f"Error parsing article {error_pmid}: {e}")
            return None


    @staticmethod
    def _clean_text(text):
        """
        Вспомогательная функция для очистки текста от HTML тегов и лишних пробелов
        
        :param text: входной текст
        :return: очищенный текст
        """
        if not text:
            return ""
        # Удаляем HTML теги
        text = re.sub(r'<[^<]+?>', '', text)
        # Удаляем лишние пробелы (двойные, табы, переносы)
        text = " ".join(text.split())
        return text
    
    
    
    
    

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