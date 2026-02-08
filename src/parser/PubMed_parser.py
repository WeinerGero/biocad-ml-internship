import json
import os
import logging
import time
import re

from Bio import Entrez


# Настройка логирования
logging.basicConfig(level=logging.INFO)


class Parser():
    """
    Класс для парсинга статей из PubMed
    
    :param query: строка запроса для поиска статей
    :max_count: максимальное количество статей для поиска
    """
    def __init__(
        self, query=None,
        max_count=10,
        email=None,
        api_key=None
        ):
        # Проверка наличия запроса
        if query is None:
            raise ValueError("Query must be provided.")
        
        self.query = query
        self.max_count = max_count 
        
        # Установка email и API ключа для Entrez
        Entrez.email = email
        Entrez.api_key = api_key  
    
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
        
        # Удаляем ссылки вида [1], [12], [1-3], [1–3] (с разными тире)
        text = re.sub(r'\[\d+[–-]?\d*(?:,\s*\d+)*\]', '', text)
        
        # Удаляем лишние пробелы (двойные, табы, переносы)
        text = " ".join(text.split())
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    
    def get_pmcid_from_pmid(self, pmid):
        """
        Конвертирует PMID в PMCID для доступа к полному тексту
        
        :param pmid: строка с PMID статьи
        :return: строка с PMCID или None, если не найдено
        """
        try:
            handle = Entrez.elink(dbfrom="pubmed", db="pmc", linkname="pubmed_pmc", id=pmid)
            results = Entrez.read(handle)
            handle.close()
            
            # Ищем ID в ответе
            if results and results[0].get("LinkSetDb"):
                # Возвращает что-то вроде 'PMC9090123'
                # Нам нужно число для efetch, API часто принимает без префикса PMC, но надежнее id
                pmc_id = results[0]["LinkSetDb"][0]["Link"][0]["Id"]
                return pmc_id
            return None
        except Exception:
            return None
    
    def fetch_full_text(self, pmcid):
        """
        Скачивает XML полного текста из базы PMC
        
        :param pmcid: строка с PMCID статьи
        :return: XML данных статьи или None в случае ошибки
        """
        try:
            handle = Entrez.efetch(db="pmc", id=pmcid, retmode="xml")
            # PMC возвращает XML, который Bio.Entrez.read парсит в сложную структуру
            record = Entrez.read(handle, validate=False)
            handle.close()
            return record[0] # Обычно возвращается список из 1 статьи
        except Exception as e:
            logging.warning(f"Failed to fetch full text for {pmcid}: {e}")
            return None
    
    def extract_sections(self, pmc_xml_record):
        """
        Пытается найти Введение и Заключение в структуре PMC XML.
        
        :param pmc_xml_record: словарь с XML данными статьи из PMC
        :return: кортеж (введение, заключение) или (None, None
        """
        intro = []
        conclusion = []
        
        try:
            # Проверяем, есть ли тело статьи
            if 'body' not in pmc_xml_record:
                return None, None
            
            body = pmc_xml_record['body']
            sections = body.get('sec', [])
            
            for sec in sections:
                title = sec.get('title', '').lower()
                
                # Рекурсивная функция для извлечения текста из параграфов <p>
                def get_text_from_sec(section):
                    text_parts = []
                    # Текст может быть в 'p' (параграф) или вложенных 'sec'
                    if 'p' in section:
                        for p in section['p']:
                            # p может быть строкой или сложным объектом с форматированием
                            if isinstance(p, str):
                                text_parts.append(p)
                            elif hasattr(p, '__iter__'): # Если это MixedElement
                                text_parts.append("".join([str(x) for x in p if isinstance(x, str)]))
                    return " ".join(text_parts)

                content = get_text_from_sec(sec)

                # Эвристика для поиска разделов
                if 'intro' in title or 'background' in title:
                    intro.append(content)
                elif 'concl' in title or 'discussion' in title or 'future' in title:
                    conclusion.append(content)
            
            return " ".join(intro), " ".join(conclusion)

        except Exception as e:
            logging.warning(f"Error parsing sections: {e}")
            return None, None

    def enrich_article_with_fulltext(self, article_dict):
        """
        Главный метод: принимает словарь статьи (с абстрактом) и пытается добавить Intro/Conclusion
        
        :param article_dict: словарь с данными статьи
        :return: обновлённый словарь с введением и заключением, если найдены
        """
        pmid = article_dict.get('pmid')
        if not pmid:
            return article_dict
        
        # 1. Ищем PMCID
        pmcid = self.get_pmcid_from_pmid(pmid)
        
        if pmcid:
            # 2. Качаем XML
            xml_record = self.fetch_full_text(pmcid)
            if xml_record:
                # 3. Парсим
                intro, concl = self.extract_sections(xml_record)
                
                # Добавляем в словарь, если нашли. Если нет - оставляем пустым или N/A
                if intro:
                    article_dict['introduction'] = self._clean_text(intro)
                if concl:
                    article_dict['conclusion'] = self._clean_text(concl)
                    
        return article_dict


if __name__ == "__main__":
    base_query = "(Alzheimer's disease[Title/Abstract]) AND (therapeutic targets[Title/Abstract] OR drug targets[Title/Abstract])"
    filtered_query = f"{base_query} AND free full text[Filter]"
    
    # Инициализация парсера и поиск статей
    parser = Parser(query=filtered_query, max_count=10)
    id_list = parser.search_pubmed_articles()
    
    final_dataset = []

    
    # Получение подробной информации о статьях
    if id_list:
        raw_articles = parser.fetch_details(id_list)

        for raw_record in raw_articles:
            # Парсинг (Заголовок, Абстракт)
            parsed_data = parser.parse_article(raw_record)
            
            if parsed_data:
                logging.info(f"   Processing: {parsed_data['pmid']} - {parsed_data['title'][:30]}...")
                
                # Обогащение (Вытаскиваем Intro/Conclusion через PMC)
                enriched_data = parser.enrich_article_with_fulltext(parsed_data)
                
                # Логика: если Intro нашлось - отлично. Если нет - оставляем хотя бы абстракт.
                if enriched_data.get('introduction'):
                    logging.info(f"      [+] Full text sections found!")
                else:
                    logging.info(f"      [-] Only Abstract available (PMC parsing failed or structure differs).")
                
                final_dataset.append(enriched_data)
                
                time.sleep(0.5)
    else:
        logging.warning("No articles found.")
    
    # Сохранение результата в JSON файл
    output_file = 'data/alzheimer_articles.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)
        
    logging.info(f"\nSaved {len(final_dataset)} articles to {output_file}")