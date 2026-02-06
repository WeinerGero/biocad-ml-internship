import os
import time
import json
from dotenv import load_dotenv

from PubMed_pasrer import Parser

# Загрузка переменных окружения из файла .env
load_dotenv()

LIMIT = 100


def start_parser(
    base_query: str= None,
    limit: int=LIMIT, 
    email: str=None,
    api_key: str=None
    ):
    """
    Запускает процесс парсинга статей из PubMed по заданному запросу.
    
    :param base_query str: Базовый поисковый запрос
    :param limit int: Максимальное количество статей для парсинга
    """
    # Формирование полного запроса с фильтром на свободный полный текст
    if base_query is None:
        base_query = "(Alzheimer's disease[Title/Abstract]) AND (therapeutic targets[Title/Abstract] OR drug targets[Title/Abstract])"
    filtered_query = f"{base_query} AND free full text[Filter]"
    
    # Получение email и API ключа из аргументов или переменных окружения
    if email is None:
        email = os.getenv("EMAIL")
    else:
        exit("Email must be provided either as argument or via EMAIL environment variable.")
    
    if api_key is None:
        api_key = os.getenv("NCBI_API_KEY")
    else:
        exit("API Key must be provided either as argument or via NCBI_API_KEY environment variable.")
    
    # Инициализация парсера и поиск статей
    parser = Parser(
        query=filtered_query,
        max_count=limit*5, # Запрашиваем больше, чтобы компенсировать отсеивание
        email=email,
        api_key=api_key
    )
    id_list = parser.search_pubmed_articles()
    
    final_dataset = []

    # Получение подробной информации о статьях
    if id_list:
        raw_articles = parser.fetch_details(id_list)

        for raw_record in raw_articles:
            if len(final_dataset) >= LIMIT:
                print(f"Limit of {LIMIT} articles reached. Stopping.")
                break
            
            # Парсинг (Заголовок, Абстракт)
            parsed_data = parser.parse_article(raw_record)
            
            if parsed_data:
                print(f"   Processing: {parsed_data['pmid']} - {parsed_data['title'][:30]}...")
                
                # Обогащение (Вытаскиваем Intro/Conclusion через PMC)
                enriched_data = parser.enrich_article_with_fulltext(parsed_data)
            
                intro = enriched_data.get('introduction')
                concl = enriched_data.get('conclusion')
                
                # Проверяем, что оба поля существуют и они не пустые строки
                if intro and concl:
                    print(f"      [+] Perfect Match! (Intro & Conclusion found). Saving.")
                    final_dataset.append(enriched_data)
                else:
                    # Логируем, чего не хватило
                    missing = []
                    if not intro: missing.append("Intro")
                    if not concl: missing.append("Conclusion")
                    print(f"      [-] Skipped. Missing sections: {', '.join(missing)}")
                                    
                time.sleep(0.5)
    else:
        print("No articles found.")
    
    # Сохранение результата в JSON файл
    output_file = 'data/alzheimer_articles.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)
        
    print(f"\nSaved {len(final_dataset)} articles to {output_file}")
    

if __name__ == "__main__":
    start_parser()