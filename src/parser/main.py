import time
import json
from PubMed_pasrer import Parser


LIMIT = 100
MAX_COUNT = 300

def main():
    base_query = "(Alzheimer's disease[Title/Abstract]) AND (therapeutic targets[Title/Abstract] OR drug targets[Title/Abstract])"
    filtered_query = f"{base_query} AND free full text[Filter]"
    
    # Инициализация парсера и поиск статей
    parser = Parser(query=filtered_query, max_count=MAX_COUNT)
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
    main()