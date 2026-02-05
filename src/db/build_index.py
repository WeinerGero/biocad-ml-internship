import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from vector_db import VectorDB


def load_json():
    with open(
        "./data/alzheimer_articles.json",
        "r",
        encoding="utf-8",
        ) as f:
        return json.load(f)
    
def create_chunks(articles: list[dict]) -> list[Document]:
    """
    Нарезает статьи на чанки и возвращает список документов.
    
    :param articles: list[dict]: Список статей
    :return chunks: list[Document]: Список чанков в формате Document
    """
    # Настройки чанкинга
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    docs = []
    for article in articles:
        # Формирование полного текста статьи
        full_text = f"Title: {article['title']}\n\n" \
                    f"Abstract: {article['abstract']}\n\n" \
                    f"Introduction: {article.get('introduction', '')}\n\n" \
                    f"Conclusion: {article.get('conclusion', '')}"
    
         # Нарезка текста на чанки
        chunks = text_splitter.split_text(full_text)
        
        # Оборачиваем в Document с метаданными
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_index": i,
                    "pmid": article.get("pmid", ""),
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "year": article.get("year", "")
                }
            )
            docs.append(doc)
    return docs
    
    
if __name__ == "__main__":
    pass