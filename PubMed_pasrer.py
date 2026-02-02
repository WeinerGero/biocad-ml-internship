import os
from Bio import Entrez


Entrez.email = os.getenv("EMAIL")
Entrez.api_key = os.getenv("NCBI_API_KEY")


def fetch_pubmed_abstracts(query, max_count=10):
    pass

