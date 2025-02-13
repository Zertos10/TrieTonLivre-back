import logging
import math
import os
from celery import shared_task,chord,group
from django.conf import settings
import numpy as np
import requests
from django.db import transaction
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from django.utils.dateparse import parse_date



from ..models import Author, Book, WordOccurrence
from .utils import downloadBook

logger = logging.getLogger(__name__)
@shared_task
def addBooks(nbBook:int=50,maxBookByPage=32):
    project_root = Path(__file__).resolve().parent.parent.parent
    os.makedirs(os.path.join(project_root, "books"), exist_ok=True)
    pages=math.ceil(nbBook/maxBookByPage)
    logger.debug(f'Pages {pages}')
    parallel_books =group(getListBook.s(i+1) for i in range(pages))
    workflow = chord(parallel_books)(index_table.s())
    
@shared_task
def getListBook(iteration):
    textFiles = dict()
    logger.debug(f'Iteration : {iteration}')
    url = settings.GUTENDEX_API + '/books?languages=en&page=' + str(iteration)
    response = requests.get(url)
    book_data = response.json()
    for book in book_data.get("results", []):
        existing_book = Book.objects.filter(idGutendex=book["id"]).first()
        if not existing_book:
            existing_book=addBook(book)
        try:
            textFiles[existing_book.ids]=downloadBook(existing_book)
        except Exception as e:
            logger.error(f"Error downloading book {existing_book.idGutendex}: {e}")
        logger.info(len(textFiles))
    return textFiles
@shared_task
def addBook(book_json):
    book_instance, created = Book.objects.get_or_create(
        idGutendex=book_json["id"],
        defaults={
            'title': book_json.get("title"),
            'summary': book_json.get("summaries"),
            'cover': book_json.get("formats", {}).get("image/jpeg", ''),
            'linkToBook': book_json.get("formats", {}).get("text/plain; charset=us-ascii", ''),
            'downloadCount': book_json.get("download_count")
        }
    )
    for bookAuthor in book_json.get("authors"):
        author, created = Author.objects.get_or_create(
            name=bookAuthor.get("name"),
            defaults={
                'name': bookAuthor.get("name"),
                'birth_date': parse_date(book_json.get("birth_date")) if book_json.get("birth_date") else None,
                    'death_date': parse_date(book_json.get("death_date")) if book_json.get("death_date") else None
            }
        )
        book_instance.author.add(author)
    return book_instance
@shared_task
def index_table(booksText:list[dict[int,str]]|dict[int,str]) -> dict[str,float]:
    if isinstance(booksText, list):
        booksText = {k: v for d in booksText for k, v in d.items()}
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Transformer les textes en matrice TF-IDF
    matrix = vectorizer.fit_transform(booksText.values())
    logger.debug(f"Matrice TF-IDF : {matrix.shape}")
    terms = vectorizer.get_feature_names_out()  # Liste des mots indexés
        # Calculer la fréquence totale de chaque terme
    term_frequencies = np.asarray(matrix.sum(axis=0)).flatten()

    # Obtenir les indices des 1000 termes les plus fréquents
    top_term_indices = term_frequencies.argsort()[-5000:]
    top_terms = terms[top_term_indices]

    # Filtrer la matrice TF-IDF pour conserver uniquement les 1000 termes les plus fréquents
    filtered_matrix = matrix[:, top_term_indices]
    filtered_matrix_array = filtered_matrix.toarray()
    # cosine_similarity_matrix =cosine_similarity(filtered_matrix)
    # print(cosine_similarity_matrix.shape)
    addTerms = []
    book_ids = list(booksText.keys())  # Liste ordonnée des book_id
    for idx, book_id in enumerate(book_ids):
        book_vector = filtered_matrix_array[idx]  # Récupérer le vecteur TF  -IDF du livre
        
        for term_idx, term in enumerate(top_terms):
            term_frequency = book_vector[term_idx]  # Poids TF-IDF du terme
            
            if term_frequency > 0:
                if not WordOccurrence.objects.filter(book_id=book_id, term=term).exists():
                    addTerms.append(WordOccurrence(
                        book_id=book_id,
                        term=term,
                        term_frequency=term_frequency,
                        tfidf_weight=term_frequency
                    ))
                    logger.debug(f"Terme ajouté : {term} ({term_frequency})")
    if addTerms:
        with transaction.atomic():
            try:
                WordOccurrence.objects.bulk_create(addTerms)
                logger.info(f"Indexation terminée : {len(addTerms)} entrées ajoutées.")
            except Exception as e:
                logger.error(f"Erreur lors de la création de l'index : {e}")
                raise e
    else:
        logger.warning("Aucun terme pertinent à indexer.")

