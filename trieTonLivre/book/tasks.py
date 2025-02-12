import logging
import os

from celery import shared_task
from networkx import Graph
import numpy as np
import pandas as pd

from regex import F
from sklearn.preprocessing import MinMaxScaler
from urllib3 import Retry
from .models import Book, Author, WordOccurrence
import requests
from django.conf import settings
from django.utils.dateparse import parse_date
from django.db import transaction
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from requests.adapters import HTTPAdapter
from django.db.models import Sum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx


@shared_task
def addBooks():
    os.makedirs(os.path.join(settings.BASE_DIR, "books"), exist_ok=True)
    url = settings.GUTENDEX_API + "/books?languages=en"
    response = requests.get(url)
    book_data = response.json()
    textFiles = dict()
    for book in book_data.get("results", []):
        existing_book = Book.objects.filter(idGutendex=book["id"]).first()
        if not existing_book:
            existing_book=addBook(book)
        # preProcessBook.delay(book)
        try:
            textFiles[existing_book.ids]=downloadBook(existing_book)
        except Exception as e:
            logger.error(f"Error downloading book {existing_book.idGutendex}: {e}")
        logger.info(len(textFiles))
    index_table(textFiles)
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
def preProcessBook(bookS):
    book = Book.objects.filter(idGutendex=bookS.get("id")).first()
    if not book:
        raise ValueError("Book not found")
    WordOccurrence.objects.filter(book=book).delete()
    try:
        file= downloadBook(book)
    except Exception as e:
        logger.error(f"Error downloading book {book.idGutendex}: {e}")
        raise
    filter_word_file = preProcessText(file)
    logger.info(f"Nombre de mots après filtrage : {len(filter_word_file)}")
    file_word_count = Counter(filter_word_file)
    file_word = index_table(file_word_count)

    bulk_create_with_retry(file_word=file_word,file_word_count=file_word_count,book=book)
    pass
logger = logging.getLogger(__name__)
def preProcessText(texte:str):
    pattern = re.compile(r'^[a-zA-Z]+$')
    # sentences = nltk.sent_tokenize(file, language="english")
    tokenize_files = nltk.word_tokenize(texte)
    stop_words  = set(stopwords.words("english"))
    filter_word_file= [ word.lower() for word in tokenize_files 
    if word.lower() not in stop_words and pattern.match(word)]
    return filter_word_file
def bulk_create_with_retry(file_word:dict[str,float],file_word_count:dict[str,int],book, retries=5, delay=0.1,chunk_size=500):
    existing_word = WordOccurrence.objects.filter(book=book, word__in=file_word.keys())
    existing_dict = {w.word: w for w in existing_word}
    new_entries= []
    for word,weight in file_word.items():
        if word in existing_dict:
            existing_dict[word].tfidf_weight += weight
            existing_dict[word].term_frequency += file_word_count[word]
        else:
            new_entries.append(WordOccurrence(book=book,word=word,count=file_word_count[word],weight=weight))
    with transaction.atomic():
        try :
            WordOccurrence.objects.bulk_update(existing_dict.values(),['count','weight'])
            WordOccurrence.objects.bulk_create(new_entries)
        except Exception as e:
            print(e)
            raise e
@shared_task
def getBookBySearch(search:str) -> list[Book]:
    nltk.download('punkt_tab')
    documentGraph= nx.Graph()
    token =nltk.SpaceTokenizer().tokenize(search)
    token = [word.lower() for word in token]
    books = Book.objects.filter(wordoccurrence__term__in=token).distinct().values_list('ids', flat=True)
    bookIds = list(books)
    print(bookIds)
    if(len(bookIds) == 0):
        return []
    centrality =documentsClosenessCentrality(bookIds,documentGraph= documentGraph,keywords=token)
    print(centrality)
    sorted_book_ids = [book_id for book_id, _ in centrality] # Extract only the book IDs
    books_sorted = list(Book.objects.filter(ids__in=sorted_book_ids))
    books_sorted.sort(key=lambda book: sorted_book_ids.index(book.ids),reverse=True)

    return books_sorted

@shared_task
def frequencyWord(books: list[int], keywords: list[str]):
    frequency = (
        WordOccurrence.objects
        .filter(book__ids__in=books, word__in=keywords)
        .values('book__ids', 'term')  
        .annotate(total_weight=Sum('weight')) 
    )
    if not frequency:
        return np.zeros((len(books), 1))  # Retourner une matrice vide si aucune donnée
    df = pd.DataFrame(list(frequency))
    
    matrix = df.set_index('book__ids')['total_weight'].reindex(books, fill_value=0)
    return matrix.to_numpy().reshape(-1, 1)

#Permet d'avoir la fréquences des mots dans un corpus
def index_table(booksText:dict[int,str]) -> dict[str,float]:
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
    cosine_similaritys(booksText.keys())
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
@shared_task
def jaccard_similarity(documents_id:list[int], keywords: list[str]):
    if not documents_id or len(documents_id) == 1:
        return np.zeros((1, 1))

    n = len(documents_id)
    similarity_matrix = np.zeros((n, n))
    
    # Récupérer les poids des termes depuis WordOccurrence
    word_weight_dicts = {}
    for doc_id in documents_id:
        # Récupérer les termes et leurs poids associés pour chaque livre
        words = WordOccurrence.objects.filter(book__ids=doc_id).values_list('term', 'tfidf_weight')
        word_weight_dicts[doc_id] = dict(words)

    for i, doc_i in enumerate(documents_id):
        for j, doc_j in enumerate(documents_id):
            if i <= j:
                words_i = word_weight_dicts[doc_i]
                words_j = word_weight_dicts[doc_j]

                # Calculer l'intersection et l'union des termes des deux documents avec les mots-clés
                intersection = set(words_i.keys()).intersection(words_j.keys()).intersection(keywords)
                union = set(words_i.keys()).union(words_j.keys()).union(keywords)
                
                # Calculer l'indice de Jaccard
                jaccard_index = len(intersection) / len(union) if len(union) > 0 else 0
                similarity_matrix[i, j] = jaccard_index
                similarity_matrix[j, i] = jaccard_index

    # Convertir la matrice de similarité en DataFrame pour une meilleure lisibilité
    df = pd.DataFrame(similarity_matrix, index=documents_id, columns=documents_id)
    
    return df.to_numpy()

def documentsClosenessCentrality(
    bookIds:list[int],
    documentGraph:Graph,
    keywords: list[str],
    threshold:float  = 0.2,
    alpha:float = 0.5,
    beta:float = 0.5,
    gamma:float= 0.7
    ):
    cosine = cosine_similaritys(bookIds)
    jaccard = jaccard_similarity(bookIds,keywords)

    documentGraph.add_nodes_from(bookIds)
    for i in range(len(bookIds)):
        for j in range(i +1,len(bookIds)):
            cos_sim = cosine[i, j]
            jac_sim = jaccard[i, j]
            print(f'i :{i} j: {j}')
            weighted_similarity =  cos_sim
            print(f'cos_sim: {cos_sim} jaccard_sim: {jac_sim} weighted_similarity: {weighted_similarity}')
            if weighted_similarity > threshold:
                documentGraph.add_edge(bookIds[i], bookIds[j], weight=weighted_similarity)

    print(nx.is_connected(documentGraph))
    centrality = nx.closeness_centrality(documentGraph,distance="weight")
    
    return  sorted(centrality.items(), key=lambda item: item[1], reverse=True)
def cosine_similaritys(book_ids: list[int]):
    index_entries = WordOccurrence.objects.filter(book__ids__in=book_ids)

    book_tfidf = defaultdict(dict)

    for entry in index_entries:
        book_tfidf[entry.book.ids][entry.term] = entry.tfidf_weight

    all_terms = sorted(set(term for terms in book_tfidf.values() for term in terms))

    book_ids = sorted(book_tfidf.keys())  # Tri des ID des livres
    tfidf_matrix = np.zeros((len(book_ids), len(all_terms)))

    for i, book_id in enumerate(book_ids):
        for j, term in enumerate(all_terms):
            tfidf_matrix[i, j] = book_tfidf[book_id].get(term, 0) 

    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix
@shared_task
def downloadBook(book:Book)-> str:
    path_save=os.path.join(settings.BASE_DIR, "books")
    save_file = os.path.join(path_save, f"{book.idGutendex }.txt")
    if not os.path.exists(save_file):
        try: 
            session = requests.Session()
            retry = Retry(
                total=5,
                read=5,
                connect=5,
                backoff_factor=0.3,
                status_forcelist=(500, 502, 504)
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            response = session.get(book.linkToBook)
            response.raise_for_status()
            with open(save_file, "w", encoding='utf-8') as f:
                f.write(response.text)
            return response.text
        except requests.RequestException as e:
            raise e
    else:
        with open(save_file, "r",encoding='utf-8') as f:
            file = f.read()
            return file