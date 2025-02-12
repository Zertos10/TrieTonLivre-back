import logging
import os

from celery import shared_task
from networkx import Graph
import numpy as np
import pandas as pd

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
    for book in book_data.get("results", []):
        existing_book = Book.objects.filter(idGutendex=book["id"]).first()
        if not existing_book:
            addBook(book)
        preProcessBook.delay(book)

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
            existing_dict[word].weight += weight
            existing_dict[word].count += file_word_count[word]
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
    books = Book.objects.filter(wordoccurrence__word__in=token).distinct().values_list('ids', flat=True)
    bookIds = list(books)
    if(len(bookIds) == 0):
        return []
    centrality =documentsClosenessCentrality(bookIds,documentGraph= documentGraph,keywords=token)
    print(centrality)
    sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    print(sorted_centrality)
    # Trier les livres par centralité décroissante
    sorted_books = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

# Extraire uniquement les IDs des livres triés
    sorted_book_ids = [book_id for book_id, _ in sorted_books]
    print(sorted_book_ids)
    books = Book.objects.filter(ids__in=sorted_book_ids)
    books_sorted = sorted(books, key=lambda book: centrality.get(book.ids, 0), reverse=True)

    return books_sorted

@shared_task
def frequencyWord(books: list[int], keywords: list[str]):
    frequency = (
        WordOccurrence.objects
        .filter(book__ids__in=books, word__in=keywords)
        .values('book__ids', 'word')  
        .annotate(total_weight=Sum('weight')) 
    )
    if not frequency:
        return np.zeros((len(books), 1))  # Retourner une matrice vide si aucune donnée
    df = pd.DataFrame(list(frequency))
    
    matrix = df.set_index('book__ids')['total_weight'].reindex(books, fill_value=0)
    return matrix.to_numpy().reshape(-1, 1)

#Permet d'avoir la fréquences des mots dans un corpus
def index_table(words:Counter) -> dict[str,float]:
    total_word = sum(words.values())
    frequencies = [count / total_word for count in words.values()]
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_frequencies = scaler.fit_transform([[freq] for freq in frequencies])
    return {word: normalized_frequencies[i][0] for i, word in enumerate(words.keys())}
@shared_task
def jaccard_similarity(documents_id:list[int], keywords: list[str]):
    if not documents_id or len(documents_id) == 1:
        return np.zeros((1,1))
    n = len(documents_id)
    similarity_matrix = np.zeros((n,n))
    word_weight_dicts = {}
    for doc_id in documents_id:
        words = WordOccurrence.objects.filter(book=Book.objects.get(ids=doc_id)).values_list('word', 'weight')
        word_weight_dicts[doc_id] = dict(words)
    for i,doc_i in enumerate(documents_id):
        for j,doc_j in enumerate(documents_id):
            if i <= j:
                words_i = word_weight_dicts[doc_i]
                words_j = word_weight_dicts[doc_j]

                intersection = set(words_i.keys()).intersection(words_j.keys()).intersection(keywords)
                union = set(words_i.keys()).union(words_j.keys()).union(keywords)
                
                jaccard_index = len(intersection) / len(union) if len(union) > 0 else 0
                similarity_matrix[i,j] = jaccard_index
                similarity_matrix[j,i] = jaccard_index
    df = pd.DataFrame(similarity_matrix, index=documents_id, columns=documents_id)
    return df.to_numpy()

def documentsClosenessCentrality(
    bookIds:list[int],
    documentGraph:Graph,
    keywords: list[str],
    threshold:float  = 0.2,
    alpha:float = 0.15,
    beta:float = 0.15,
    gamma:float= 0.7
    ):
    cosine = cosine_similaritys(bookIds)
    jaccard = jaccard_similarity(bookIds,keywords)
    frequency = frequencyWord(bookIds,keywords)

    documentGraph.add_nodes_from(bookIds)
    for i in range(len(bookIds)):
        for j in range(i +1,len(bookIds)):
            cos_sim = cosine[i, j]
            jac_sim = jaccard[i, j] *1000
            freq_sim = frequency[i,0]* 1000
            print(f'i :{i} j: {j}')
            print(f'cos_sim: {cos_sim}, jac_sim: {jac_sim}, freq_sim : {freq_sim}')
            weighted_similarity = alpha * cos_sim + beta * jac_sim + gamma * freq_sim
            if weighted_similarity > threshold:
                print(f'weighted_similarity: {weighted_similarity}')
                documentGraph.add_edge(bookIds[i], bookIds[j], weight=weighted_similarity)

    print(nx.is_connected(documentGraph))
    centrality = nx.closeness_centrality(documentGraph)
    return centrality

def cosine_similaritys(book_ids: list[int]):
    word_occurrences = defaultdict(dict)
    
    for book_id in book_ids:
        occurrences = WordOccurrence.objects.filter(book__ids=book_id)
        for occ in occurrences:
            word_occurrences[book_id][occ.word] = occ.weight
    
    documents = []
    for book_id in book_ids:
        document = " ".join([word * int(weight) for word, weight in word_occurrences[book_id].items()])
        documents.append(document)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
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