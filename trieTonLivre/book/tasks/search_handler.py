from celery import shared_task
import nltk
import networkx as nx
from .scoring_processing import documentsClosenessCentrality
from ..models import Book


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
    centrality =documentsClosenessCentrality(bookIds,documentGraph= documentGraph)
    print(centrality)
    sorted_book_ids = [book_id for book_id, _ in centrality] # Extract only the book IDs
    books_sorted = list(Book.objects.filter(ids__in=sorted_book_ids))
    books_sorted.sort(key=lambda book: sorted_book_ids.index(book.ids))

    return books_sorted