from celery import shared_task
import nltk
import networkx as nx
from .scoring_processing import documentsClosenessCentrality
from ..models import Book
from .graph_manager import get_graph


@shared_task
def getBookBySearch(search:str) -> list[Book]:
    token =nltk.SpaceTokenizer().tokenize(search)
    token = [word.lower() for word in token]
    graph =get_graph()    
    nx.closeness_centrality(graph)
    books = Book.objects.filter(wordoccurrence__term__in=token).distinct().order_by('-score')
    if(len(books) == 0):
        return []
    print({f'{book.ids} : {book.score}' for book in books})
    return books