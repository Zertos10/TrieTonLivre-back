from collections import defaultdict

from networkx import Graph
import networkx as nx
import numpy as np
import pandas as pd
from django.db.models import Q
from ..models import WordOccurrence
from sklearn.metrics.pairwise import cosine_similarity
import logging
logger = logging.getLogger(__name__)
def cosine_similaritys(book_ids: list[int]):
    index_entries = WordOccurrence.objects.filter(Q(book__ids__in=book_ids))
    logging.info(f'index_entries: {index_entries}')
    book_tfidf = defaultdict(dict)

    for entry in index_entries:
        book_tfidf[entry.book.ids][entry.term] = entry.tfidf_weight

    all_terms = sorted(set(term for terms in book_tfidf.values() for term in terms))
    
    book_ids = sorted(book_tfidf.keys())  # Tri des ID des livres
    tfidf_matrix = np.zeros((len(book_ids), len(all_terms)))

    for i, book_id in enumerate(book_ids):
        for j, term in enumerate(all_terms):
            tfidf_matrix[i, j] = book_tfidf[book_id].get(term, 0)
    logging.info(f'tfidf_matrix: {tfidf_matrix}')
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix
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

    # Convertir la matrice de similarité en DataFrame pour une meilleure lisibilité
    df = pd.DataFrame(similarity_matrix, index=documents_id, columns=documents_id)
    
    return df.to_numpy()
def documentsClosenessCentrality(
    bookIds:list[int],
    documentGraph:Graph,
    ):
    cosine = cosine_similaritys(bookIds)
    logger.info(f'cosine: {cosine}')
    documentGraph.add_nodes_from(bookIds)
    for i in range(len(bookIds)):
        for j in range(i + 1, len(bookIds)):
            if i < cosine.shape[0] and j < cosine.shape[1]:
                cos_sim = cosine[i, j]
                print(f'i: {i}, j: {j}')
                print(f'cos_sim: {cos_sim}')
                documentGraph.add_edge(bookIds[i], bookIds[j], weight=cos_sim)

    print(nx.is_connected(documentGraph))
    centrality = nx.closeness_centrality(documentGraph,distance="weight")
    
    return centrality.items()