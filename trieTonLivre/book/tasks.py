from celery import shared_task
from .models import Book, Author, WordOccurrence
import requests
import os
from django.conf import settings
from django.utils.dateparse import parse_date
from django.db import transaction
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter,defaultdict
from django.db.models import Count,Sum

import networkx as nx


@shared_task
def addBooks():
    os.makedirs(os.path.join(settings.BASE_DIR, "books"), exist_ok=True)
    url = settings.GUTENDEX_API + "/books?languages=en,fr"
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
    file= downloadBook(book)
    
    pattern = "r'^[a-zA-Z]'$"
    sentences = nltk.sent_tokenize(file, language="english")
    newSentences = []
    for sentence in sentences:
        tokenize_files =nltk.SpaceTokenizer().tokenize(sentence)
        stop_words = set(stopwords.words("english"))
        pattern = re.compile(r'^[a-zA-Z]+$')
        filter_file= [ word.lower() for word in tokenize_files if word.lower() not in stop_words and pattern.match(word)]
        newSentences.append(filter_file)
    result =closeness_centrality(newSentences)
    
    print("Nombre de mots filtrés :", len(result))
    existing_word = WordOccurrence.objects.filter(book=book, word__in=result.keys())
    existing_dict = {w.word: w for w in existing_word}
    new_entries= []
    counter = countWords(newSentences)
    for word,weight in result.items():
        if word in existing_dict:
            existing_dict[word].count += counter[word]
            existing_dict[word].weight += weight
        else:
            new_entries.append(WordOccurrence(book=book,word=word,count=counter[word],weight=weight))
            existing_dict[word] = new_entries[-1]
    objects_to_update = [obj for obj in existing_dict.values() if obj.pk is not None]
    with transaction.atomic():
        try :
            WordOccurrence.objects.bulk_update(objects_to_update,['count','weight'])
            WordOccurrence.objects.bulk_create(new_entries)
        except Exception as e:
            print(e)
            raise e
    pass
@shared_task
def getBookBySearch(search:str,typeSearch:str) -> list[Book]:
    nltk.download('punkt_tab')
    token =nltk.SpaceTokenizer().tokenize(search)
    token = [word.lower() for word in token]
    filtredBook=sortByOccurenceWord(token)
    return sortByJaccardGraph(filtredBook,token)

@shared_task
def sortByOccurenceWord(keywords: list[str]) -> list[Book]:
     return (Book.objects
    .all()  
    .annotate(
        matched_words=Count('wordoccurrence__word', distinct=True),  
        total_word_count=Sum('wordoccurrence__count')  
    )
    ).order_by('-total_word_count')
#Permet de mesurer la similarité entre les livres et les mots clés
@shared_task
def sortByJaccardGraph(results: list[Book], keywords: list[str]) -> list[Book]:
    print(results)
    if not results or len(results) == 1:
        return results
    newResult= []
    
    for result in results:
        words = WordOccurrence.objects.filter(book=result).values_list('word', 'weight') 
        word_weight_dict = dict(words) 
        
        intersection = set(word_weight_dict.keys()).intersection(keywords)
        union = set(word_weight_dict.keys()).union(keywords)
        intesection_weighted = sum(word_weight_dict.get(word,0) for word in intersection)
        union_weighted = sum(word_weight_dict.get(word,0) for word in union)
        
        jaccard_index = intesection_weighted / union_weighted if union_weighted > 0 else 0
        print(jaccard_index)
        newResult.append((result, jaccard_index))
    newResult.sort(key=lambda x: x[1], reverse=True)
    return [result for result, _ in newResult]

def countWords(words: list[list[str]]) -> dict[str,int]:
    word_counter = Counter()
    for sentence in words:
        word_counter.update(sentence)
    return word_counter
    
@shared_task
def closeness_centrality(text:list[list[str]]) -> dict[str,float]:
    bookGraph = nx.Graph()
    co_occurence = defaultdict(int)
    for sentence in text:
        for i, word1 in enumerate(sentence):
            for j, word2 in enumerate(sentence):
                if i < j:
                    co_occurence[(word1, word2)] += 1
    for (word1, word2), weight in co_occurence.items():
        bookGraph.add_edge(word1, word2, weight=weight)
    result=nx.closeness_centrality(bookGraph)
    return result

@shared_task
def downloadBook(bookId:Book)-> str:
    path_save=os.path.join(settings.BASE_DIR, "books")
    save_file = os.path.join(path_save, f"{bookId.idGutendex}.txt")
    if not os.path.exists(save_file):
        try: 
            file = requests.get(bookId.linkToBook)
            with open(save_file, "w",encoding='utf-8') as f:
                f.write(file.text)
            return file.text
        except requests.RequestException as e:
            print(f"Error fetching book file: {e}")
            return
    else:
        with open(save_file, "r",encoding='utf-8') as f:
            file = f.read()
            return file
   
