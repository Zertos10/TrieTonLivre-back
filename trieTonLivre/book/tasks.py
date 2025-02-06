from celery import shared_task
from .models import Book, Author, WordOccurrence
import requests
from django.conf import settings
from django.utils.dateparse import parse_date
from django.db import transaction
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from django.db.models import Count,Sum

@shared_task
def addBooks():
    url = settings.GUTENDEX_API + "/books"
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
    try: 
        file = requests.get(book.linkToBook)
    except requests.RequestException as e:
        print(f"Error fetching book file: {e}")
        return
    
    WordOccurrence.objects.filter(book=book).delete()

    file= file.text
    pattern = "r'^[a-zA-Z]'$"
    tokenize_files =nltk.SpaceTokenizer().tokenize(file)
    stop_words = set(stopwords.words("english"))
    stop_words.update(stopwords.words("chinese"))
    pattern = re.compile("^[a-zA-Z\']+$")
    filter_file= [ word.lower() for word in tokenize_files if word.lower() not in stop_words and pattern.match(word)]
    print("Nombre de mots filtrés :", len(filter_file))
    words_counts = Counter(filter_file)
    existing_word = WordOccurrence.objects.filter(book=book, word__in=words_counts.keys())
    existing_dict = {w.word: w for w in existing_word}
    new_entries= []
    for word,count in words_counts.items():
        if word in existing_dict:
            existing_dict[word].count += count
        else:
            new_entries.append(WordOccurrence(book=book,word=word,count=count))
    with transaction.atomic():
        try :
            WordOccurrence.objects.bulk_update(existing_dict.values(),['count'])
            WordOccurrence.objects.bulk_create(new_entries)
        except Exception as e:
            print(e)
            raise e
    pass
@shared_task
def getBookBySearch(search):
    nltk.download('punkt_tab')
    token =nltk.SpaceTokenizer().tokenize(search)
    token = [word.lower() for word in token]
    bookSearch = (
    Book.objects
    .filter(wordoccurrence__word__in=token)  
    .annotate(
        matched_words=Count('wordoccurrence__word', distinct=True),  
        total_word_count=Sum('wordoccurrence__count')  
    )
    .filter(matched_words=len(token)) 
    .order_by('-total_word_count') 
)
    print(str(bookSearch.query))
    return bookSearch
@shared_task
def downloadBook(bookId):
    pass
