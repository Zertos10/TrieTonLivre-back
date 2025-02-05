from celery import shared_task
from .models import Book, Author
import requests
import json
from django.conf import settings
from django.utils.dateparse import parse_date

@shared_task
def addBooks():
    url = settings.GUTENDEX_API + "/books"
    response = requests.get(url)
    book_data = response.json()
    for book in book_data.get("results", []):
        existing_book = Book.objects.filter(idGutendex=book["id"]).first()
        if not existing_book:
            addBook(book)

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
def preProcessBook(bookId):
    book = Book.objects.get(idGutendex=bookId)
    if not book:
        raise ""
    file=requests.get(book.linkToBook)
    
    
    
    pass

@shared_task
def downloadBook(bookId):
    pass
