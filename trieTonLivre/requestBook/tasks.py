import os
import re
from celery import shared_task
from django.conf import settings
import requests

@shared_task
def requestBooks():
    url = settings.GUTENDEX_API + "/books"
    response = requests.get(url)
    return response.json()

@shared_task
def get_books_by_regex_task(pattern):
    books_dir = os.path.join(os.path.dirname(__file__), '../assets')
    regex = re.compile(pattern)
    titles = []

    for filename in os.listdir(books_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(books_dir, filename), 'r', encoding='utf-8') as file:
                for line in file:
                    match = regex.search(line)
                    if match:
                        titles.append(line.strip())
    return titles