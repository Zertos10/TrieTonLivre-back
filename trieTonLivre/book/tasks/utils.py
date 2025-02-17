import logging
import os
from celery import shared_task
from django.conf import settings
import requests
from urllib3 import Retry
from ..models import Book
from requests.adapters import HTTPAdapter

logger = logging.getLogger("book.tasks.utils")

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