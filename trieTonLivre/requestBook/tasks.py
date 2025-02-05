from celery import shared_task
import requests
from django.conf import settings

"""_summary_

Returns:
    _type_: _description_
"""
@shared_task
def requestBooks():
    url = settings.GUTENDEX_API+"/books"
    response = requests.get(url)
    return response.json()