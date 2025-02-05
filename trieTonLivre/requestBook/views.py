from django.shortcuts import render
from .tasks import requestBooks, get_books_by_regex_task
from django.http import HttpResponse, JsonResponse
import re

from book.models import Book


# Create your views here.
def test_request(request):
    response = requestBooks.delay()
    print(response.get())
    response_result = response.get()
    return HttpResponse(response_result)


def get_books_by_regex_view(request):
    if 'regex' in request.GET:
        regex = request.GET['regex']
        response = get_books_by_regex_task.delay(regex)
        response_result = response.get()
        return JsonResponse(response_result, safe=False)
    else:
        return JsonResponse({'error': 'No regex provided'}, status=400)