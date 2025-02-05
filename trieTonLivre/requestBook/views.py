from django.shortcuts import render
from .tasks import requestBooks
from django.http import HttpResponse, JsonResponse
import re

from book.models import Book

# Create your views here.
def test_request(request):
    response =requestBooks.delay()
    print(response.get())
    response_result = response.get()
    return HttpResponse(response_result)

def advanced_search(request):
    regex = request.GET.get('regex', '')
    if not regex:
        return JsonResponse({'error': 'No regex provided'}, status=400)

    try:
        pattern = re.compile(regex)
    except re.error:
        return JsonResponse({'error': 'Invalid regex'}, status=400)

    matching_books = Book.objects.filter(summary__regex=pattern)
    results = [{'id': book.id, 'title': book.title, 'author': book.author, 'summary': book.summary} for book in matching_books]

    return JsonResponse(results, safe=False)