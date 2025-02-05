from django.shortcuts import render
from .tasks import requestBooks
from django.http import HttpResponse

# Create your views here.
def test_request(request):
    response =requestBooks.delay()
    print(response.get())
    response_result = response.get()
    return HttpResponse(response_result)