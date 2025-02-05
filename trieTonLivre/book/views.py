from django.http import HttpResponse
from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.response import Response
from .serializers import BookShortSerializer
from .models import Book
from .tasks import addBooks
# Create your views here.
def request_book(request):
    addBooks.delay()
    return HttpResponse("Hello, World!")

class BookViewSet(viewsets.ViewSet):
    def list(self, request):
        queryset = Book.objects.all()
        serializer = BookShortSerializer(queryset,many=True)
        return HttpResponse(serializer.data)  
 
def getBook():
    books = Book.objects.all()
    print(books)
    return HttpResponse(books)