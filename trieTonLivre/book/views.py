from django.http import HttpResponse
from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from .serializers import BookShortSerializer, WordOccurrenceSerializer,BookSearchSerializer
from .models import Book, WordOccurrence
from .tasks import addBooks,getBookBySearch,closeness_centrality
# Create your views here.
def request_book(request):
    addBooks.delay()
    return HttpResponse("Hello, World!")

class BookViewSet(viewsets.ViewSet):
    renderer_classes = [JSONRenderer]
    def list(self, request):
        queryset = Book.objects.all()
        serializer = BookShortSerializer(queryset,many=True)
        return Response(serializer.data)
class WordOccurency(viewsets.ViewSet):
    renderer_classes = [JSONRenderer]
    def list(self, request):
        queryset = WordOccurrence.objects.all()
        serializer = WordOccurrenceSerializer(queryset,many=True)
        return Response(serializer.data)
    def search(self, request):
        search_words= request.query_params.get("word")
        
        print(search_words)
        queryset = getBookBySearch(search_words,"word")
        serializer = BookSearchSerializer(queryset,many=True)
        return Response(serializer.data)
def getBook():
    books = Book.objects.all()
    print(books)
    return HttpResponse(books)