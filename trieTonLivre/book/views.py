import requests
from django.http import HttpResponse, JsonResponse
from rest_framework import viewsets, pagination
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializers import BookShortSerializer, WordOccurrenceSerializer, BookSearchSerializer
from .models import Book, WordOccurrence
from .tasks import addBooks, getBookBySearch
import Levenshtein


# Proxy pour récupérer le contenu d'un livre et éviter les erreurs CORS
def fetch_book_text(request):
    url = request.GET.get("url")
    print("URL reçue :", url)  

    if not url:
        return JsonResponse({"error": "Missing URL parameter"}, status=400)

    try:
        response = requests.get(url)
        response.raise_for_status()
        return JsonResponse({"content": response.text})
    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": str(e)}, status=500)


# Vue pour déclencher une tâche asynchrone d'ajout de livre
def request_book(request):
    addBooks.delay()
    return HttpResponse("Hello, World!")


# Ajout d'une pagination personnalisée
class BookPagination(pagination.PageNumberPagination):
    page_size = 8  # Nombre de livres par défaut par page
    page_size_query_param = "limit"  # Permet d'ajuster via ?limit=xx


class BookViewSet(viewsets.ViewSet):
    renderer_classes = [JSONRenderer]
    pagination_class = BookPagination  # Intégration de la pagination

    def list(self, request):
        queryset = Book.objects.all()
        paginator = self.pagination_class()
        result_page = paginator.paginate_queryset(queryset, request)
        serializer = BookShortSerializer(result_page, many=True)
        return paginator.get_paginated_response(serializer.data)

    def retrieve(self, request, pk=None):
        try:
            book = Book.objects.get(pk=pk)
            serializer = BookSearchSerializer(book)
            return Response(serializer.data)
        except Book.DoesNotExist:
            return Response({"error": "Book not found"}, status=404)

class WordOccurency(viewsets.ViewSet):
    renderer_classes = [JSONRenderer]

    def list(self, request):
        queryset = WordOccurrence.objects.all()
        serializer = WordOccurrenceSerializer(queryset, many=True)
        return Response(serializer.data)

    def search(self, request):
        search_words = request.query_params.get("word")
        print("Mot recherché :", search_words)

        queryset = getBookBySearch(search_words)
        
        # Appliquer la pagination
        paginator = BookPagination()
        result_page = paginator.paginate_queryset(queryset, request)
        serializer = BookSearchSerializer(result_page, many=True)
        return paginator.get_paginated_response(serializer.data)
import requests
from django.http import HttpResponse, JsonResponse
from rest_framework import viewsets, pagination
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializers import BookShortSerializer, WordOccurrenceSerializer, BookSearchSerializer
from .models import Book, WordOccurrence
from .tasks import addBooks, getBookBySearch


# Proxy pour récupérer le contenu d'un livre et éviter les erreurs CORS
def fetch_book_text(request):
    url = request.GET.get("url")
    print("URL reçue :", url)  

    if not url:
        return JsonResponse({"error": "Missing URL parameter"}, status=400)

    try:
        response = requests.get(url)
        response.raise_for_status()
        return JsonResponse({"content": response.text})
    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": str(e)}, status=500)


# Vue pour déclencher une tâche asynchrone d'ajout de livre
def request_book(request):
    addBooks.delay()
    return HttpResponse("Hello, World!")


# Ajout d'une pagination personnalisée
class BookPagination(pagination.PageNumberPagination):
    page_size = 8  # Nombre de livres par défaut par page
    page_size_query_param = "limit"  # Permet d'ajuster via ?limit=xx


class BookViewSet(viewsets.ViewSet):
    renderer_classes = [JSONRenderer]
    pagination_class = BookPagination  # Intégration de la pagination

    def list(self, request):
        queryset = Book.objects.all()
        paginator = self.pagination_class()
        result_page = paginator.paginate_queryset(queryset, request)
        serializer = BookShortSerializer(result_page, many=True)
        return paginator.get_paginated_response(serializer.data)

    def retrieve(self, request, pk=None):
        try:
            book = Book.objects.get(pk=pk)
            serializer = BookSearchSerializer(book)
            return Response(serializer.data)
        except Book.DoesNotExist:
            return Response({"error": "Book not found"}, status=404)


class WordOccurency(viewsets.ViewSet):
    renderer_classes = [JSONRenderer]

    def list(self, request):
        queryset = WordOccurrence.objects.all()
        serializer = WordOccurrenceSerializer(queryset, many=True)
        return Response(serializer.data)

    def search(self, request):
        search_words = request.query_params.get("word")
        print("Mot recherché :", search_words)

        queryset = getBookBySearch(search_words)
        
        # Appliquer la pagination
        paginator = BookPagination()
        result_page = paginator.paginate_queryset(queryset, request)
        serializer = BookSearchSerializer(result_page, many=True)
        return paginator.get_paginated_response(serializer.data)


# Fonction de récupération des livres (non utilisée, peut être supprimée)
def getBook():
    books = Book.objects.all()
    print(books)
    return HttpResponse(books)
