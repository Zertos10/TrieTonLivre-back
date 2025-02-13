from django.urls import path
from .views import BookViewSet, request_book, WordOccurency, fetch_book_text

book_list = BookViewSet.as_view({'get': 'list'})
book_detail = BookViewSet.as_view({'get': 'retrieve'})

urlpatterns = [
    path('', book_list, name='book-list'),
    path('<int:pk>/', book_detail, name='book-detail'),
    path('occurence', WordOccurency.as_view({'get': 'list'})),
    path('search', WordOccurency.as_view({'get': 'search'})),
    path("proxy-book/", fetch_book_text, name="proxy-book"),
    path("request", request_book),
]
