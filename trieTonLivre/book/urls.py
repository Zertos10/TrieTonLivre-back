from django.urls import path
from .views import BookViewSet,request_book,WordOccurency

urlpatterns = [
    path('', BookViewSet.as_view({'get': 'list'})),
    path('occurence', WordOccurency.as_view({'get': 'list'})),
    path('search', WordOccurency.as_view({'get': 'search'})),
    path("request",view=request_book),
]