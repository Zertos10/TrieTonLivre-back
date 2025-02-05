from rest_framework import serializers
from .models import Book

class BookShortSerializer(serializers.ModelSerializer):
    class Meta:
        model = Book
        fields = ['idGutendex', 'title', 'cover','linkToBook', 'downloadCount']
