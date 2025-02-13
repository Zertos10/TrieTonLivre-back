from rest_framework import serializers
from .models import Book, WordOccurrence

class BookShortSerializer(serializers.ModelSerializer):
    class Meta:
        model = Book
        fields = ['idGutendex', 'title', 'cover','linkToBook', 'downloadCount']
class WordOccurrenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = WordOccurrence
        fields = '__all__'
class AuthorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Author
        fields = ['name']
        
class BookSearchSerializer(serializers.ModelSerializer):
    author = serializers.SerializerMethodField()

    class Meta:
        model = Book
        fields = ["ids", "idGutendex", "title", "cover", "linkToBook", "downloadCount", "author"]
