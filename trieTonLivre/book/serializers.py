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
        
class BookSearchSerializer(serializers.ModelSerializer):
    author = serializers.SerializerMethodField()

    class Meta:
        model = Book
        fields = ["ids", "idGutendex", "title", "cover", "linkToBook", "downloadCount", "author"]

    def get_author(self, obj):
        return [author.name for author in obj.author.all()]
