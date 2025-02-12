from django.contrib import admin

from .models import Author, Book, WordOccurrence

# Register your models here.
admin.site.register(Book)
admin.site.register(WordOccurrence)
admin.site.register(Author)