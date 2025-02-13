from django.db import models

# Create your models here.
class Book(models.Model):
    ids = models.AutoField(primary_key=True)
    idGutendex = models.IntegerField(unique=True)
    title = models.CharField(max_length=200)
    author = models.ManyToManyField('Author')
    summary = models.TextField()
    cover = models.URLField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    linkToBook = models.URLField(max_length=200)
    downloadCount = models.IntegerField(default=0)


class Author(models.Model):
    idsAuthor = models.AutoField(primary_key=True)
    name = models.CharField(max_length=200)
    birth_date = models.DateField(null=True, blank=True)
    death_date = models.DateField(null=True, blank=True)
class WordOccurrence(models.Model):
    term = models.CharField(max_length=255)
    count = models.PositiveIntegerField(default=1)
    book = models.ForeignKey(Book,on_delete=models.CASCADE)
    tfidf_weight = models.FloatField(null=True, blank=True)
    term_frequency = models.FloatField(default=0)
    class Meta:
        unique_together  = ("term","book")
