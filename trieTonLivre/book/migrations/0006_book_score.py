# Generated by Django 5.1.5 on 2025-02-13 09:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('book', '0005_rename_word_wordoccurrence_term_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='book',
            name='score',
            field=models.FloatField(default=0),
        ),
    ]
