# Generated by Django 5.0.6 on 2024-06-15 15:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analytic', '0005_favorite'),
    ]

    operations = [
        migrations.AddField(
            model_name='service',
            name='category',
            field=models.CharField(choices=[('AI', 'Artificial Intelligence'), ('DataAnalysis', 'Data Analysis'), ('DataScience', 'Data Science'), ('Other', 'Other')], default='Other', max_length=20),
        ),
    ]
