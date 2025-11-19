from django.contrib import admin
from .models import Notebook


@admin.register(Notebook)
class NotebookAdmin(admin.ModelAdmin):
    list_display = ['title', 'notebook_id', 'order', 'created_at']
    list_filter = ['created_at', 'updated_at']
    search_fields = ['title', 'description', 'notebook_id']
    ordering = ['order']
