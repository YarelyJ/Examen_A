from rest_framework import serializers
from .models import Notebook


class NotebookSerializer(serializers.ModelSerializer):
    """Serializer for Notebook model"""
    
    class Meta:
        model = Notebook
        fields = ['id', 'notebook_id', 'title', 'description', 'order', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']


class NotebookDetailSerializer(serializers.Serializer):
    """Serializer for detailed notebook information"""
    
    notebook_id = serializers.CharField()
    title = serializers.CharField()
    description = serializers.CharField()
    sections = serializers.ListField(child=serializers.DictField())
    code_examples = serializers.ListField(child=serializers.DictField())
    key_points = serializers.ListField(child=serializers.CharField())
