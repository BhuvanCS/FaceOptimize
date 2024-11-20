from django import forms
from .models import ImageProcessing
import os


class ImageProcessingForm(forms.ModelForm):
    class Meta:
        model = ImageProcessing
        fields = ['image', 'preprocessing_technique', 'segmentation_technique']
