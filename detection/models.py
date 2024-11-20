from django.db import models

class ImageProcessing(models.Model):
    image = models.ImageField(upload_to='uploaded_images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    PREPROCESSING_CHOICES = [
        ('histogram_equalization', 'Histogram Equalization'),
        ('clahe', 'CLAHE'),
        ('gaussian_blur', 'Gaussian Blur'),
        ('median_filtering', 'Median Filtering'),
        ('dog', 'Difference of Gaussian'),
        ('bilateral', 'Bilateral Filtering'),
        ('no', 'None'),
    ]

    SEGMENTATION_CHOICES = [
        ('thresholding', 'Thresholding'),
        ('otsu_thresholding', 'Otsu Thresholding'),
        ('kmeans', 'K Means Clustering Segmentation'),
        ('watershed', 'Watershed Segmentation'),
        ('color', 'Color Segmentation'),
        ('grabcut', 'GrabCut Algorithm'),
        ('no', 'None'),
    ]

    preprocessing_technique = models.CharField(max_length=50, choices=PREPROCESSING_CHOICES, default='no')
    segmentation_technique = models.CharField(max_length=50, choices=SEGMENTATION_CHOICES, default='no')

    def __str__(self):
        return f"Image {self.id}: Preprocessing - {self.get_preprocessing_technique_display()}, Segmentation - {self.get_segmentation_technique_display()}"

class DetectionResult(models.Model):
    image = models.ImageField(upload_to='images/')
    preprocessing = models.CharField(max_length=100)  # Store preprocessing technique as text
    segmentation = models.CharField(max_length=100)  # Store segmentation technique as text
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()

    def __str__(self):
        return f"Result for {self.image_name}"  # Return the name of the image

