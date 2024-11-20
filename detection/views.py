from django.shortcuts import render, redirect
from .models import DetectionResult, ImageProcessing
from .forms import ImageProcessingForm
from .helpers import *
import cv2
from django.http import HttpResponse
import os
import itertools
from django.conf import settings


def single_view(request):
    if request.method == "POST":
        form = ImageProcessingForm(request.POST, request.FILES)
        print(request.POST)
        print(request.FILES)
        if form.is_valid():
            form.save()
            request.session["preprocessing"] = form.cleaned_data[
                "preprocessing_technique"
            ]
            request.session["segmentation"] = form.cleaned_data[
                "segmentation_technique"
            ]
            return redirect("detect_faces")
    else:
        form = ImageProcessingForm()
    return render(request, "single_view.html", {"form": form})


def detect_faces_view(request):
    # Get the latest uploaded image
    print("hello")
    image_record = ImageProcessing.objects.order_by("-uploaded_at").first()
    image_instance = image_record.image
    image_path = image_instance.path
    image_name = image_path.split("\\")[-1]  # Get the filename for matching
    print(image_name)
    upload_folder = os.path.join(settings.MEDIA_ROOT, "uploaded_images")
    preprocessed_upload_folder = os.path.join(
        settings.MEDIA_ROOT, "preprocessed_images"
    )
    segmented_upload_folder = os.path.join(settings.MEDIA_ROOT, "segmented_images")
    detected_upload_folder = os.path.join(settings.MEDIA_ROOT, "detected_images")
    # Get selected preprocessing and segmentation techniques
    preprocessing_technique = request.session["preprocessing"]
    segmentation_technique = request.session["segmentation"]
    img = cv2.imread(image_path)
    # Apply selected preprocessing technique
    preprocessed_image = apply_preprocessing(img, preprocessing_technique)

    # Apply selected segmentation technique
    segmented_image = apply_segmentation(preprocessed_image, segmentation_technique)

    # Perform face detection on the processed image
    detected_image, detected_faces = detect_faces(segmented_image)

    preprocessed_file_path = os.path.join(preprocessed_upload_folder, image_name)
    segmented_file_path = os.path.join(segmented_upload_folder, image_name)
    detected_file_path = os.path.join(detected_upload_folder, image_name)
    upload_file_path = os.path.join(upload_folder, image_name)
    cv2.imwrite(upload_file_path, img)
    cv2.imwrite(preprocessed_file_path, preprocessed_image)
    cv2.imwrite(segmented_file_path, segmented_image)
    cv2.imwrite(detected_file_path, detected_image)

    precision, recall, accuracy = calculate_metrics(detected_faces, image_name)

    # Save the detection results
    result = DetectionResult.objects.create(
        image=image_instance,
        preprocessing=preprocessing_technique,
        segmentation=segmentation_technique,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
    )

    # Save the detected image for display
    detected_image_path = f"media/detected_{image_record.image.name}"
    print(detected_image_path)
    cv2.imshow("d", detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(detected_image_path, detected_image)

    return render(
        request,
        "detection_result.html",
        {
            "result": result,
            "image_url": os.path.join(
                settings.MEDIA_URL, "uploaded_images", image_name
            ),
            "preprocessed_url": os.path.join(
                settings.MEDIA_URL, "preprocessed_images", image_name
            ),
            "segmented_url": os.path.join(
                settings.MEDIA_URL, "segmented_images", image_name
            ),
            "detected_image_url": os.path.join(
                settings.MEDIA_URL, "detected_images", image_name
            ),
        },
    )


def batch_view(request):
    if request.method == "POST":
        files = request.FILES.getlist("files")
        upload_folder = os.path.join(settings.MEDIA_ROOT, "uploaded_images")
        preprocessed_upload_folder = os.path.join(
            settings.MEDIA_ROOT, "preprocessed_images"
        )
        segmented_upload_folder = os.path.join(settings.MEDIA_ROOT, "segmented_images")
        detected_upload_folder = os.path.join(settings.MEDIA_ROOT, "detected_images")

        # Ensure the upload folder exists
        os.makedirs(upload_folder, exist_ok=True)

        preprocessing_techniques = [
            "no",
            "gaussian_blur",
            "histogram_equalization",
            "median_filtering",
            "clahe",
            "dog",
            "bilateral",
        ]
        segmentation_techniques = [
            "no",
            "thresholding",
            "otsu_thresholding",
            "watershed",
            "kmeans",
            "color",
        ]

        technique_combinations = list(
            itertools.product(preprocessing_techniques, segmentation_techniques)
        )
        results = []

        for file in files:
            file_path = os.path.join(upload_folder, file.name)
            with open(file_path, "wb") as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            image = cv2.imread(file_path)
            best_accuracy = 0
            best_precision = 0
            best_combination = None
            best_preprocessed_image = None
            best_segmented_image = None
            best_detected_image = None
            metrics_comparison = []

            preprocessed_file_path = os.path.join(preprocessed_upload_folder, file.name)
            segmented_file_path = os.path.join(segmented_upload_folder, file.name)
            detected_file_path = os.path.join(detected_upload_folder, file.name)
            for preprocessing, segmentation in technique_combinations:
                # Apply preprocessing
                preprocessed_image = apply_preprocessing(image, preprocessing)

                # Apply segmentation
                segmented_image = apply_segmentation(preprocessed_image, segmentation)

                # Perform face detection
                detected_image, detected_faces = detect_faces(segmented_image)

                # Compare with true annotations (placeholder for actual ground truth comparison)
                precision, recall, accuracy = calculate_metrics(
                    detected_faces, file.name
                )

                metrics_comparison.append(
                    {
                        "preprocessing": preprocessing,
                        "segmentation": segmentation,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                    }
                )

                # Check if this is the best combination so far
                if accuracy > best_accuracy or (
                    accuracy == best_accuracy and precision > best_precision
                ):
                    best_accuracy = accuracy
                    best_precision = precision
                    best_combination = (preprocessing, segmentation)
                    best_preprocessed_image = preprocessed_image
                    best_segmented_image = segmented_image
                    best_detected_image = detected_image

            metrics_comparison = sorted(
                metrics_comparison,
                key=lambda x: (x["accuracy"], x["precision"]),
                reverse=True,
            )[:5]
            cv2.imwrite(preprocessed_file_path, best_preprocessed_image)
            cv2.imwrite(segmented_file_path, best_segmented_image)
            cv2.imwrite(detected_file_path, best_detected_image)

            best_preprocessing, best_segmentation = (
                best_combination if best_combination else ("No", "No")
            )

            # Determine why the combination was best
            reason = determine_best_reason(best_preprocessing, best_segmentation)
            # Save results for this image
            print(file.name)
            print(best_combination)
            results.append(
                {
                    "filename": file.name,
                    "best_preprocessing": (
                        best_combination[0] if best_combination else "No"
                    ),
                    "best_segmentation": (
                        best_combination[1] if best_combination else "No"
                    ),
                    "accuracy": best_accuracy,
                    "precision": best_precision,
                    "metrics_comparison": metrics_comparison,
                    "reason": reason,
                    "image_url": os.path.join(
                        settings.MEDIA_URL, "uploaded_images", file.name
                    ),
                    "preprocessed_url": os.path.join(
                        settings.MEDIA_URL, "preprocessed_images", file.name
                    ),
                    "segmented_url": os.path.join(
                        settings.MEDIA_URL, "segmented_images", file.name
                    ),
                    "detected_image_url": os.path.join(
                        settings.MEDIA_URL, "detected_images", file.name
                    ),
                }
            )

        return render(request, "batch_view.html", {"results": results})

    return render(request, "batch_view.html")
