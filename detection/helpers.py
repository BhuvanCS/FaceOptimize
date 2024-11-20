import cv2
import numpy as np
from django.http import HttpResponse

# Helper function for image preprocessing
def apply_preprocessing(img, technique):
    if technique == 'gaussian_blur':
        processed_img = cv2.GaussianBlur(img, (5, 5), 0)
    elif technique == 'histogram_equalization':
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.equalizeHist(img_gray)
    elif technique == 'clahe':
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed_img = clahe.apply(img_gray)
    elif technique == 'median_filtering':
        processed_img = cv2.medianBlur(img, 5)
    elif technique == 'dog':
        blurred1 = cv2.GaussianBlur(img, (5, 5), 0)
        blurred2 = cv2.GaussianBlur(img, (9, 9), 0)
        processed_img = cv2.subtract(blurred1, blurred2)
    elif technique == 'bilateral':
        processed_img = cv2.bilateralFilter(img, 9, 75, 75)
    else:
        processed_img = img  # Default: no preprocessing
    return processed_img

# Helper function for segmentation
def apply_segmentation(image, technique):
    if technique == 'thresholding':
        _, segmented_img = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    elif technique == 'otsu_thresholding':
        if len(image.shape) == 3:  
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        _, segmented_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif technique == 'kmeans':
        K=2
        if len(image.shape) == 2:  # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to RGB
        Z = image.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        segmented_img = res.reshape((image.shape))
    elif technique == 'grabcut':
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        segmented_img = image * mask2[:, :, np.newaxis]
    elif technique == 'watershed':
        if(len(image.shape) == 3):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Apply Otsu's thresholding to obtain a binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert the binary image
        inverted_binary = cv2.bitwise_not(binary)

        # Find sure regions
        distance_transform = cv2.distanceTransform(inverted_binary, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(distance_transform, 0.7 * distance_transform.max(), 255, 0)

        # Find unknown region (background - foreground)
        unknown = cv2.subtract(inverted_binary, sure_fg.astype(np.uint8))

        # Label markers
        _, markers = cv2.connectedComponents(sure_fg.astype(np.uint8))

        # Add one to all labels to distinguish the background
        markers = markers + 1
        markers[unknown == 255] = 0  # Mark the unknown region as 0

        # Apply the watershed algorithm
        cv2.watershed(image, markers)
        segmented_img = image.copy()
        segmented_img[markers == -1] = [255, 0, 0]  # Mark boundaries in red
    elif technique == 'color':
        # Example: Simple color segmentation for skin tones
        lower_skin = np.array([0, 20, 70], dtype="uint8")
        upper_skin = np.array([20, 255, 255], dtype="uint8")
        if(len(image.shape) != 3):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Convert to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a mask
        mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
        segmented_img = cv2.bitwise_and(image, image, mask=mask)
    else:
        segmented_img = image  # Default: no segmentation
    return segmented_img

# Helper function for face detection
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error loading cascade classifier.")
    if len(image.shape) == 3:  # Color image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Grayscale image
        gray = image
    faces = face_cascade.detectMultiScale(gray, 1.3, minNeighbors=5, minSize=(30,30), flags= cv2.CASCADE_SCALE_IMAGE)

    detected_faces = []
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        detected_faces.append([x, y, w, h])
    

    return image, detected_faces

def parse_annotation(annotation_path, image_name):
    """
    Given an annotation file and image name, parse and return the bounding box data.
    """
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
    
    # Iterate through lines and find the image name
    for i, line in enumerate(lines):
        if image_name in line:
            # The next line after image name contains the number of faces
            num_faces = int(lines[i + 1].strip())
            # Extract the bounding boxes for all faces
            bounding_boxes = []
            for j in range(i + 2, i + 2 + num_faces):
                bbox_data = list(map(int, lines[j].strip().split()[:4]))  # Only take x, y, w, h
                bounding_boxes.append(bbox_data)
            return bounding_boxes
    return None  # No matching image found

def calculate_metrics(detected_faces, image_name):
    # Parse the ground truth bounding boxes from the annotation file
    annotation_file = (
        "media/wider_dataset/WIDER_val/WIDER_val/wider_face_val_bbx_gt.txt"
    )
    ground_truth_boxes = parse_annotation(annotation_file, image_name)
    print(ground_truth_boxes)
    if ground_truth_boxes is None:
        return HttpResponse("No annotations found for this image.")

    true_positives = 0
    false_positives = 0
    false_negatives = len(ground_truth_boxes)

    for detected_box in detected_faces:
        for ground_truth_box in ground_truth_boxes:
            if iou(detected_box, ground_truth_box) > 0.5:  # IoU threshold for matching
                true_positives += 1
                false_negatives -= 1
                break
        else:
            false_positives += 1

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    accuracy = (
        true_positives / len(ground_truth_boxes) if len(ground_truth_boxes) > 0 else 0
    )

    return precision, recall, accuracy

def iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    box = [x, y, width, height]
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    # Area of the intersection
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Areas of the two boxes
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # Area of the union
    union_area = box1_area + box2_area - inter_area
    
    # Intersection over Union (IoU)
    iou_value = inter_area / union_area if union_area > 0 else 0
    return iou_value

def determine_best_reason(preprocessing, segmentation):
    """Explain why a preprocessing and segmentation combination was best."""
    explanations = {
        "gaussian_blur": "Reduces noise effectively for smoother segmentation.",
        "histogram_equalization": "Enhances contrast, making features more distinct.",
        "median_filtering": "Removes salt-and-pepper noise, preserving edges.",
        "clahe": "Enhances local contrast, especially useful for non-uniform lighting.",
        "dog": "Highlights edges, improving facial boundary detection.",
        "bilateral": "Reduces noise while preserving edges, suitable for textured regions.",
        "thresholding": "Simplifies the image into binary regions, useful for high contrast.",
        "otsu_thresholding": "Automatically determines an optimal threshold for segmentation.",
        "watershed": "Segments overlapping regions based on intensity gradients.",
        "kmeans": "Clusters pixels into regions, effective for complex backgrounds.",
        "color": "Separates regions based on color similarity, useful for distinct color features.",
    }
    return f"{explanations.get(preprocessing, '')} {explanations.get(segmentation, '')}"