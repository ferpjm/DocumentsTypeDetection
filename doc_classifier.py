import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import os
import sys

def load_images(directory, size=(400, 300)):
    """Load and preprocess images from the specified directory."""
    images = []
    labels = []
    class_map = {
        'Comics': 0, 'Libros': 1, 'Manuscrito': 2,
        'Mecanografiado': 3, 'Tickets': 4
    }
    valid_extensions = ('.jpg', '.jpeg', '.png')

    for class_name in class_map:
        class_dir = os.path.join(directory, class_name)
        if not os.path.exists(class_dir):
            print(f"Directory not found: {class_dir}")
            continue
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(valid_extensions):
                continue
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
                img_vector = (img.flatten().astype(np.float32) / 255.0)
                images.append(img_vector)
                labels.append(class_map[class_name])
            else:
                print(f"Failed to load image: {img_path}")

    print(f"Loaded {len(images)} images with {len(labels)} labels from {directory}")
    images_array = np.array(images)
    if images_array.ndim == 1:
        images_array = images_array.reshape(1, -1)
    return images_array, np.array(labels, dtype=np.int32)

def build_classifier_c1(images, labels):
    """Build the C1 classifier using SVM."""
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(images, labels)
    return clf

def build_classifier_c2(images, labels, n_components=4, kernel='rbf', C=100.0):
    """Build the C2 classifier using LDA + SVM."""
    scaler = StandardScaler()
    images_scaled = scaler.fit_transform(images)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    reduced_images = lda.fit_transform(images_scaled, labels)
    print(f"Reduced features shape: {reduced_images.shape}")
    print(f"Feature variance: {np.var(reduced_images, axis=0)}")
    clf = SVC(kernel=kernel, C=C, class_weight='balanced', random_state=42)
    clf.fit(reduced_images, labels)
    return clf, lda, scaler

def evaluate_classifier(classifier, test_images, test_labels):
    """Evaluate the classifier and return metrics."""
    predictions = classifier.predict(test_images)
    accuracy = np.mean(predictions == test_labels)
    report = classification_report(test_labels, predictions,
                                  target_names=['Comics', 'Libros', 'Manuscrito', 'Mecanografiado', 'Tickets'],
                                  output_dict=True)
    conf_matrix = confusion_matrix(test_labels, predictions)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    return accuracy, report

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python doc_classifier.py imagen.jpg")
        sys.exit(1)

    train_images, train_labels = load_images('MUESTRA\\Aprendizaje')
    if len(train_images) != 125:
        print(f"Error: Expected 125 training images, but loaded {len(train_images)}. Check directory structure.")
        sys.exit(1)
    class_map = {0: 'Comics', 1: 'Libros', 2: 'Manuscrito', 3: 'Mecanografiado', 4: 'Tickets'}
    label_counts = np.bincount(train_labels)
    print("Training label distribution:")
    for i, count in enumerate(label_counts):
        print(f"{class_map[i]}: {count}")

    clf_c1 = build_classifier_c1(train_images, train_labels)
    clf_c2, lda_c2, scaler_c2 = build_classifier_c2(train_images, train_labels, n_components=4, kernel='rbf', C=100.0)

    test_images, test_labels = load_images('MUESTRA\\Test')
    if len(test_images) != 40:
        print(f"Error: Expected 40 test images, but loaded {len(test_images)}. Check directory structure.")
        sys.exit(1)

    # Evaluate C1
    accuracy_c1, report_c1 = evaluate_classifier(clf_c1, test_images, test_labels)
    print(f"C1 Test Accuracy: {accuracy_c1:.2f}")
    print("\nC1 Classification Report:")
    for label, metrics in report_c1.items():
        if label in ['Comics', 'Libros', 'Manuscrito', 'Mecanografiado', 'Tickets']:
            print(f"{label}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1-Score={metrics['f1-score']:.2f}")

    # Evaluate C2
    test_images_scaled = scaler_c2.transform(test_images)
    test_images_reduced = lda_c2.transform(test_images_scaled)
    accuracy_c2, report_c2 = evaluate_classifier(clf_c2, test_images_reduced, test_labels)
    print(f"\nC2 Test Accuracy: {accuracy_c2:.2f}")
    print("\nC2 Classification Report:")
    for label, metrics in report_c2.items():
        if label in ['Comics', 'Libros', 'Manuscrito', 'Mecanografiado', 'Tickets']:
            print(f"{label}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1-Score={metrics['f1-score']:.2f}")

    # Classify a single image with both classifiers
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (400, 300), interpolation=cv2.INTER_LINEAR)
        img_vector = (img.flatten().astype(np.float32) / 255.0).reshape(1, -1)
        prediction_c1 = clf_c1.predict(img_vector)
        img_vector_scaled = scaler_c2.transform(img_vector)
        img_vector_reduced = lda_c2.transform(img_vector_scaled)
        prediction_c2 = clf_c2.predict(img_vector_reduced)
        class_map = {0: 'Comics', 1: 'Libros', 2: 'Manuscrito', 3: 'Mecanografiado', 4: 'Tickets'}
        print(f"\nC1 Predicted class for {img_path}: {class_map[prediction_c1[0]]}")
        print(f"C2 Predicted class for {img_path}: {class_map[prediction_c2[0]]}")
    else:
        print(f"Error: Could not load image {img_path}")