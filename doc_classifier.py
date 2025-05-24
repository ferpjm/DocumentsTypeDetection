import cv2
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import os
import sys

# Identity scaler for precomputed features
class IdentityScaler:
    def transform(self, X):
        return X

# Rectification functions

def order_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)


def rectify_document(img, dst_size=(400,300)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4,2).astype(np.float32)
            rect = order_points(pts)
            w, h = dst_size
            dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (w, h))
            return warped
    return cv2.resize(img, dst_size, interpolation=cv2.INTER_LINEAR)

# HOG extraction with OpenCV

def extract_hog_opencv(gray, win_size=(400,300)):
    cell_size = (10,10)
    block_size = (cell_size[0]*2, cell_size[1]*2)
    block_stride = cell_size
    nbins = 12
    hog = cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=block_size,
        _blockStride=block_stride,
        _cellSize=cell_size,
        _nbins=nbins
    )
    return hog.compute(gray).flatten()

# Data loading

def load_images(directory, size=(400,300), rectify=False, use_hog=False):
    images, labels = [], []
    class_map = {'Comics':0, 'Libros':1, 'Manuscrito':2, 'Mecanografiado':3, 'Tickets':4}
    for cls, lbl in class_map.items():
        path = os.path.join(directory, cls)
        if not os.path.isdir(path): continue
        for fname in os.listdir(path):
            if not fname.lower().endswith(('.jpg','.jpeg','.png')): continue
            img = cv2.imread(os.path.join(path, fname))
            if img is None: continue
            img_proc = rectify_document(img, size) if rectify else cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
            vec = extract_hog_opencv(gray, win_size=size) if use_hog else gray.flatten().astype(np.float32)/255.0
            images.append(vec)
            labels.append(lbl)
    X = np.array(images)
    y = np.array(labels, dtype=np.int32)
    mode = ('rectified ' if rectify else '') + ('HOG ' if use_hog else '')
    print(f"Loaded {len(X)} {mode}images from {directory}")
    return X, y

# Classifier builders

def build_classifier_c1(X, y, kernel='linear', C=1.0):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = SVC(kernel=kernel, C=C, class_weight='balanced', random_state=42)
    clf.fit(Xs, y)
    return clf, scaler


def build_classifier_c2(X, y, n_pca=50, n_lda=4, kernel='rbf', C=1.0, gamma='scale'):
    scaler1 = StandardScaler()
    Xs = scaler1.fit_transform(X)
    pca = PCA(n_components=n_pca, whiten=True, random_state=42)
    Xp = pca.fit_transform(Xs)
    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    Xl = lda.fit_transform(Xp, y)
    scaler2 = StandardScaler()
    Xf = scaler2.fit_transform(Xl)
    clf = SVC(kernel=kernel, C=C, gamma=gamma, class_weight='balanced', random_state=42)
    clf.fit(Xf, y)
    return clf, scaler1, pca, lda, scaler2

def build_classifier_c3(X, y, n_pca=75, n_lda=4):
    scaler1 = StandardScaler()
    Xs = scaler1.fit_transform(X)
    if np.any(np.isnan(Xs)) or np.any(np.isinf(Xs)):
        print("Datos con NaN o inf después de StandardScaler")
    pca = PCA(n_components=n_pca, whiten=True, random_state=42)
    Xp = pca.fit_transform(Xs)
    print(f"C3 PCA output shape: {Xp.shape}")
    if np.any(np.isnan(Xp)) or np.any(np.isinf(Xp)):
        print("Datos con NaN o inf después de PCA")
    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    Xl = lda.fit_transform(Xp, y)
    print(f"C3 LDA output shape: {Xl.shape}")
    if np.any(np.isnan(Xl)) or np.any(np.isinf(Xl)):
        print("Datos con NaN o inf después de LDA")
    scaler2 = StandardScaler()
    Xf = scaler2.fit_transform(Xl)
    if np.any(np.isnan(Xf)) or np.any(np.isinf(Xf)):
        print("Datos con NaN o inf después del segundo StandardScaler")

    param_grid = {'C': [1, 10], 'kernel': ['rbf'], 'gamma': ['scale']}
    clf = GridSearchCV(SVC(class_weight='balanced', random_state=42), param_grid, cv=5, verbose=2)
    try:
        print(f"Entrenando GridSearchCV con datos de forma: {Xf.shape}")
        clf.fit(Xf, y)
        print(f"Mejores parámetros: {clf.best_params_}")
    except Exception as e:
        print(f"Error en GridSearchCV: {e}")
        raise
    return clf.best_estimator_, scaler1, pca, lda, scaler2

# Evaluation utility

def evaluate(clf, scaler, X, y, names):
    Xs = scaler.transform(X)
    preds = clf.predict(Xs)
    acc = np.mean(preds == y)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, preds))
    print(classification_report(y, preds, target_names=names))
    print(f"Accuracy: {acc:.2f}\n")
    return acc

# Main

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python doc_classifier.py <image>")
        sys.exit(1)

    names = ['Comics', 'Libros', 'Manuscrito', 'Mecanografiado', 'Tickets']

    # Load training data
    X_train, y_train = load_images('MUESTRA/Aprendizaje')
    # Build classifiers
    clf1, sc1 = build_classifier_c1(X_train, y_train)
    clf2, sc2_1, pca, lda, sc2_2 = build_classifier_c2(X_train, y_train)
    Xr_train, yr_train = load_images('MUESTRA/Aprendizaje', rectify=True, use_hog=True)
    clf3, c3_s1, c3_pca, c3_lda, c3_s2 = build_classifier_c3(Xr_train, yr_train)
    # Load test data
    X_test, y_test = load_images('MUESTRA/Test')
    Xr_test, yr_test = load_images('MUESTRA/Test', rectify=True, use_hog=True)

    # Evaluate C1
    print('=== C1 Results ===')
    evaluate(clf1, sc1, X_test, y_test, names)

    # Evaluate C2
    Xt = sc2_1.transform(X_test)
    Xt = pca.transform(Xt)
    Xt = lda.transform(Xt)
    Xt = sc2_2.transform(Xt)
    print('=== C2 Results ===')
    evaluate(clf2, IdentityScaler(), Xt, y_test, names)

        # Evaluate C3 with PCA+LDA on HOG
    clf3, c3_s1, c3_pca, c3_lda, c3_s2 = build_classifier_c3(Xr_train, yr_train)
    # Prepare C3 test features
    Xc = c3_s1.transform(Xr_test)
    Xc = c3_pca.transform(Xc)
    Xc = c3_lda.transform(Xc)
    Xc = c3_s2.transform(Xc)
    print('=== C3 Results (HOG + PCA+LDA) ===')
    evaluate(clf3, IdentityScaler(), Xc, yr_test, names)

        # Classify single image using all three pipelines
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Error loading image {sys.argv[1]}")
        sys.exit(1)

        # Preprocess vector for C1 & C2
    img_resized = cv2.resize(img, (400,300), interpolation=cv2.INTER_LINEAR)
    gray_single = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    vec = gray_single.flatten().astype(np.float32) / 255.0
    # C1 prediction
    p1 = clf1.predict(sc1.transform(vec.reshape(1,-1)))[0]
    # C2 prediction
    v2 = sc2_1.transform(vec.reshape(1,-1))
    v2 = pca.transform(v2)
    v2 = lda.transform(v2)
    v2 = sc2_2.transform(v2)
    p2 = clf2.predict(v2)[0]

    # C3 prediction
    rec = rectify_document(img)
    gray = cv2.cvtColor(rec, cv2.COLOR_BGR2GRAY)
    hog_vec = extract_hog_opencv(gray, win_size=(400,300)).reshape(1, -1)
    v3 = c3_s1.transform(hog_vec)
    v3 = c3_pca.transform(v3)
    v3 = c3_lda.transform(v3)
    v3 = c3_s2.transform(v3)
    p3 = clf3.predict(v3)[0]

    print(f"C1 prediction: {names[p1]}")
    print(f"C2 prediction: {names[p2]}")
    print(f"C3 prediction: {names[p3]}")
