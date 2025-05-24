# This is a sample Python script.
from doc_classifier import build_classifier_c1, evaluate_classifier, train_images, train_labels, test_images, \
    test_labels


# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    clf_c1_linear = build_classifier_c1(train_images, train_labels, kernel='linear')
    clf_c1_rbf = build_classifier_c1(train_images, train_labels, kernel='rbf')
    accuracy_linear, _ = evaluate_classifier(clf_c1_linear, test_images, test_labels)
    accuracy_rbf, _ = evaluate_classifier(clf_c1_rbf, test_images, test_labels)
    print(f"C1 (Linear) Accuracy: {accuracy_linear:.2f}")
    print(f"C1 (RBF) Accuracy: {accuracy_rbf:.2f}")
    clf_c1 = clf_c1_rbf if accuracy_rbf > accuracy_linear else clf_c1_linear  # Use best
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
