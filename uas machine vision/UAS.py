import numpy as np
from skimage.feature import hog
from sklearn import datasets, model_selection, svm, metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load MNIST dataset
digits = datasets.fetch_openml('mnist_784')
data = digits.data
target = digits.target.astype(int)

# Extract HOG features
def extract_hog_features(data):
    hog_features = []
    for image in data:
        fd = hog(image.reshape((28, 28)), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=False)
        hog_features.append(fd)
    return np.array(hog_features)

hog_features = extract_hog_features(data)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(hog_features, target, test_size=0.2, random_state=42)

# Train SVM classifier
clf = svm.SVC()
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate performance
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(np.arange(10), np.arange(10))
plt.yticks(np.arange(10), np.arange(10))
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.show()
