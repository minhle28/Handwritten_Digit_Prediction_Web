import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=9, figsize=(10, 6))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
plt.subplots_adjust(wspace=1)
# plt.savefig("cloud/img/Graph_Training.png")
plt.savefig("cloud/img/training_plot.png")
#############################
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Learn the digits on the train subset
###########################
import ray
import joblib
from ray.util.joblib import register_ray
ray.init(address='auto')
register_ray()
with joblib.parallel_backend('ray'):
    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )
    clf.fit(X_train, y_train)
    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)
#################

#####################
_, axes = plt.subplots(nrows=1, ncols=9, figsize=(10, 6))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
plt.subplots_adjust(wspace=1)
# plt.savefig("cloud/img/Graph_Predicted.png")
plt.savefig("predicted_plot.png")
#####################
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

########################
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

# plt.savefig("cloud/img/graph.png")
plt.savefig("confusion_matrix_plot.png")
##########################
# The ground truth and predicted lists
y_true = []
y_pred = []
cm = disp.confusion_matrix

# For each cell in the confusion matrix, add the corresponding ground truths
# and predictions to the lists
for gt in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [gt] * cm[gt][pred]
        y_pred += [pred] * cm[gt][pred]

print(
    "Classification report rebuilt from confusion matrix:\n"
    f"{metrics.classification_report(y_true, y_pred)}\n"
)

