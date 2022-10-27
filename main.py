import numpy as np
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
import sklearn
from sklearn import svm, metrics, neighbors, linear_model, gaussian_process, cross_decomposition, tree, neural_network
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Import data
csvname = 'data/HTRU_2.csv'
data = np.loadtxt(csvname, delimiter=',')

# Split data in X and Y
x = data[:,:-1]
y = np.ravel(data[:, -1:])

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, shuffle=True) # Use features_pca to use pca

# Normalize data with mean and deviation from train data
scaler = preprocessing.StandardScaler().fit(X_train)
data_normalized_train = scaler.transform(X_train)
data_normalized_test = scaler.transform(X_test)

# Use Linear Support Vector Classifier
# model = svm.LinearSVC().fit(data_normalized_train, y_train)

# K neighbours classifier
# model = neighbors.KNeighborsClassifier().fit(data_normalized_train, y_train)

# Check deze website voor info: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
# Supervised neural network
model = neural_network.MLPClassifier(hidden_layer_sizes=(100,50,20,10,5,2), random_state=1, solver='sgd', learning_rate='adaptive', activation='relu').fit(data_normalized_train, y_train)


# Accuracy, score and confusion matrix generation
y_pred_test = model.predict(data_normalized_test)
y_pred_train = model.predict(data_normalized_train)
score_test = round(model.score(data_normalized_test, y_test), 4)
score_train = round(model.score(data_normalized_train, y_train), 4)
print(f'Score for test: {score_test}')
print(f'Score for train: {score_train}')
print(f'Balanced test accuracy score is: {metrics.balanced_accuracy_score(y_test, y_pred_test)}')
print(f'Balanced train accuracy score is: {metrics.balanced_accuracy_score(y_train, y_pred_train)}')
disp = metrics.plot_confusion_matrix(model, data_normalized_test, y_test)
disp.figure_.suptitle("Confusion Matrix Test")
disp2 = metrics.plot_confusion_matrix(model, data_normalized_train, y_train)
disp2.figure_.suptitle("Confusion Matrix Train")
plt.show()


