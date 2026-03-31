# 1. Завантажити набір даних за допомогою load_iris() бібліотеки sklearn.
# 2. Розподілити дані на навчальні та для тестування за допомогою train_test_split() бібліотеки sklearn.
# 3. Використати вибірку ознак окремо для кожного класа.
# 4. Реалізувати розрахунок матриць коваріації для набору ознак кожного класа.
# 5. Реалізувати розрахунок обернених матриць коваріації.
# 6. Обчислити апріорні імовірності кожного класа у тренувальних даних.
# 7. Реалізувати функцію обчислення значень дискримінантної функції для одного рядка (вектора) тестових даних.
# 8. Реалізувати функцію обчислення значень дискримінантної функції та імовірностей приналежності кожному класу для всієї матриці тестових даних.
# 9. Виконати прогнозування на тестових даних за допомогою функії QuadraticDiscriminantAnalysis() бібліотеки sklearn та порівняти отримані результати.
# 10. Зробити висновок про ступінь схожості результатів, отриманих власною функцією та бібліотекою sklearn.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro
from sklearn.datasets import load_iris  # pip install scikit-learn
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

shapiro_wilk_threshold = 0.05
test_size = 0.8
random_state = 42

# 1. Завантажити та створити DataFrame: Використовуйте бібліотеку  pandas для створення та роботи з DataFrame. Дані можна завантажити за допомогою load_iris() з бібліотеки **sklearn**.
iris = load_iris()
# print(iris.DESCR[:1500])
feature_names = iris.feature_names
class_names = iris.target_names

print("\nFeature names:", feature_names)
print("Target names:", class_names)

print("\nclass 0 is represented by:", class_names[0])
print("class 1 is represented by:", class_names[1])
print("class 2 is represented by:", class_names[2])

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)


print("\nFirst 5 rows of the DataFrame:\n", df.head())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
print("Total record count:", len(scaled_data))

# 2. Розподілити дані на навчальні та для тестування за допомогою train_test_split() бібліотеки sklearn.
X_train, X_test, y_train, y_test = train_test_split(scaled_data, iris.target, test_size=0.8, random_state=42, stratify=iris.target)

print("\n\nTrain test split result:")
print("\tTraining set size:", X_train.shape[0])
print("\tTest set size:", X_test.shape[0])
print("\tNumber of features:", X_train.shape[1])
print("\tNumber of classes:", len(np.unique(y_train)))
print("\tClass distribution in training set:", np.bincount(y_train))
print("\tClass distribution in test set:", np.bincount(y_test))


plt.figure(figsize=(20, 15))

for i in range(3):
    for j, feature_name in enumerate(feature_names):
        # show result of Shapiro-Wilk Test for normality on the same plot for each class and feature
        stat, p = shapiro(X_train[y_train == i][:, j])
        normal = p > shapiro_wilk_threshold

        plt.subplot(3, 4, i * 4 + j + 1)
        sns.histplot(X_train[y_train == i][:, j], kde=True)
        plt.title(f"Class {i} ({class_names[i]}) - {feature_name}\nShapiro-Wilk: normal={normal}")

plt.show(block=True)

# 3. Використати вибірку ознак окремо для кожного класа.
print("\nExtracting features for each class in the training data:")
class_0_features = X_train[y_train == 0]
class_1_features = X_train[y_train == 1]
class_2_features = X_train[y_train == 2]

print(f"\tClass 0-({class_names[0]}) features shape:", class_0_features.shape)
print(f"\tClass 1-({class_names[1]}) features shape:", class_1_features.shape)
print(f"\tClass 2-({class_names[2]}) features shape:", class_2_features.shape)

# 4. Реалізувати розрахунок матриць коваріації для feature_name кожного класа.
print("\nCovariance matrices for each class in the training data:".capitalize())
cov_matrix_class_0 = np.cov(class_0_features, rowvar=False)
cov_matrix_class_1 = np.cov(class_1_features, rowvar=False)
cov_matrix_class_2 = np.cov(class_2_features, rowvar=False)
print(f"\n\tclass 0-({class_names[0]}):\n", cov_matrix_class_0)
print(f"\n\tclass 1-({class_names[1]}):\n", cov_matrix_class_1)
print(f"\n\tclass 2-({class_names[2]}):\n", cov_matrix_class_2)


# 5. Реалізувати розрахунок обернених матриць коваріації.
print("\nInverse covariance matrices for each class in the training data:".capitalize())
inv_cov_matrix_class_0 = np.linalg.inv(cov_matrix_class_0)
inv_cov_matrix_class_1 = np.linalg.inv(cov_matrix_class_1)
inv_cov_matrix_class_2 = np.linalg.inv(cov_matrix_class_2)
print(f"\n\tclass 0-({class_names[0]}):\n", inv_cov_matrix_class_0)
print(f"\n\tclass 1-({class_names[1]}):\n", inv_cov_matrix_class_1)
print(f"\n\tclass 2-({class_names[2]}):\n", inv_cov_matrix_class_2)

# Calculate log-determinants for QDA discriminant function to avoid numerical instability issues with small determinants
print("\nLog-determinants of covariance matrices for each class in the training data:")
log_det_cov_class_0 = np.log(np.linalg.det(cov_matrix_class_0))
log_det_cov_class_1 = np.log(np.linalg.det(cov_matrix_class_1))
log_det_cov_class_2 = np.log(np.linalg.det(cov_matrix_class_2))
print(f"\tclass 0-({class_names[0]}):", log_det_cov_class_0)
print(f"\tclass 1-({class_names[1]}):", log_det_cov_class_1)
print(f"\tclass 2-({class_names[2]}):", log_det_cov_class_2)


# 6. Обчислити апріорні імовірності кожного класа у тренувальних даних.
print("\nPrior probabilities for each class in the training data:")
prior_prob_class_0 = np.mean(y_train == 0)
prior_prob_class_1 = np.mean(y_train == 1)
prior_prob_class_2 = np.mean(y_train == 2)
print(f"\tclass 0-({class_names[0]}):", prior_prob_class_0)
print(f"\tclass 1-({class_names[1]}):", prior_prob_class_1)
print(f"\tclass 2-({class_names[2]}):", prior_prob_class_2)


# 7. Реалізувати функцію обчислення значень дискримінантної функції для одного рядка (вектора) тестових даних.
def discriminant_function(x, mean_vector, inv_cov_matrix, prior_prob, log_det_cov):
    """Calculates the discriminant function value for a single test data vector."""
    diff = x - mean_vector
    return -0.5 * log_det_cov - 0.5 * np.dot(diff.T, np.dot(inv_cov_matrix, diff)) + np.log(prior_prob)


# 8. Реалізувати функцію обчислення значень дискримінантної функції та імовірностей приналежності кожному класу для всієї матриці тестових даних.
def custom_predict(X, means, inv_covs, log_dets, priors):
    """Calculates the discriminant function values and class membership probabilities for the entire test data matrix."""
    predictions = []
    for x in X:
        g0 = discriminant_function(x, means[0], inv_covs[0], priors[0], log_dets[0])
        g1 = discriminant_function(x, means[1], inv_covs[1], priors[1], log_dets[1])
        g2 = discriminant_function(x, means[2], inv_covs[2], priors[2], log_dets[2])

        predicted_class = np.argmax([g0, g1, g2])
        predictions.append(predicted_class)

    return np.array(predictions)


# 9. Виконати прогнозування на тестових даних за допомогою функії QuadraticDiscriminantAnalysis() бібліотеки sklearn та порівняти отримані результати.

# Prepare parameters for custom predict function
mean_vector_class_0 = np.mean(class_0_features, axis=0)
mean_vector_class_1 = np.mean(class_1_features, axis=0)
mean_vector_class_2 = np.mean(class_2_features, axis=0)

means = [mean_vector_class_0, mean_vector_class_1, mean_vector_class_2]
inv_covs = [inv_cov_matrix_class_0, inv_cov_matrix_class_1, inv_cov_matrix_class_2]
log_dets = [log_det_cov_class_0, log_det_cov_class_1, log_det_cov_class_2]
priors = [prior_prob_class_0, prior_prob_class_1, prior_prob_class_2]

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred_qda = qda.predict(X_test)
y_pred_custom = custom_predict(X_test, means, inv_covs, log_dets, priors)

# 10. Зробити висновок про ступінь схожості результатів, отриманих власною функцією та бібліотекою sklearn.
print("\nComparison of Custom QDA and Sklearn QDA Predictions:")
print("\tCustom: ", y_pred_custom)
print("\tSklearn:", y_pred_qda)

print("\nQDA Accuracy Comparison:")
custom_prediction_accuracy = np.mean(y_pred_custom == y_test)
sklearn_prediction_accuracy = np.mean(y_pred_qda == y_test)
print("\tCustom: ", custom_prediction_accuracy)
print("\tSklearny:", sklearn_prediction_accuracy)

print("\nConfusion Matrix Comparison:")
cm_custom = confusion_matrix(y_test, y_pred_custom)
print("\tConfusion Matrix for Custom QDA:\n", cm_custom)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_custom, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Custom QDA")
plt.show(block=False)

cm_sklearn = confusion_matrix(y_test, y_pred_qda)
print("\tConfusion Matrix for Sklearn QDA:\n", cm_sklearn)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_sklearn, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Oranges)
plt.title("Confusion Matrix for Sklearn QDA")
plt.show(block=False)

print("\nConclusion:")
# mention train/test split, class distribution, and normality of features
print(
    f"\tThe training set consisted of { X_train.shape[0]/(X_train.shape[0] + X_test.shape[0]) * 100:.2f}% of the data, while the test set consisted of { X_test.shape[0]/(X_train.shape[0] + X_test.shape[0]) * 100:.2f}%."
)
print("\tThe class distribution was balanced in both sets, with each class having an equal number of samples.")
print("\tThe Shapiro-Wilk test indicated that in the most cases, the features were approximately normally distributed within each class.")
print(
    f"\n\n\tThe custom implementation of QDA achieved an accuracy of {custom_prediction_accuracy * 100:.2f}% compared to {sklearn_prediction_accuracy * 100:.2f}% for the sklearn implementation."
)
print("\tThe confusion matrices show that both implementations made similar predictions.")
