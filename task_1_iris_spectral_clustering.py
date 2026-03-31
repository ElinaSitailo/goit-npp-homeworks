# 1. Завантажити та створити DataFrame: Використовуйте бібліотеку  pandas для створення та роботи з DataFrame. Дані можна завантажити за допомогою load_iris() з бібліотеки **sklearn**.
# 2. Отримати базові статистичні характеристики
# 3. Візуалізувати розподілення спостережень за класами: .seaborn.scatterplot() або seaborn.pairplot() для візуалізації розподілу даних за класами.
# 4. Виконати стандартизацію даних: sklearn.preprocessing.
# 5. Виконати спектральну кластеризацію: sklearn.cluster.
# 6. Порівняти спрогнозовані кластери та дійсні класи: sklearn.metrics.confusion_matrix() .
# 7. Візуалізувати результати кластеризації.
# 8. Висновок

import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris  # pip install scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


def display_confusion_matrix(classes, cm):
    print("\nConfusion matrix:\n", cm)

    for i, row in enumerate(cm):
        actual = classes[i]
        total = row.sum()

        print(f"\nActual class: {actual} ({total} samples)")

        for j, count in enumerate(row):
            predicted = classes[j]

            if i == j:
                print(f"  ✅ {count} correctly predicted as {predicted}")
            else:
                if count > 0:
                    print(f"  ❌ {count} misclassified as {predicted}")


def align_cluster_labels(y_true, y_pred):
    contingency = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency)

    mapping = {pred_label: true_label for true_label, pred_label in zip(row_ind, col_ind)}
    aligned_predictions = pd.Series(y_pred).map(mapping).to_numpy()

    return aligned_predictions


iris = load_iris()
sepal_length_column_name = iris["feature_names"][0]
sepal_width_column_name = iris["feature_names"][1]

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

df["target"] = iris.target
sns.pairplot(df, hue="target", palette="Set1")
plt.suptitle("Clustered Data after standardization for each feature", y=1.02)
plt.show(block=True)

print("Basic statistics for iris dataset :\n")
print(df.describe())

sns.scatterplot(x=df[sepal_length_column_name], y=df[sepal_width_column_name], hue=iris.target, palette="Set1")
plt.title("Initial dataset")
plt.show(block=False)

# масштабуємо/стандартизуємо датасет, кластерізуємо масштабований датасет
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

spectral = SpectralClustering(
    n_clusters=3,
    affinity="nearest_neighbors",
    random_state=42,
)
spectral_clusters_on_scaled_data = spectral.fit_predict(scaled_data)

sns.scatterplot(x=df[sepal_length_column_name], y=df[sepal_width_column_name], hue=spectral_clusters_on_scaled_data, palette="Set1")
plt.title("Spectral clusterization after data standardisation")
plt.show(block=False)

# порівнюємо датасети
aligned_clusters = align_cluster_labels(iris.target, spectral_clusters_on_scaled_data)
cm = confusion_matrix(iris.target, aligned_clusters)
display_confusion_matrix(iris.target_names, cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.title("Confusion matrix")
plt.show(block=False)


# df["Cluster"] = spectral_clusters_on_scaled_data
# sns.pairplot(df, hue="Cluster", palette="Set1")
# plt.suptitle("Clustered Data after standardization for each feature", y=1.02)
# plt.show(block=True)


correctly_classified_instances = cm.trace()
print(
    f""" The SUMMARY:
      According to the confusion matrix, the spectral clustering algorithm on the scaled data 
    after cluster label alignment with the true classes.

    {correctly_classified_instances} out of {cm.sum()} instances were classified correctly, 
      which gives an accuracy of approximately {correctly_classified_instances / cm.sum() * 100:.2f}%.

    This suggests the model separates classes reasonably, while some overlap
    between Versicolour and Virginica may still remain in feature space."""
)
