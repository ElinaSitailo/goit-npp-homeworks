from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler


def load_data_to_dataframe() -> pd.DataFrame:
    data = load_breast_cancer()

    print(data.DESCR[:1500])

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target  # цільова змінна в DataFrame
    df["target"] = df["target"].map({0: "malignant", 1: "benign"})  # Замінюємо числові значення цільової змінної на текстові мітки класів

    return df


def show_feature_summary(df):
    print(f"  Mean (The first 5 features): {df.iloc[:, :5].mean().values.round(3)}")
    print(f"  Std  (The first 5 features): {df.iloc[:, :5].std().values.round(3)}")


def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    features = df.columns[:-1]  # крім 'target'
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])  # Стандартизуємо ознаки, замінюючи їх на стандартизовані значення
    return df


def visualize_pairplot(
    df: pd.DataFrame,
    max_features: int = 5,
) -> None:

    feature_columns = [column for column in df.columns if column != "target"]  # Отримуємо список стовпців, які є ознаками (виключаючи 'target')
    selected_features = feature_columns[:max_features]  # Вибираємо перші max_features ознак для візуалізації
    plot_df = df[selected_features + ["target"]]  # Створюємо DataFrame для візуалізації, який містить вибрані ознаки та цільову змінну

    sns.pairplot(plot_df, hue="target", palette="Set1")
    plt.suptitle("Pairplot of Breast Cancer Dataset", y=1.02)  # заголовок графіка з невеликим відступом вгору
    plt.show()


# Обчисліть матриці відстаней.
# Використовуйте алгоритми та функції з конспекту для обчислення матриці відстаней для різних метрик:
# cityblock, cosine, euclidean, l1, manhattan.
def compute_distance_matrices(df: pd.DataFrame) -> Dict[str, np.ndarray]:

    std_data_without_target = df.drop("target", axis=1).values

    metrics = ["cityblock", "cosine", "euclidean", "l1", "manhattan"]

    distance_matrices: Dict[str, np.ndarray] = {}
    for metric in metrics:
        dist_matrix = pairwise_distances(std_data_without_target, metric=metric)  # отримуємо матрицю відстаней для кожної метрики та зберігаємо її у словнику
        distance_matrices[metric] = dist_matrix
        print(f"\n METRIC {metric}")
        print(f"   MATRIX SHAPE: {dist_matrix.shape}")
        print(f"   MIN (excluding 0): {dist_matrix[dist_matrix > 0].min():.4f}")
        print(f"   MAX: {dist_matrix.max():.4f}")
        print(f"   MEAN: {dist_matrix.mean():.4f}")
        print(f"   MEDIAN: {np.median(dist_matrix):.4f}")

    return distance_matrices


def visualize_distance_matrices(df: pd.DataFrame, distance_matrices: Dict[str, np.ndarray]) -> None:

    fig, axes = plt.subplots(1, len(distance_matrices), figsize=(20, 5))  # фігура з підграфіками для кожної матриці відстаней

    for i, (name, matrix) in enumerate(distance_matrices.items()):

        sns.heatmap(matrix, ax=axes[i], cmap="viridis", cbar=True, cbar_kws={"label": "Distance"})

        n_malignant = (df["target"] == "malignant").sum()
        axes[i].hlines(n_malignant, *axes[i].get_xlim(), colors="red", linestyles="dashed", linewidth=2)  # Додаємо горизонтальну пунктирну лінію для відділення зразків "malignant"
        axes[i].vlines(n_malignant, *axes[i].get_ylim(), colors="red", linestyles="dashed", linewidth=2)  # Додаємо вертикальну пунктирну лінію для відділення зразків "malignant"

        axes[i].set_title(f"Distance Matrix - {name}")

    plt.show()


def main() -> None:
    df = load_data_to_dataframe()
    print("\n\n", "=" * 70, "\n\t\tData Information:", "\n", "=" * 70)
    print(df.info())

    print("\n", "=" * 70, "\n\t\tDescriptive Statistics:", "\n", "=" * 70)
    print(df.describe())

    print(f"\nThe first 5 rows of the dataset:")
    print(df.head())

    print(f"\nTarget Distribution:")
    print(df["target"].value_counts())

    print("\n\n", "=" * 70, "\n\t\tSTANDARDIZATION:", "\n", "=" * 70)
    print("\nBEFORE Standardization:")
    show_feature_summary(df)

    df = standardize_data(df)

    print("\nAFTER Standardization:")
    show_feature_summary(df)
    print("\nThe first 5 rows of the dataset:")
    print(df.head())

    visualize_pairplot(df)

    print("\n\n", "=" * 70, "\n\t\tDISTANCE METRICES:", "\n", "=" * 70)
    distance_matrices = compute_distance_matrices(df)

    print("\n\nComparing distance matrices for uniqueness:")
    unique_matrices = {}
    for metric, matrix in distance_matrices.items():
        is_unique = True
        for other_metric, other_matrix in distance_matrices.items():
            if metric != other_metric and np.allclose(matrix, other_matrix):
                is_unique = False
                print(f"  {metric} distance matrix is the same as {other_metric}")

                # додаємо першу за алфавітом метрику, яка має однакову матрицю, і припиняємо порівняння для цієї метрики
                if other_metric < metric:
                    unique_matrices[other_metric] = other_matrix
                break
        if is_unique:
            unique_matrices[metric] = matrix

    print(f"\n\nUnique distance matrices: {list(unique_matrices.keys())}")

    visualize_distance_matrices(df, unique_matrices)

    print("\n\n", "=" * 70, "\n\t\tDATA ANALYSIS CONCLUSION:", "\n", "=" * 70)

    print(
        "Dataset contains 569 samples with 30 features each, and a binary target variable indicating malignant or benign tumors. "
        "\nThe dataset is well-balanced with 212 (37%) malignant and 357 (63%) benign samples. "
        "\nThe features have varying scales, with means and standard deviations that differ significantly across features."
        "\nBefore standardization, the features have different scales, which can affect distance-based analyses. "
        "\nAfter standardization, all features have a mean of approximately 0 and a standard deviation of 1, which is crucial for distance-based analyses. "
        "\nThe pairplot shows clear separation between malignant and benign samples in the first few features, indicating that these features are informative for classification."
        "\nThe distance matrices computed using different metrics (cityblock, cosine, euclidean) show unique patterns of distances between samples. "
        "\nThe heatmaps of the distance matrices reveal that samples of the same class (malignant or benign) tend to have smaller distances between them compared to samples of different classes, which is expected in a well-structured dataset."
    )
    # порівняйте ефективність різних метрик відстаней для даного набору даних.
    print(
        "\n Comparing the effectiveness of different distance metrics for this dataset:"
        "\nThe cityblock (Manhattan) and euclidean distances capture the absolute differences between samples, which can be useful for datasets where the magnitude of feature differences is important. "
        "\nThe cosine distance, on the other hand, emphasizes the similarity in the direction of the feature vectors, which can be useful when the orientation of the data points is more important than their magnitude."
    )


if __name__ == "__main__":
    main()
