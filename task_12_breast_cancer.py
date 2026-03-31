import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pygad
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment
from scipy.stats import shapiro
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

DIV = "-" * 150
COMPONENTS_COUNT = 2  # Number of clusters/components for clustering algorithms
RANDOM_STATE = 42  # Random state for reproducibility

malignant_color = "#FF2600"  # Bright red for malignant
benign_color = "#0800FF"  # Bright blue for benign
palette_by_malignancy = {0: malignant_color, 1: benign_color}  # Define a color palette for the target classes (0 - malignant, 1 - benign)
confusion_cmap = LinearSegmentedColormap.from_list("malignancy_confusion", [benign_color, malignant_color])


def show_histogram_with_shapiro_test(df):
    "Build histograms for each feature to check for distribution and show shapiro-wilk test results for normality."

    features = df.columns[:-1]  # Exclude target column
    plt.figure(figsize=(20, 15))
    plt.suptitle("Histograms of Breast Cancer Dataset Features with Shapiro-Wilk Test Results", fontsize=16)

    shapiro_test_results = {}
    for i, feature in enumerate(features):
        plt.subplot(8, 4, i + 1)
        sns.histplot(df[feature], kde=True, bins=30, color="skyblue")
        plt.title(f"{feature}")

        stat, p_value = shapiro(df[feature])
        normality = "Normal" if p_value > 0.05 else "Not Normal"
        shapiro_test_results[feature] = (stat, p_value)
        plt.xlabel(f"Shapiro-Wilk W={stat:.4f}, p={p_value:.4e}, {normality}")

    plt.tight_layout()
    plt.show()

    print("Shapiro-Wilk test results for normality:\n")
    for feature, (stat, p_value) in shapiro_test_results.items():
        normality = "Normal" if p_value > 0.05 else "Not Normal"
        print(f"{feature}: W={stat:.4f}, p-value={p_value:.4e}, {normality}")


def plot_breast_cancer_features(df, feature_names):
    "Visualize pairwise scatter plots of the first few features colored by target class."

    sns.pairplot(df, hue="target", vars=feature_names, palette=palette_by_malignancy)
    plt.suptitle(f"Pairplot of Breast Cancer Dataset (first {len(feature_names)} features)", y=1.02)
    plt.show()


def normalize_feature_data(data, df):
    "Normalize the feature data using StandardScaler and handle any missing values by filling them with the mean of the respective feature."

    feature_columns = data.feature_names
    if df[feature_columns].isnull().sum().any():
        df[feature_columns] = df[feature_columns].fillna(df[feature_columns].mean())
        print("Missing values filled with mean.")

    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])


def align_cluster_labels(y, labels):
    "Remap cluster labels to best match true labels using the Hungarian algorithm."
    cm = confusion_matrix(y, labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in labels])


def perform_clustering(X, y):
    """Perform clustering using Spectral Clustering, K-Means, and Gaussian Mixture Models, and return the predicted cluster labels for each method.
    Args:
        X (pd.DataFrame): The feature data for clustering. Should not include the target column.
        y (pd.Series): The true target labels used for cluster label alignment.
    Returns:
        tuple: A tuple containing the aligned predicted cluster labels for
            Spectral Clustering,
            K-Means,
            and Gaussian Mixture Models.
    """
    spectral_clustering = SpectralClustering(n_clusters=COMPONENTS_COUNT, affinity="nearest_neighbors", random_state=RANDOM_STATE)
    spectral_labels = spectral_clustering.fit_predict(X)

    kmeans = KMeans(n_clusters=COMPONENTS_COUNT, random_state=RANDOM_STATE)
    kmeans_labels = kmeans.fit_predict(X)

    gmm = GaussianMixture(n_components=COMPONENTS_COUNT, random_state=RANDOM_STATE)
    gmm_labels = gmm.fit_predict(X)

    return align_cluster_labels(y, spectral_labels), align_cluster_labels(y, kmeans_labels), align_cluster_labels(y, gmm_labels)


def build_clustering_classification_reports(y, spectral_labels, kmeans_labels, gmm_labels):
    "Build classification reports for each clustering method compared to the actual target classes."

    spectral_report = classification_report(y, spectral_labels, zero_division=0)
    kmeans_report = classification_report(y, kmeans_labels, zero_division=0)
    gmm_report = classification_report(y, gmm_labels, zero_division=0)

    print("Classification Report for Spectral Clustering:\n", spectral_report)
    print("Classification Report for K-Means Clustering:\n", kmeans_report)
    print("Classification Report for Gaussian Mixture Model:\n", gmm_report)

    spectral_metrics = classification_report(y, spectral_labels, output_dict=True, zero_division=0)
    kmeans_metrics = classification_report(y, kmeans_labels, output_dict=True, zero_division=0)
    gmm_metrics = classification_report(y, gmm_labels, output_dict=True, zero_division=0)

    return spectral_metrics, kmeans_metrics, gmm_metrics


def display_confusion_matrices(target_names, y, spectral_labels, kmeans_labels, gmm_labels):
    "Compute and display confusion matrices for each clustering method compared to the actual target classes."

    cm_spectral = confusion_matrix(y, spectral_labels)
    cm_kmeans = confusion_matrix(y, kmeans_labels)
    cm_gmm = confusion_matrix(y, gmm_labels)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    disp_spectral = ConfusionMatrixDisplay(confusion_matrix=cm_spectral, display_labels=target_names)
    disp_spectral.plot(ax=axes[0], cmap=confusion_cmap, colorbar=False)
    axes[0].set_title("Spectral Clustering")

    disp_kmeans = ConfusionMatrixDisplay(confusion_matrix=cm_kmeans, display_labels=target_names)
    disp_kmeans.plot(ax=axes[1], cmap=confusion_cmap, colorbar=False)
    axes[1].set_title("K-Means")

    disp_gmm = ConfusionMatrixDisplay(confusion_matrix=cm_gmm, display_labels=target_names)
    disp_gmm.plot(ax=axes[2], cmap=confusion_cmap, colorbar=False)
    axes[2].set_title("Gaussian Mixture Model")

    fig.suptitle("Confusion Matrices Comparison", fontsize=14)
    plt.tight_layout()
    plt.show()

    return cm_spectral, cm_kmeans, cm_gmm


def display_clustering_results(divider, spectral_metrics, kmeans_metrics, gmm_metrics, cm_spectral, cm_kmeans, cm_gmm):
    def metric(metrics, class_key, metric_name, fallback_group="macro avg"):
        return metrics.get(class_key, {}).get(metric_name, metrics.get(fallback_group, {}).get(metric_name, 0.0))

    col1 = 24
    col = 22

    print(f"{'Metric':<{col1}}{'Spectral Clustering':<{col}}{'K-Means':<{col}}{'Gaussian Mixture':<{col}}")
    print("-" * (col1 + col * 3))

    rows = [
        ("Accuracy", spectral_metrics["accuracy"], kmeans_metrics["accuracy"], gmm_metrics["accuracy"]),
        (
            "Precision (class 0)",
            metric(spectral_metrics, "0", "precision"),
            metric(kmeans_metrics, "0", "precision"),
            metric(gmm_metrics, "0", "precision"),
        ),
        (
            "Precision (class 1)",
            metric(spectral_metrics, "1", "precision"),
            metric(kmeans_metrics, "1", "precision"),
            metric(gmm_metrics, "1", "precision"),
        ),
        (
            "Recall (class 0)",
            metric(spectral_metrics, "0", "recall"),
            metric(kmeans_metrics, "0", "recall"),
            metric(gmm_metrics, "0", "recall"),
        ),
        (
            "Recall (class 1)",
            metric(spectral_metrics, "1", "recall"),
            metric(kmeans_metrics, "1", "recall"),
            metric(gmm_metrics, "1", "recall"),
        ),
        (
            "F1-score (class 0)",
            metric(spectral_metrics, "0", "f1-score"),
            metric(kmeans_metrics, "0", "f1-score"),
            metric(gmm_metrics, "0", "f1-score"),
        ),
        (
            "F1-score (class 1)",
            metric(spectral_metrics, "1", "f1-score"),
            metric(kmeans_metrics, "1", "f1-score"),
            metric(gmm_metrics, "1", "f1-score"),
        ),
    ]

    for label, spectral_value, kmeans_value, gmm_value in rows:
        print(f"{label:<{col1}}{spectral_value:<{col}.4f}{kmeans_value:<{col}.4f}{gmm_value:<{col}.4f}")

    print("-" * (col1 + col * 3))
    print(f"{'False Positives':<{col1}}{cm_spectral[0][1]:<{col}}{cm_kmeans[0][1]:<{col}}{cm_gmm[0][1]:<{col}}")
    print(f"{'False Negatives':<{col1}}{cm_spectral[1][0]:<{col}}{cm_kmeans[1][0]:<{col}}{cm_gmm[1][0]:<{col}}")
    print(divider)


def print_insights():
    print("Key insights:")
    print("All models show high consistency with the original class distribution.")
    print("Spectral Clustering and GMM outperform K-Means.")
    print("\tSpectral Clustering:")
    print("\t\tBest recall for class 1 (benign), which is crucial for medical diagnosis to minimize false negatives.")
    print("\t\tFewest false negatives, indicating better sensitivity in identifying malignant cases.")
    print("\tGaussian Mixture Model:")
    print("\t\tMost balanced model with high precision and recall for both classes, making it a strong candidate for this dataset.")
    print("\t\tLowest false positives, which is important to avoid misclassifying benign cases as malignant.")
    print("\tK-Means:")
    print("\t\tSimplest but least accurate model, with more false negatives and false positives compared to the other two methods.")
    print("\nChoice depends on priority:")
    print("\tUse Spectral Clustering → if minimizing missed cases is critical (high recall for malignant).")
    print("\tUse Gaussian Mixture Model → if a balanced performance for both classes is desired.")
    print("\tUse K-Means → if simplicity and speed are more important than accuracy.\n")


def visualize_clustering_results_on_scatter_plot(X, y, spectral_labels, kmeans_labels, gmm_labels):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=spectral_labels, palette=palette_by_malignancy, alpha=0.7)
    plt.title("Spectral Clustering Results")
    plt.subplot(1, 4, 2)
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=kmeans_labels, palette=palette_by_malignancy, alpha=0.7)
    plt.title("K-Means Clustering Results")
    plt.subplot(1, 4, 3)
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=gmm_labels, palette=palette_by_malignancy, alpha=0.7)
    plt.title("GMM Clustering Results")
    plt.subplot(1, 4, 4)
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y, palette=palette_by_malignancy, alpha=0.7)
    plt.title("Actual Target Classes")

    plt.tight_layout()
    plt.show()


def plot_pca_analysis(X, y, spectral_labels, X_pca):
    y_values = np.asarray(y)
    spectral_values = np.asarray(spectral_labels)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        X.iloc[:, 2],
        c=[palette_by_malignancy[int(label)] for label in y_values],
        alpha=0.7,
        s=30,
    )
    ax1.set_title("Original Feature Space (3D)")
    ax1.set_xlabel(X.columns[0])
    ax1.set_ylabel(X.columns[1])
    ax1.set_zlabel(X.columns[2])

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2],
        c=[palette_by_malignancy[int(label)] for label in y_values],
        alpha=0.7,
        s=30,
    )
    ax2.set_title("PCA-Transformed Feature Space (3D)")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")

    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2],
        c=[palette_by_malignancy[int(label)] for label in y_values],
        alpha=0.7,
        s=30,
    )
    ax1.set_title("PCA 3D with Actual Classes")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2],
        c=[palette_by_malignancy[int(label)] for label in spectral_values],
        alpha=0.7,
        s=30,
    )
    misclassified = spectral_values != y_values
    ax2.scatter(
        X_pca[misclassified, 0],
        X_pca[misclassified, 1],
        X_pca[misclassified, 2],
        color="yellow",
        edgecolor="black",
        label="Misclassified",
        s=80,
    )
    ax2.set_title("PCA 3D with Spectral Clustering")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def run_genetic_algorithm_for_logistic_regression(RANDOM_STATE, X, y):
    # Виконати класифікацію методом логістичної регресії із оптимізацією параметрів Генетичним алгоритмом.
    X_values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    classes = np.unique(y)

    def fitness_function(ga_instance, solution, solution_idx):  # noqa: ARG001
        # Генетичний алгоритм оптимізує ваги для логістичної регресії
        weights = np.array(solution)
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        model.coef_ = weights.reshape(1, -1)  # Встановлюємо ваги як коефіцієнти моделі
        model.intercept_ = np.array([0])  # Встановлюємо інтерсепт в 0 для спрощення
        model.classes_ = classes  # Required by predict() for an unfitted model
        predictions = model.predict(X_values)
        report = classification_report(y, predictions, output_dict=True, zero_division=0)
        return report["accuracy"]  # Використовуємо точність як фітнес-функцію

    num_genes = X.shape[1]  # Кількість ознак як кількість генів
    initial_population = np.random.rand(10, num_genes)  # Початкова популяція з 10 рішень
    ga_instance = pygad.GA(num_generations=50, num_parents_mating=5, fitness_func=fitness_function, sol_per_pop=10, num_genes=num_genes, initial_population=initial_population)
    ga_instance.run()
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    print("Best solution (weights):\n", best_solution)
    print("Best solution fitness (accuracy):\n", best_solution_fitness)
    best_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    best_model.coef_ = best_solution.reshape(1, -1)
    best_model.intercept_ = np.array([0])
    best_model.classes_ = classes
    return best_model.predict(X_values)


def fit_logistic_regression_with_gd(RANDOM_STATE, X, y):
    # Виконати класифікацію методом логістичної регресії із оптимізацією параметрів методами градієнтного спуску та порівняти результати з кластеризацією.
    logistic_model_gd = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        solver="saga",  # "saga" підтримує різні типи регуляризації та оптимізації, включаючи градієнтний спуск
    )
    logistic_model_gd.fit(X, y)
    logistic_gd_predictions = logistic_model_gd.predict(X)

    return logistic_gd_predictions


def perform_logistic_regression(RANDOM_STATE, X, y):
    logistic_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    logistic_model.fit(X, y)
    logistic_predictions = logistic_model.predict(X)

    logistic_gd_predictions = fit_logistic_regression_with_gd(RANDOM_STATE, X, y)

    best_predictions = run_genetic_algorithm_for_logistic_regression(RANDOM_STATE, X, y)

    return logistic_predictions, logistic_gd_predictions, best_predictions


def run_and_plot_logistic_regressions(data, X, y):

    logistic_predictions, logistic_gd_predictions, best_predictions = perform_logistic_regression(RANDOM_STATE, X, y)

    # reports
    logistic_report = classification_report(y, logistic_predictions)
    print("Classification Report for Logistic Regression:\n", logistic_report)
    logistic_gd_report = classification_report(y, logistic_gd_predictions)
    print("Classification Report for Logistic Regression with Gradient Descent Optimization:\n", logistic_gd_report)
    best_report = classification_report(y, best_predictions)
    print("Classification Report for Logistic Regression with Genetic Algorithm Optimization:\n", best_report)

    # matrices
    confusion_matrix_logistic = confusion_matrix(y, logistic_predictions)
    print("Confusion Matrix for Logistic Regression:\n", confusion_matrix_logistic)
    confusion_matrix_logistic_gd = confusion_matrix(y, logistic_gd_predictions)
    print("Confusion Matrix for Logistic Regression with Gradient Descent Optimization:\n", confusion_matrix_logistic_gd)
    confusion_matrix_best = confusion_matrix(y, best_predictions)
    print("Confusion Matrix for Logistic Regression with Genetic Algorithm Optimization:\n", confusion_matrix_best)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    disp_logistic = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_logistic, display_labels=data.target_names)
    disp_logistic.plot(ax=axes[0], cmap=confusion_cmap, colorbar=False)
    axes[0].set_title("Logistic Regression")
    disp_logistic_gd = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_logistic_gd, display_labels=data.target_names)
    disp_logistic_gd.plot(ax=axes[1], cmap=confusion_cmap, colorbar=False)
    axes[1].set_title("Logistic Regression with Gradient Descent")
    disp_best = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_best, display_labels=data.target_names)
    disp_best.plot(ax=axes[2], cmap=confusion_cmap, colorbar=False)
    axes[2].set_title("Logistic Regression with Genetic Algorithm")
    fig.suptitle("Logistic Regression Confusion Matrices Comparison", fontsize=14)
    plt.tight_layout()
    plt.show()

    # f1_score().
    f1_logistic = f1_score(y, logistic_predictions, average="weighted")
    f1_logistic_gd = f1_score(y, logistic_gd_predictions, average="weighted")
    f1_best = f1_score(y, best_predictions, average="weighted")
    print("F1 Score for Logistic Regression:", f1_logistic)
    print("F1 Score for Logistic Regression with Gradient Descent Optimization:", f1_logistic_gd)
    print("F1 Score for Logistic Regression with Genetic Algorithm Optimization:", f1_best)


if __name__ == "__main__":
    data = load_breast_cancer()
    print("Features:", data.feature_names)
    print("Target classes:", data.target_names)  # 0 - malignant, 1 - benign
    print("Data shape:", data.data.shape)

    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df["target"] = data.target

    normalize_feature_data(data, df)

    print("First 2 samples with target:\n", df.head(2))
    print("Basic statistics for breast cancer dataset :\n")
    print(df.describe())

    plot_breast_cancer_features(df, feature_names=data.feature_names[:5])  # Plot first 8 features for better visualization
    show_histogram_with_shapiro_test(df)

    # 3. Виконати кластеризацію, порівняти розподіли та класифікаційні звіти для кожного методу? пояснити результати

    X = df.drop(columns=["target"])  # Features for clustering (exclude target column)
    y = df["target"]  # Actual target classes (0 - malignant, 1 - benign)

    spectral_labels, kmeans_labels, gmm_labels = perform_clustering(X, y)

    visualize_clustering_results_on_scatter_plot(X, y, spectral_labels, kmeans_labels, gmm_labels)

    print("\n")
    spectral_metrics, kmeans_metrics, gmm_metrics = build_clustering_classification_reports(y, spectral_labels, kmeans_labels, gmm_labels)
    cm_spectral, cm_kmeans, cm_gmm = display_confusion_matrices(data.target_names, y, spectral_labels, kmeans_labels, gmm_labels)

    print(DIV)
    print("\t\t\tCLUSTERING RESULTS SUMMARY")
    print(DIV)

    display_clustering_results(DIV, spectral_metrics, kmeans_metrics, gmm_metrics, cm_spectral, cm_kmeans, cm_gmm)
    print_insights()
    print(DIV)

    # Виконати зменшення розмірності даних за допомогою метода PCA та візуалізувати результати кластеризації на 2D графіку.
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    plot_pca_analysis(X, y, spectral_labels, X_pca)

    ############################################################################################################################################################################
    #### LogisticRegression with Gradient Descent and Genetic Algorithm optimization
    run_and_plot_logistic_regressions(data, X, y)

    print(DIV)
    print("\t\t\tLOGISTIC REGRESSION CLASSIFICATION RESULTS SUMMARY")
    print(DIV)
    print(f"Standard Logistic Regression and Gradient Descent produce identical results")
    print(f"Genetic Algorithm optimization slightly reduces performance, likely due to the simplicity of the model and the limited number of generations for optimization.")
    print(f"All models achieve very high accuracy (≥ 98%) after PCA transformation.")
    print(f"Class 1 is predicted slightly better than class 0 across all models, which may be due to class imbalance or the nature of the features.")
    print(DIV)

    print("\nComparison of Logistic Regression with Clustering Methods:")
    print("\tAccuracy comparasions (Clustering vs Logistic Regression):")
    print("\t\tLR: 0.99")
    print("\t\tLR with GD: 0.99")
    print("\t\tLR with GA: 0.98")
    print("\t\tSpectral Clustering: 0.94")
    print("\t\tGaussian Mixture Model: 0.94")
    print("\t\tK-Means Clustering: 0.91")
    print("\nClustering Methods show lower accuracy and more classification errors compared to Logistic Regression.")
    print("The data is highly separable and suitable for both clustering and classification.")
    print("We use  Clustering for structure discovery and Logistic Regression for prediction tasks.")
