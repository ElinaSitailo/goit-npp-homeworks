import base64
import math
import os
import re
import time
import urllib.request
import zipfile
from urllib.error import HTTPError, URLError

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy.fft import fft
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import probplot
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.mixture import GaussianMixture


ARCHIVE_DOWNLOAD_URL = "https://github.com/goitacademy/NUMERICAL-PROGRAMMING-IN-PYTHON/blob/main/WorldHappinessReport.zip"
DATA_FILE_NAME = "combined_data.csv"
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")


def _to_raw_github_url(url):
    """Convert GitHub blob URLs to raw content URLs."""

    marker = "github.com/"
    blob_marker = "/blob/"

    if marker in url and blob_marker in url:
        _, rest = url.split(marker, 1)
        repo_and_path = rest.split(blob_marker, 1)
        if len(repo_and_path) == 2:
            repo_part, branch_and_file = repo_and_path
            return "https://raw.githubusercontent.com/" f"{repo_part}/{branch_and_file}"

    return url


def download_file(url, username, password, max_retries=5):
    """Download a file from a URL with optional authentication and retry logic."""

    archive_file_name_without_extension = os.path.splitext(os.path.basename(url))[0]
    save_path = f"{archive_file_name_without_extension}.zip"

    download_url = _to_raw_github_url(url)
    headers = {
        "User-Agent": "goit-npp-hw-7-downloader/1.0",
        "Accept": "application/octet-stream",
    }

    if username and password:
        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {token}"

    for attempt in range(1, max_retries + 1):
        try:
            print("Downloading file from " f"{download_url} to {save_path} " f"(attempt {attempt}/{max_retries})...")
            request = urllib.request.Request(download_url, headers=headers)

            with urllib.request.urlopen(request, timeout=60) as response, open(save_path, "wb") as output_file:
                output_file.write(response.read())

            print("Download completed successfully.")
            return True

        except HTTPError as error:
            if error.code == 429 and attempt < max_retries:
                retry_after = error.headers.get("Retry-After")
                wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else min(2**attempt, 30)
                print("HTTP 429: Too Many Requests. " f"Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
                continue

            print(f"HTTP error {error.code}: {error.reason}")
            return False

        except URLError as error:
            if attempt < max_retries:
                wait_seconds = min(2**attempt, 30)
                print(f"Network error: {error.reason}. " f"Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
                continue

            print(f"Network error: {error.reason}")
            return False

        except OSError as error:
            print(f"File I/O error: {error}")
            return False

    return False


def get_dataframe(url, github_username, github_token, data_filename):
    """Download a ZIP archive from a URL, extract it, and load a CSV file into a DataFrame."""

    archive_file_name_without_extension = os.path.splitext(os.path.basename(url))[0]
    archive_file_path = f"{archive_file_name_without_extension}.zip"

    is_archive_found = False
    if os.path.exists(archive_file_path):
        print(f"File {archive_file_path} already exists. Skipping download.")
        is_archive_found = True

    is_archive_loaded = False
    if not is_archive_found:
        is_archive_loaded = download_file(url, github_username, github_token)

    if is_archive_loaded or is_archive_found:

        data_file_full_path = os.path.join(f"./{archive_file_name_without_extension}/", data_filename)

        if os.path.exists(data_file_full_path):
            print(f"File {data_file_full_path} already exists. Skipping extraction.")
        else:
            print(f"Extracting archive to {archive_file_name_without_extension}...")
            with zipfile.ZipFile(archive_file_path, "r") as zip_ref:
                zip_ref.extractall(f"./{archive_file_name_without_extension}/")
            print("Archive extracted successfully.")

        if os.path.exists(data_file_full_path):
            return pd.read_csv(data_file_full_path)

        raise Exception(f"Error: File {data_file_full_path} not found after extraction.")

    raise Exception("Failed to load the archive after multiple attempts.")


def scale_numeric_columns(input_df):
    "Applies StandardScaler and MinMaxScaler to numeric columns in the DataFrame to standardize and normalize the data, respectively."

    numeric_columns = input_df.select_dtypes(include=[np.float64]).columns
    print(f"\nNumeric columns identified for scaling: {numeric_columns.tolist()}")  # Debug statement to list numeric columns

    scaled_df = input_df.copy()  # make a full copy of dataframe to avoid modifying the original one

    scaler = StandardScaler()  # StandardScaler is used to standardize features by removing the mean and scaling to unit variance
    scaled_df[numeric_columns] = scaler.fit_transform(scaled_df[numeric_columns])

    # apply MinMaxScaler to ensure all values are between 0 and 1
    min_max_scaler = MinMaxScaler()  # MinMaxScaler is used to scale features to a given range, typically between 0 and 1
    scaled_df[numeric_columns] = min_max_scaler.fit_transform(scaled_df[numeric_columns])

    print("Numeric columns scaled successfully.")

    return scaled_df


def map_happiness_dataframes(dfs):

    # declare combined_df pd.DataFrame with common columns and their types
    merged_df = pd.DataFrame(
        {
            "Rank": pd.Series(dtype=np.int32),
            "Country": pd.Series(dtype=str),
            "Region": pd.Series(dtype=str),
            "Score": pd.Series(dtype=np.float64),
            "Standard Error": pd.Series(dtype=np.float64),
            "Lower Confidence Interval": pd.Series(dtype=np.float64),
            "Upper Confidence Interval": pd.Series(dtype=np.float64),
            "GDP per capita": pd.Series(dtype=np.float64),
            "Social support": pd.Series(dtype=np.float64),
            "Healthy life expectancy": pd.Series(dtype=np.float64),
            "Freedom to make life choices": pd.Series(dtype=np.float64),
            "Perceptions of corruption": pd.Series(dtype=np.float64),
            "Generosity": pd.Series(dtype=np.float64),
            "Dystopia Residual": pd.Series(dtype=np.float64),
            "Year": pd.Series(dtype=np.int32),
        }
    )

    for df in dfs:
        year = df["Year"].iloc[0]
        print(f"Map DataFrame for year {year}...")

        if year == 2015:
            df_mapped = pd.DataFrame(
                {
                    "Rank": df.get("Happiness Rank"),
                    "Country": df.get("Country"),
                    "Region": df.get("Region"),
                    "Score": df.get("Happiness Score"),
                    "Standard Error": df.get("Standard Error"),
                    "Lower Confidence Interval": np.nan,
                    "Upper Confidence Interval": np.nan,
                    "GDP per capita": df.get("Economy (GDP per Capita)"),
                    "Social support": df.get("Family"),
                    "Healthy life expectancy": df.get("Health (Life Expectancy)"),
                    "Freedom to make life choices": df.get("Freedom"),
                    "Perceptions of corruption": df.get("Trust (Government Corruption)"),
                    "Generosity": df.get("Generosity"),
                    "Dystopia Residual": df.get("Dystopia Residual"),
                    "Year": df.get("Year"),
                }
            )
        elif year == 2016:
            df_mapped = pd.DataFrame(
                {
                    "Rank": df.get("Happiness Rank"),
                    "Country": df.get("Country"),
                    "Region": df.get("Region"),
                    "Score": df.get("Happiness Score"),
                    "Standard Error": np.nan,
                    "Lower Confidence Interval": df.get("Lower Confidence Interval"),
                    "Upper Confidence Interval": df.get("Upper Confidence Interval"),
                    "GDP per capita": df.get("Economy (GDP per Capita)"),
                    "Social support": df.get("Family"),
                    "Healthy life expectancy": df.get("Health (Life Expectancy)"),
                    "Freedom to make life choices": df.get("Freedom"),
                    "Perceptions of corruption": df.get("Trust (Government Corruption)"),
                    "Generosity": df.get("Generosity"),
                    "Dystopia Residual": df.get("Dystopia Residual"),
                    "Year": df.get("Year"),
                }
            )
        elif year == 2017:
            df_mapped = pd.DataFrame(
                {
                    "Rank": df.get("Happiness.Rank"),
                    "Country": df.get("Country"),
                    "Region": df.get("Region"),
                    "Score": df.get("Happiness.Score"),
                    "Standard Error": np.nan,
                    "Lower Confidence Interval": df.get("Whisker.low"),
                    "Upper Confidence Interval": df.get("Whisker.high"),
                    "GDP per capita": df.get("Economy..GDP.per.Capita."),
                    "Social support": df.get("Family"),
                    "Healthy life expectancy": df.get("Health..Life.Expectancy."),
                    "Freedom to make life choices": df.get("Freedom"),
                    "Perceptions of corruption": df.get("Trust..Government.Corruption."),
                    "Generosity": df.get("Generosity"),
                    "Dystopia Residual": df.get("Dystopia.Residual"),
                    "Year": df.get("Year"),
                }
            )
        elif year == 2018 or year == 2019:
            df_mapped = pd.DataFrame(
                {
                    "Rank": df.get("Overall rank"),
                    "Country": df.get("Country or region"),
                    "Region": np.nan,
                    "Score": df.get("Score"),
                    "Standard Error": np.nan,
                    "Lower Confidence Interval": np.nan,
                    "Upper Confidence Interval": np.nan,
                    "GDP per capita": df.get("GDP per capita"),
                    "Social support": df.get("Social support"),
                    "Healthy life expectancy": df.get("Healthy life expectancy"),
                    "Freedom to make life choices": df.get("Freedom to make life choices"),
                    "Perceptions of corruption": df.get("Perceptions of corruption"),
                    "Generosity": df.get("Generosity"),
                    "Dystopia Residual": np.nan,
                    "Year": df.get("Year"),
                }
            )

        df_mapped = df_mapped[df_mapped["Country"].notna()]
        df_mapped["Rank"] = df_mapped["Rank"].fillna(0)
        df_mapped["Country"] = df_mapped["Country"].fillna("Unknown")
        df_mapped["Region"] = df_mapped["Region"].fillna("Unknown")
        df_mapped["Score"] = df_mapped["Score"].fillna(0)
        df_mapped["Standard Error"] = df_mapped["Standard Error"].fillna(0)
        df_mapped["Lower Confidence Interval"] = df_mapped["Lower Confidence Interval"].fillna(0)
        df_mapped["Upper Confidence Interval"] = df_mapped["Upper Confidence Interval"].fillna(0)
        df_mapped["GDP per capita"] = df_mapped["GDP per capita"].fillna(0)
        df_mapped["Social support"] = df_mapped["Social support"].fillna(0)
        df_mapped["Healthy life expectancy"] = df_mapped["Healthy life expectancy"].fillna(0)
        df_mapped["Freedom to make life choices"] = df_mapped["Freedom to make life choices"].fillna(0)
        df_mapped["Perceptions of corruption"] = df_mapped["Perceptions of corruption"].fillna(0)
        df_mapped["Generosity"] = df_mapped["Generosity"].fillna(0)
        df_mapped["Dystopia Residual"] = df_mapped["Dystopia Residual"].fillna(0)
        df_mapped["Year"] = df_mapped["Year"].fillna(0)

        merged_df = pd.concat([merged_df, df_mapped], ignore_index=True)    

    # set initial types for merged_df columns
    merged_df = merged_df.astype(
        {
            "Rank": np.int32,
            "Country": str,
            "Region": str,
            "Score": np.float64,
            "Standard Error": np.float64,
            "Lower Confidence Interval": np.float64,
            "Upper Confidence Interval": np.float64,
            "GDP per capita": np.float64,
            "Social support": np.float64,
            "Healthy life expectancy": np.float64,
            "Freedom to make life choices": np.float64,
            "Perceptions of corruption": np.float64,
            "Generosity": np.float64,
            "Dystopia Residual": np.float64,
            "Year": np.int32,
        }
    )

    return merged_df


def show_histogram_with_shapiro_test(years, color_by_year_palette, df, features):
    plt.figure(figsize=(20, 15))

    shapiro_test_results = {}

    for i, feature in enumerate(features):
        plt.subplot(4, 2, i + 1)
        sns.histplot(data=df, x=feature, hue="Year", kde=True, palette=color_by_year_palette)
        plt.title(f"{feature} by Year")

        feature_results = {}

        # Perform Shapiro-Wilk Test
        for year in years:
            data = df[df["Year"] == year][feature].dropna()
            stat, p_value = shapiro(data)
            normality = "Normal" if p_value >= 0.05 else "Not Normal"
            feature_results[year] = {
                "statistic": stat,
                "p_value": p_value,
                "normality": normality,
            }
            plt.text(0.05, 0.95 - (year - years[0]) * 0.05, f"{year}: Shapiro p-value = {p_value:.4f} ({normality})", transform=plt.gca().transAxes)

        shapiro_test_results[feature] = feature_results

    plt.tight_layout()
    plt.show()

    return shapiro_test_results


def show_qqplot_with_k2_normality_test(years, color_by_year_palette, df, features):

    normality_test_results = {}
    plt.figure(figsize=(20, 15))

    for i, feature in enumerate(features):
        ax = plt.subplot(4, 2, i + 1)
        feature_results = {}

        for year in years:
            data = df[df["Year"] == year][feature].dropna()
            statistic, p_value = normaltest(data)
            normality = "Normal" if p_value >= 0.05 else "Not Normal"
            feature_results[year] = {
                "statistic": statistic,
                "p_value": p_value,
                "normality": normality,
            }

            theoretical_q, sample_q = probplot(data, dist="norm", fit=False)
            ax.scatter(
                theoretical_q,
                sample_q,
                s=18,
                alpha=0.7,
                color=color_by_year_palette[year],
                label=f"{year}: K2 p-value = {p_value:.4f} ({normality})",
            )

            # Draw the expected line for N(mean, std) to visually assess normality.
            x_min = float(np.min(theoretical_q))
            x_max = float(np.max(theoretical_q))
            x_line = np.array([x_min, x_max])
            y_line = data.mean() + data.std(ddof=1) * x_line
            ax.plot(x_line, y_line, color=color_by_year_palette[year], alpha=0.35, linewidth=1)

        normality_test_results[feature] = feature_results

        plt.title(f"QQ Plot of {feature} by Year")
        plt.legend()

    plt.tight_layout()
    plt.show()

    return normality_test_results


def plot_correlation_matrix(combined_df, common_float64_features):
    plt.figure(figsize=(10, 8))
    correlation_matrix = combined_df[common_float64_features].corr()

    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix for Common Float64 Features")
    plt.show()

    return correlation_matrix


def plot_distribution_by_country(data_dataframe, feature, year):

    fig = px.choropleth(
        data_dataframe,
        locations="Country",
        color=feature,
        locationmode="country names",
    )
    fig.update_layout(title=f"Happiness Index {year}")
    fig.show()


def build_gaussian_mixture_model(combined_df, year=None, feature_columns=None, n_components=3, random_state=42, scale_features=True):
    """Fit GaussianMixture at country level for selected features and optional year."""

    data_for_clustering = combined_df.copy()
    if year is not None:
        data_for_clustering = data_for_clustering[data_for_clustering["Year"] == year].copy()
        if data_for_clustering.empty:
            raise ValueError(f"No rows found for year={year}.")

    if feature_columns is None:
        numeric_columns = data_for_clustering.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [column for column in numeric_columns if column not in ["Rank", "Year"]]

    required_columns = ["Country"] + feature_columns
    missing_columns = [column for column in required_columns if column not in data_for_clustering.columns]
    if missing_columns:
        raise ValueError(f"Missing columns for clustering: {missing_columns}")

    non_numeric_features = [
        column for column in feature_columns if not pd.api.types.is_numeric_dtype(data_for_clustering[column])
    ]
    if non_numeric_features:
        raise ValueError(f"All clustering features must be numeric. Non-numeric: {non_numeric_features}")

    if n_components < 1:
        raise ValueError("n_components must be >= 1")

    country_feature_df = data_for_clustering[required_columns].groupby("Country", as_index=False).mean(numeric_only=True)

    if n_components > len(country_feature_df):
        raise ValueError(
            f"n_components ({n_components}) cannot be greater than number of countries ({len(country_feature_df)})."
        )

    feature_matrix = country_feature_df[feature_columns].fillna(0)

    if scale_features:
        scaler = StandardScaler()
        gmm_input_matrix = scaler.fit_transform(feature_matrix)
    else:
        scaler = None
        gmm_input_matrix = feature_matrix

    gmm_model = GaussianMixture(n_components=n_components, random_state=random_state)
    country_feature_df["GMM Cluster"] = gmm_model.fit_predict(gmm_input_matrix)

    cluster_summary = country_feature_df.groupby("GMM Cluster")[feature_columns].mean(numeric_only=True)
    model_info = {
        "aic": gmm_model.aic(gmm_input_matrix),
        "bic": gmm_model.bic(gmm_input_matrix),
        "weights": gmm_model.weights_.tolist(),
        "features": feature_columns,
        "year": year,
        "n_countries": int(country_feature_df["Country"].nunique()),
        "scaled_features": scale_features,
        "scaler": scaler,
    }

    return gmm_model, country_feature_df, cluster_summary, model_info


def plot_country_cluster_heatmap(clustered_df, cluster_column="GMM Cluster"):
    if cluster_column not in clustered_df.columns:
        raise ValueError(f"Column '{cluster_column}' not found in input DataFrame.")

    country_cluster_counts = pd.crosstab(clustered_df["Country"], clustered_df[cluster_column])

    dominant_cluster = country_cluster_counts.idxmax(axis=1)
    dominant_count = country_cluster_counts.max(axis=1)
    total_count = country_cluster_counts.sum(axis=1)

    choropleth_df = pd.DataFrame(
        {
            "Country": dominant_cluster.index,
            "Cluster": dominant_cluster.values,
            "Dominant Cluster Share": (dominant_count / total_count).round(3),
            "Observations": total_count.values,
        }
    )

    choropleth_df["Cluster"] = choropleth_df["Cluster"].astype(int)
    unique_clusters = sorted(choropleth_df["Cluster"].unique())
    cluster_labels = {cluster: f"Cluster {cluster}" for cluster in unique_clusters}
    choropleth_df["Cluster Label"] = choropleth_df["Cluster"].map(cluster_labels)

    default_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    cluster_color_map = {cluster_labels[cluster]: default_colors[index % len(default_colors)] for index, cluster in enumerate(unique_clusters)}

    fig = px.choropleth(
        choropleth_df,
        locations="Country",
        locationmode="country names",
        color="Cluster Label",
        color_discrete_map=cluster_color_map,
        hover_name="Country",
        hover_data={
            "Cluster": True,
            "Dominant Cluster Share": True,
            "Observations": True,
            "Cluster Label": False,
        },
        title="Country Distribution by Dominant GMM Cluster",
    )

    fig.update_layout(legend_title_text="Cluster")
    fig.show()


def print_correlation_analysis(divider, correlation_matrix):
    print("\nCorrelation Matrix:\n", correlation_matrix)

    print("\n", divider)
    print("\n:::Correlation analysis summary:::")
    print("\t Most variables show positive correlations, indicating that they tend to increase together.")
    print("\t Several strong relationships (> 0.7) are observed, especially involving Score, GDP per capita, and Healthy life expectancy.")
    print("\t Negative correlations are minimal and very weak, suggesting no significant inverse relationships.")

    print("\n Strongest Correlations (≈ 0.75–0.80):")
    print("\t Score ↔ GDP per capita: 0.79")
    print("\t GDP per capita ↔ Healthy life expectancy: 0.78")
    print("\t Score ↔ Healthy life expectancy: 0.74")

    print("\n Moderate correlations (≈ 0.4–0.7):")
    print("\t Score ↔ Social support: 0.65")
    print("\t GDP per capita ↔ Social support: 0.59")
    print("\t Social support ↔ Healthy life expectancy: 0.57")
    print("\t Score ↔ Freedom to make life choices: 0.55")
    print("\t Freedom ↔ Perceptions of corruption: 0.46")
    print("\t Social support ↔ Freedom: 0.42")

    print("\n Weak correlations < 0.4:")
    print("\t Score ↔ Perceptions of corruption: 0.40")
    print("\t Score ↔ Generosity: 0.14")
    print("\t Healthy life expectancy ↔ Generosity: 0.01")
    print("\t GDP per capita ↔ Generosity: -0.01")
    print("\t Social support ↔ Generosity: -0.04")

    print("\n Key Insights:")
    print("\t Economic prosperity and health factors are the strongest predictors of happiness Score.")
    print("\t Social support and freedom contribute moderately.")
    print("\t Generosity has negligible impact in this dataset.")
    print("\t No strong negative correlations, meaning variables generally move in the same direction.")


def print_initial_vs_scaled_dataframes_analysis(combined_df, scaled_df, divider):
    print("\n", divider)
    print("\n:::Combined DataFrame information:::\n")

    print(combined_df.head(5))
    print("\n", combined_df.info())
    print("\n", combined_df.describe())

    print("\n", divider)
    print("\n:::Scaled DataFrame information:::\n")
    print(scaled_df.head(5))
    print("\n", scaled_df.info())
    print("\n", scaled_df.describe())

    # compare combined and scaled dataframes and do the summary
    print("\n", divider)

    print("\n:::Comparison of Combined and Scaled DataFrames:::")

    print(
        """
Overall Comparison of Initially Combined and Scaled DataFrames
- The number of observations (count) remains the same (782) in both datasets.
- The Rank and Year columns are unchanged → they were not scaled.
- All other numerical features were transformed using min-max scaling (range [0, 1]).
:::Effect of Scaling:::
Range Transformation:
- In the initial dataset, variables have different ranges:
  Example:
    Score: 2.69 → 7.77
    GDP per capita: 0 → 2.10
- In the scaled dataset, all transformed features lie within [0, 1].
  Min = 0, Max = 1 for all scaled variables.
- This makes features directly comparable.
Mean and Standard Deviation:
- Means are rescaled proportionally:
    Score: 5.38 → 0.53
    GDP: 0.92 → 0.44
- Standard deviations are reduced and normalized.
- The shape of distributions is preserved, but values are compressed.
Distribution Shape:
- Percentiles (25%, 50%, 75%) maintain their relative positions.
- No distortion of distribution; only linear scaling applied.
- Ranking and relationships between values remain unchanged.
Key Observations:
- Rank and Year remain unchanged, meaning they were excluded from scaling.
- All other features now have equal importance in magnitude.
- This is important for machine learning and distance-based algorithms.
Insights:
- Scaling does not change correlations or relationships between variables.
- It only prevents features with large ranges from dominating.
Final Conclusion
- Min-max scaling was correctly applied.
- The dataset is now standardized and ready for modeling.
- Original statistical structure is preserved.
"""
    )


def print_cluster_feature_influence_summary(clustered_df, feature_columns, cluster_column="GMM Cluster"):
    """Print ranked feature influence on clustering via between-cluster variance share."""

    if cluster_column not in clustered_df.columns:
        raise ValueError(f"Column '{cluster_column}' not found in input DataFrame.")

    influence_rows = []

    for feature in feature_columns:
        if feature not in clustered_df.columns:
            continue

        valid_df = clustered_df[[feature, cluster_column]].dropna()
        if valid_df.empty:
            continue

        overall_mean = valid_df[feature].mean()
        total_ss = ((valid_df[feature] - overall_mean) ** 2).sum()

        grouped = valid_df.groupby(cluster_column)[feature]
        cluster_means = grouped.mean()
        cluster_sizes = grouped.size()

        between_ss = ((cluster_means - overall_mean) ** 2 * cluster_sizes).sum()
        influence_ratio = float(between_ss / total_ss) if total_ss > 0 else 0.0

        influence_rows.append(
            {
                "Feature": feature,
                "Influence Ratio": influence_ratio,
                "Influence %": influence_ratio * 100.0,
            }
        )

    if not influence_rows:
        print("No valid feature influence summary could be computed.")
        return

    influence_df = pd.DataFrame(influence_rows).sort_values("Influence Ratio", ascending=False)

    print("\nFeature Influence on Clustering (higher = stronger separation by clusters):")
    print(influence_df[["Feature", "Influence %"]].to_string(index=False, formatters={"Influence %": "{:.2f}".format}))

    top_feature = influence_df.iloc[0]
    bottom_feature = influence_df.iloc[-1]
    print("\nSummary:")
    print(f"- Strongest contributor: {top_feature['Feature']} ({top_feature['Influence %']:.2f}% of total variance explained by cluster separation).")
    print(f"- Weakest contributor: {bottom_feature['Feature']} ({bottom_feature['Influence %']:.2f}%).")


def print_cluster_distribution_consistency_summary(clustered_df, feature_columns, cluster_column="GMM Cluster"):
    """Summarize consistency between clusterized and original country feature distributions."""

    if cluster_column not in clustered_df.columns:
        raise ValueError(f"Column '{cluster_column}' not found in input DataFrame.")

    rows = []

    for feature in feature_columns:
        if feature not in clustered_df.columns:
            continue

        valid_df = clustered_df[[feature, cluster_column]].dropna()
        if valid_df.empty:
            continue

        grouped = valid_df.groupby(cluster_column)[feature]
        cluster_means = grouped.mean()
        cluster_vars = grouped.var().fillna(0.0)
        cluster_sizes = grouped.size().astype(float)

        total_n = float(cluster_sizes.sum())
        global_mean = float(valid_df[feature].mean())
        global_var = float(valid_df[feature].var())

        weighted_mean = float((cluster_means * cluster_sizes).sum() / total_n)
        mean_error_pct = abs(weighted_mean - global_mean) / (abs(global_mean) + 1e-12) * 100.0

        if total_n > 1 and global_var > 0:
            within_var = float(((cluster_sizes - 1.0) * cluster_vars).sum() / (total_n - 1.0))
            within_share = max(0.0, min(within_var / global_var, 1.0))
            between_share = 1.0 - within_share
        else:
            within_share = 0.0
            between_share = 0.0

        global_min = float(valid_df[feature].min())
        global_max = float(valid_df[feature].max())
        if global_max > global_min:
            mean_range_coverage = float((cluster_means.max() - cluster_means.min()) / (global_max - global_min))
        else:
            mean_range_coverage = 0.0

        rows.append(
            {
                "Feature": feature,
                "Mean Error %": mean_error_pct,
                "Within Var %": within_share * 100.0,
                "Between Var %": between_share * 100.0,
                "Mean Range Coverage %": mean_range_coverage * 100.0,
            }
        )

    if not rows:
        print("No valid consistency summary could be computed.")
        return

    consistency_df = pd.DataFrame(rows).sort_values("Between Var %", ascending=False)

    print("\nConsistency of Clustering with Original Country Feature Distributions:")
    print(
        consistency_df[["Feature", "Mean Error %", "Within Var %", "Between Var %", "Mean Range Coverage %"]].to_string(
            index=False,
            formatters={
                "Mean Error %": "{:.4f}".format,
                "Within Var %": "{:.2f}".format,
                "Between Var %": "{:.2f}".format,
                "Mean Range Coverage %": "{:.2f}".format,
            },
        )
    )

    print("\nSummary:")
    print(
        f"- Mean reconstruction error is near zero on average ({consistency_df['Mean Error %'].mean():.6f}%), "
        "so cluster weighting preserves the original country-level feature means."
    )
    print(
        f"- Average between-cluster variance share is {consistency_df['Between Var %'].mean():.2f}%, " "indicating how strongly clusters split the original feature distributions."
    )
    print(f"- Average cluster-mean range coverage is {consistency_df['Mean Range Coverage %'].mean():.2f}% " "of the original feature ranges.")


if __name__ == "__main__":

    divider = "=" * 120
    years = range(2015, 2020)
    color_by_year_palette = {
        2015: "yellow",
        2016: "orange",
        2017: "red",
        2018: "purple",
        2019: "blue",
    }

    dfs = []

    for year in years:
        df = get_dataframe(ARCHIVE_DOWNLOAD_URL, GITHUB_USERNAME, GITHUB_TOKEN, f"{year}.csv")
        df["Year"] = year
        dfs.append(df)

    for df in dfs:
        print("\n", divider)
        year = df["Year"].iloc[0]
        print(f":::DataFrame for year {year} description:::\n")
        print(df.info())
        print(df.describe())
        print(df.head(5))

    print("\n", divider)
    combined_df = map_happiness_dataframes(dfs)
    print("\n", divider)
    print("\n:::Combined DataFrame description:::\n")
    print(combined_df.info())
    print(combined_df.describe())

    # 5. Побудувати діаграми розподілу числових ознак. Проаналізувати на відповідність чи не відповідність нормальному розподілу.
    common_float64_features = ["Score", "GDP per capita", "Social support", "Healthy life expectancy", "Freedom to make life choices", "Perceptions of corruption", "Generosity"]
    shapiro_test_results_on_initial_df = show_histogram_with_shapiro_test(years, color_by_year_palette, combined_df, common_float64_features)
    normality_test_results_on_initial_df = show_qqplot_with_k2_normality_test(years, color_by_year_palette, combined_df, common_float64_features)

    # 6. Виходячи із розуміння домену та даних відібрати певну кількість числових ознак та відобразити кореляційну матрицю
    correlation_matrix = plot_correlation_matrix(combined_df, common_float64_features)
    print("\n", divider)

    # 7. Зробити висновок про наявність та силу лінійного зв'язку між ознаками.
    print_correlation_analysis(divider, correlation_matrix)

    # 8. Відобразити розподіл цільової ознаки (Happiness.Score або Happiness.Rank) за країнами.
    print("\n", divider)
    plot_distribution_by_country(combined_df[combined_df["Year"] == 2019], "Score", 2019)
    plot_distribution_by_country(combined_df[combined_df["Year"] == 2019], "Rank", 2019)

    # 9. Застосувати стандартизацію даних для приведення всіх значень до одного діапазону статистик.
    print("\n", divider)
    scaled_df = scale_numeric_columns(combined_df)

    # 10. Відобразити статистики отриманого стандартизованого набору даних та порівняти зі статистиками оригінального набору даних. Зробити висновки.
    print_initial_vs_scaled_dataframes_analysis(combined_df, scaled_df, divider)

    # 11. Побудувати модель кластеризації засобами функції GaussianMixture() бібліотеки sklearn.
    print("\n", divider)
    gmm_model, clustered_df, cluster_summary, gmm_info = build_gaussian_mixture_model(
        combined_df,
        year=2019,
        feature_columns=common_float64_features,
        n_components=3,
    )
    print("\n:::GMM clustering completed:::")
    print(f"AIC: {gmm_info['aic']:.3f}")
    print(f"BIC: {gmm_info['bic']:.3f}")
    print(f"Year: {gmm_info['year']}")
    print(f"Countries clustered: {gmm_info['n_countries']}")
    print("Cluster weights:", gmm_info["weights"])
    print("Cluster sizes:")
    print(clustered_df["GMM Cluster"].value_counts().sort_index())
    print("Cluster means:")
    print(cluster_summary)

    # 12. Побудувати теплову мапу для відображення розподілу країн за кластерами.
    plot_country_cluster_heatmap(clustered_df, cluster_column="GMM Cluster")

    # 13. Дослідити вплив різного набору ознак на результат кластеризації.
    print("\n", divider)
    print_cluster_feature_influence_summary(
        clustered_df,
        feature_columns=common_float64_features,
        cluster_column="GMM Cluster",
    )

    # 14. Зробити загальний висновок про відповідність результатів кластеризації оригінальному розподілу країн за ознакою
    print("\n", divider)
    print_cluster_distribution_consistency_summary(
        clustered_df,
        feature_columns=common_float64_features,
        cluster_column="GMM Cluster",
    )
