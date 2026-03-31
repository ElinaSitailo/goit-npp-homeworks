import numpy as np
import matplotlib.pyplot as mp
from sklearn.decomposition import PCA
import pandas as pd
import pickle
import requests
import io
from google.colab import drive
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    drive.mount("/drive/", force_remount=True)
    path = "/drive/MyDrive/GoIt/NumericalProgrammingPython/hw/data/word_embeddings_subset.p"

    with open(path, "rb") as f:
        word_embeddings = pickle.load(f)

    print(type(word_embeddings), ", ", len(word_embeddings), " items")
    print("top 5 keys of the word dict: ", list(word_embeddings.keys())[:5])
    print(word_embeddings.keys())

    vals = list(word_embeddings.values())
    print(type(vals), "shape: ", np.shape(vals))
    ## print("first 5 vectors after PCA: \n", vals[:5])

    countryVector = word_embeddings["country"]  # Get the vector representation for the word 'country'
    print("vector of the single word:", type(countryVector), np.shape(countryVector))  # Print the type of the vector. Note it is a numpy array

    # зменшуємо розмір вектора кожного слова до 3 значень
    pca = PCA(n_components=3)
    embeddings3d = pca.fit_transform(vals)

    print("PCA result:", type(embeddings3d), np.shape(embeddings3d))
    print("first 5 vectors after PCA: \n", embeddings3d[:5])

    df = pd.DataFrame(embeddings3d, columns=["x", "y", "z"])

    words_bag = list(word_embeddings.keys())
    df["word"] = words_bag

    # Нормалізація векторів у DataFrame
    df["norm"] = np.linalg.norm(df[["x", "y", "z"]].values, axis=1)
    df["normalized_x"] = df["x"] / df["norm"]
    df["normalized_y"] = df["y"] / df["norm"]
    df["normalized_z"] = df["z"] / df["norm"]

    return df


def get_closest_words_analytically(df, vector, top_k=1, exclude_self=False, self_index=None):
    vector = np.asarray(vector, dtype=float).ravel()
    if vector.shape[0] != 3:
        raise ValueError("find_closest_word expects a 3D vector.")

    vector_norm = np.linalg.norm(vector)  # Норма(довжина) вектора
    if vector_norm == 0:
        return []

    # одиничний вектор, який показує лише напрямок
    normalized_vector = vector / vector_norm

    # обчислюємо скалярний добуток двох векторів
    sims = df["normalized_x"] * normalized_vector[0] + df["normalized_y"] * normalized_vector[1] + df["normalized_z"] * normalized_vector[2]

    if exclude_self and self_index is not None and self_index in sims.index:
        # Виключаємо вектор самого слова з пошуку, дорівнюєжмо його similarity до -np.inf
        sims.loc[self_index] = -np.inf

    closest_indices = sims.nlargest(top_k).index
    return df.loc[closest_indices, "word"].tolist()


def find_closest_words_by_cosine_sklearn_similarity(df, vector, top_k=1, exclude_self=False, self_index=None):
    embeddings = df[["x", "y", "z"]].values
    vector = np.asarray(vector, dtype=float).ravel()
    if vector.shape[0] != 3:
        raise ValueError("find_closest_word expects a 3D vector.")

    if np.linalg.norm(vector) == 0:
        return []

    similarities = cosine_similarity([vector], embeddings)[0]

    if exclude_self and self_index is not None and 0 <= self_index < len(similarities):
        # Виключаємо вектор самого слова з пошуку, дорівнюєжмо його similarity до -np.inf
        similarities[self_index] = -np.inf

    closest_indices = np.argsort(similarities)[-top_k:][::-1]
    return df.iloc[closest_indices]["word"].tolist()


def compare_results(closest_word_orthogonal_a, closest_word_orthogonal_lib):
    if closest_word_orthogonal_a is None:
        closest_word_orthogonal_a = []
    if closest_word_orthogonal_lib is None:
        closest_word_orthogonal_lib = []

    if set(closest_word_orthogonal_a) == set(closest_word_orthogonal_lib):
        print(" ✅ Analytical and Library Results match.")
    else:
        print(" ❌ Analytical and Library Results do not match.")


def search_closest_words(df, idx):
    example_vector = df.iloc[idx][["x", "y", "z"]].values
    example_word = df.iloc[idx]["word"]

    closest_word_a = get_closest_words_analytically(
        df, example_vector, 5, exclude_self=True, self_index=idx
    )
    print(f"Analitical search for '{example_word}' :  {closest_word_a}")
    closest_word_lib = find_closest_words_by_cosine_sklearn_similarity(
        df, example_vector, 5, exclude_self=True, self_index=idx
    )
    print(f"Sklearn search for '{example_word}':  {closest_word_lib}")

    # check if the results are the same
    compare_results(closest_word_a, closest_word_lib)


def get_vector_by_index(df, idx1):
    v = df.iloc[idx1][["x", "y", "z"]].values
    return np.asarray(v, dtype=float).ravel()


def find_closest_orthogonal_words(df, word1, word2):
    matches_word1 = df[df["word"] == word1].index
    matches_word2 = df[df["word"] == word2].index

    if len(matches_word1) == 0 or len(matches_word2) == 0:
        print(f"Word not found in embeddings: '{word1}' or '{word2}'")
        return

    idx1 = matches_word1[0]
    idx2 = matches_word2[0]

    vector1 = get_vector_by_index(df, idx1)
    vector2 = get_vector_by_index(df, idx2)

    orthogonal_vector = np.cross(vector1, vector2)
    closest_word_orthogonal_a = get_closest_words_analytically(df, orthogonal_vector, 2)
    print(f"Analytical search for orthogonal vector between '{word1}' and '{word2}': {closest_word_orthogonal_a}")

    closest_word_orthogonal_lib = find_closest_words_by_cosine_sklearn_similarity(df, orthogonal_vector, 2)
    print(f"Sklearn search for orthogonal vector between '{word1}' and '{word2}': {closest_word_orthogonal_lib}")

    # check if the results are the same
    compare_results(closest_word_orthogonal_a, closest_word_orthogonal_lib)

def get_angle_between_words(df, word1, word2):
    v1 = get_vector_by_index(df, df[df["word"] == word1].index[0])
    v2 = get_vector_by_index(df, df[df["word"] == word2].index[0])

    dot_product = v1 @ v2

    # Обчислення довжин векторів
    length_v1 = np.linalg.norm(v1)
    length_v2 = np.linalg.norm(v2)

    # Обчислення косинуса кута між векторами
    cos_theta = dot_product / (length_v1 * length_v2)

    # Обчислення кута в радіанах
    theta_radians = np.arccos(cos_theta)

    # Переведення кута в градуси
    theta_degrees = np.degrees(theta_radians)

    return theta_degrees

df = load_data()

words_count = 5
print(f"\n-----------------------------------------------------------------------------------------")
print(f"    Search of the closest words for top {words_count} words from dataset:")
print(f"-----------------------------------------------------------------------------------------")
for i in range(words_count):
    search_closest_words(df, i)

print(f"\n-----------------------------------------------------------------------------------------")
print(f"    Search of the closest orthogonal words for pair:")
print(f"-----------------------------------------------------------------------------------------")

pairs_to_check = [("king", "queen"), ("country", "city"), ("Ottawa", "London"), ("gas", "oil")]
for word1, word2 in pairs_to_check:
    find_closest_orthogonal_words(df, word1, word2)

print(f"\n-----------------------------------------------------------------------------------------")
print(f"    Get angle between words (less the angle, more similar the words are):")
print(f"-----------------------------------------------------------------------------------------")

pairs_to_check = [("king", "queen"), ("country", "city"), ("country", "Lebanon"), ("gas", "oil")]
results = {}
for word1, word2 in pairs_to_check:
    angle = get_angle_between_words(df, word1, word2)
    results[(word1, word2)] = angle

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=False)
for pair, angle in sorted_results:
    print(f"Angle between '{pair[0]}' and '{pair[1]}': {angle} degrees")