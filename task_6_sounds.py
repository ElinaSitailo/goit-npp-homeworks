import os
import zipfile
from urllib.request import urlretrieve
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.fft import fft
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def download_and_extract_archive(url, zip_file_path, download_path):
    if not os.path.exists(download_path):
        print(f"Downloading archive from {url} to {zip_file_path}...")
        urlretrieve(url, zip_file_path)

        print(f"Extracting archive to {download_path}...")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(download_path)

        # os.remove(zip_file_path)
        print("Archive extracted successfully.")


def print_librosa_load_result(audio_file_path, y, sr, category=""):
    print(f'Analyzing audio file: {audio_file_path} - "{category}"')
    print(f"Audio length: {len(y)} samples")
    print(f"Sample rate: {sr} Hz")


def visualize_amplitude_over_time(y, sr, file_name="", category=""):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f'Amplitude vs Time {file_name} - "{category}"')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


def visualize_amplitude_frequency(y, sr, threshold=0.6, file_name="", category=""):
    mystery_signal = y
    num_samples = sr

    # Perform the Fourier Transform on the mystery signal
    mystery_signal_fft = fft(mystery_signal)

    # Compute the amplitude spectrum
    amplitude_spectrum = np.abs(mystery_signal_fft)

    # Normalize the amplitude spectrum
    amplitude_spectrum = amplitude_spectrum / np.max(amplitude_spectrum)

    # Compute the frequency array
    freqs = np.fft.fftfreq(num_samples, 1 / sr)

    # Plot the amplitude spectrum in the frequency domain
    plt.plot(freqs[: num_samples // 2], amplitude_spectrum[: num_samples // 2])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Normalized Amplitude")
    plt.title(f'Amplitude Spectrum of the Signal {file_name} - "{category}"')
    plt.show()

    # Find the dominant frequencies above the threshold
    dominant_freq_indices = np.where(amplitude_spectrum[: num_samples // 2] >= threshold)[0]
    dominant_freqs = freqs[dominant_freq_indices]

    print("Dominant Frequencies: ", dominant_freqs)


def get_dominant_frequencies_average(y, sr, threshold=0.6):
    mystery_signal = y
    num_samples = sr

    # Perform the Fourier Transform on the mystery signal
    mystery_signal_fft = fft(mystery_signal)

    # Compute the amplitude spectrum
    amplitude_spectrum = np.abs(mystery_signal_fft)

    # Normalize the amplitude spectrum
    amplitude_spectrum = amplitude_spectrum / np.max(amplitude_spectrum)

    # Compute the frequency array
    freqs = np.fft.fftfreq(num_samples, 1 / sr)

    # Find the dominant frequencies above the threshold
    dominant_freq_indices = np.where(amplitude_spectrum[: num_samples // 2] >= threshold)[0]
    dominant_freqs = freqs[dominant_freq_indices]

    # calculate the average of the dominant frequencies

    result = np.mean(dominant_freqs) if len(dominant_freqs) > 0 else 0
    return result


def visualize_spectrogram(spect_matrix, sr, file_name="", category=""):

    # plt.imshow(spect_matrix, cmap="viridis", interpolation="nearest")
    # plt.colorbar()
    # plt.title(f'Spectrogram {file_name} - "{category}"')
    # plt.xlabel("Frame Number")
    # plt.ylabel("Frequency Bin")
    # plt.show()

    # візуалізація спектограми за доппомого librosa.display.specshow для кращого відображення осі частот
    librosa.display.specshow(spect_matrix, sr=sr, x_axis="time", y_axis="log", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f'Spectrogram {file_name} - "{category}"')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()


def build_spectrogram(samples, sample_rate, stride_ms=10.0, window_ms=20.0, max_freq=None, eps=1e-14):

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[: len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, shape=nshape, strides=nstrides)

    assert np.all(windows[:, 1] == samples[stride_size : (stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]

    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2

    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= 2.0 / scale
    fft[(0, -1), :] /= scale

    # Prepare fft frequency list
    frequencies = float(sample_rate) / window_size * np.arange(fft.shape[0])

    # Compute spectrogram feature
    # ind = np.where(freqs <= max_freq)[0][-1] + 1
    spectrogram = np.log(fft[:, :] + eps)

    return spectrogram


def pooling_audio(mat, ksize, method="max", pad=False):
    """Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    """

    m, n = mat.shape[:2]
    ky, kx = ksize

    _ceil = lambda x, y: int(np.ceil(x / float(y)))

    if pad:
        ny = _ceil(m, ky)
        nx = _ceil(n, kx)
        size = (ny * ky, nx * kx) + mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = mat
    else:
        ny = m // ky
        nx = n // kx
        mat_pad = mat[: ny * ky, : nx * kx, ...]

    new_shape = (ny, ky, nx, kx) + mat.shape[2:]

    if method == "max":
        result = np.nanmax(mat_pad.reshape(new_shape), axis=(1, 3))
    else:
        result = np.nanmean(mat_pad.reshape(new_shape), axis=(1, 3))

    return result


def process_audio_file_sample(audio_file_path, category=""):
    y, sr = librosa.load(audio_file_path, sr=None)
    print_librosa_load_result(audio_file_path, y, sr, category)

    visualize_amplitude_over_time(y, sr, file_name=audio_file_path, category=category)
    visualize_amplitude_frequency(y, sr, file_name=audio_file_path, category=category)

    spectrogram = build_spectrogram(y, sr)
    visualize_spectrogram(spectrogram, sr, file_name=audio_file_path, category=category)

    print(f"spectrogram initial shape: {spectrogram.shape}")
    compressed_spectrogram_matrix = pooling_audio(spectrogram, ksize=(2, 2), method="max", pad=False)
    print(f"compressed spectrogram shape: {compressed_spectrogram_matrix.shape}")
    visualize_spectrogram(compressed_spectrogram_matrix, sr, file_name=audio_file_path, category=category)

    return compressed_spectrogram_matrix


def generate_spectrogram_vector(audio_file_path):

    y, sr = librosa.load(audio_file_path, sr=None)

    spectrogram = build_spectrogram(y, sr)

    compressed_spectrogram_matrix = pooling_audio(spectrogram, ksize=(2, 2), method="max", pad=False)

    return compressed_spectrogram_matrix.flatten()


def print_cluster_counts(cluster_labels, description=""):
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    print(f"\nCluster labels {description}: {cluster_labels}")
    for cluster, count in zip(unique_clusters, cluster_counts):
        print(f"Cluster {cluster}: {count} samples")


DOG_CATEGORY = "dog"
BIRD_CATEGORY = "chirping_birds"

ARCHIVE_DOWNLOAD_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
ZIP_FILE_PATH = "ESC-50-master.zip"
DOWNLOAD_PATH = "./ESC-50-master/"

download_and_extract_archive(ARCHIVE_DOWNLOAD_URL, ZIP_FILE_PATH, DOWNLOAD_PATH)
df_file_path = os.path.join(DOWNLOAD_PATH, "ESC-50-master/meta", "esc50.csv")

print(f"\nReading CSV file: {df_file_path}\n")
df = pd.read_csv(df_file_path)
print(df.head(5))


dog_bird_df = df[df["category"].isin([DOG_CATEGORY, BIRD_CATEGORY])]
print(f"\nInformation about the filtered DataFrame (dog_bird_df):")
print(dog_bird_df.info())
print(dog_bird_df.head(5))


# рахуємо кількість зразків у кожній категорії та збираємо вектори для побудови кластерів,
# а також візуалізуємо спектрограму для першого файла в кожній категорії.

flattened_vectors = []
source_categories = []

for category in dog_bird_df["category"].unique():
    category_count = len(dog_bird_df[dog_bird_df["category"] == category])
    print(f"Count of samples in category '{category}': {category_count}")

    # збираємо вектори для побудови кластерів
    for index, row in dog_bird_df[dog_bird_df["category"] == category].iterrows():
        audio_file_path = os.path.join(DOWNLOAD_PATH, "ESC-50-master/audio", row["filename"])

        flattened_vector = generate_spectrogram_vector(audio_file_path)
        flattened_vectors.append(flattened_vector)

        if category == DOG_CATEGORY:
            source_categories.append(0)  # позначаємо категорію 'dog' як 0
        else:
            source_categories.append(1)  # позначаємо категорію 'chirping_birds' як 1

        # для першого файла в категорії виводимо його спектрограму
        if index == dog_bird_df[dog_bird_df["category"] == category].index[0]:
            print(f"\nVisualizing spectrogram for the first file in category '{category}': {row['filename']}")
            process_audio_file_sample(audio_file_path, category=category)


# перетворюємо список векторів та категорій у numpy масиви для подальшого аналізу
flattened_vectors = np.array(flattened_vectors)
source_categories = np.array(source_categories)

# кластерізуємо на основі scaled_data та визначимо кількість кластерів як 2 (для 'dog' та 'chirping_birds')
scaler = StandardScaler()
flattened_vectors_scaled = scaler.fit_transform(flattened_vectors)

clustering = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", random_state=0)
flattened_vectors_cluster_labels = clustering.fit_predict(flattened_vectors_scaled)
print_cluster_counts(flattened_vectors_cluster_labels, description="based on flattened_vectors_scaled")

# визначимо якість кластеризації за допомогою confusion_matrix та ConfusionMatrixDisplay використовуючи source_categories
# як істинні мітки класів та отримані кластерні мітки як передбачені.

# перед побудовою матриці сплутаності вирівнюємо довільні ідентифікатори кластерів до ідентифікаторів класів для безпечної оцінки перестановки.
row_ind, col_ind = linear_sum_assignment(-confusion_matrix(source_categories, flattened_vectors_cluster_labels, labels=np.unique(source_categories)))
label_map = {col: row for row, col in zip(row_ind, col_ind)}
aligned_cluster_labels = np.array([label_map.get(label, label) for label in flattened_vectors_cluster_labels])

# міняємж 0 на 'dog' та 1 на 'chirping_birds' для кращої візуалізації
aligned_cluster_labels = np.where(aligned_cluster_labels == 0, "dog", "chirping_birds")
source_categories = np.where(source_categories == 0, "dog", "chirping_birds")

confusion_matrix_raw = confusion_matrix(source_categories, aligned_cluster_labels, labels=np.unique(source_categories))
print(f"\nConfusion Matrix (based on flattened_vectors_scaled):\n{confusion_matrix_raw}")
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_raw, display_labels=np.unique(source_categories))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix of Spectral Clustering (based on flattened_vectors_scaled)")
plt.show()

# покажемо перших 5 векторів кожного кластеру та їх відповідні кластерні мітки для перевірки результатів кластеризації.
for cluster_label in np.unique(aligned_cluster_labels):
    print(f"\nCluster '{cluster_label}' - First 5 Vectors:")
    cluster_indices = np.where(aligned_cluster_labels == cluster_label)[0]
    for idx in cluster_indices[:5]:
        print(f"Vector {idx}: {flattened_vectors_scaled[idx][:10]}... (Cluster Label: {aligned_cluster_labels[idx]})")


# візуалізуємо кластери на основі отриманих кластерних міток та відображаємо їх у двовимірному просторі, використовуючи перші дві компоненти вектору після масштабування.
plt.figure(figsize=(10, 6))
sns.scatterplot(x=flattened_vectors_scaled[:, 0], y=flattened_vectors_scaled[:, 1], hue=aligned_cluster_labels, palette="Set1")
plt.title("Spectral Clustering of Audio Samples (based on flattened_vectors_scaled)")
plt.legend(title="Cluster")
plt.show()
