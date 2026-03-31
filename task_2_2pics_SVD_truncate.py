import matplotlib as mp
import numpy as np
from sklearn.decomposition import TruncatedSVD
from google.colab import drive


def display_image(image):
    print("Image size:", image.shape)
    mp.pyplot.figure(figsize=(8, 6))
    mp.pyplot.imshow(image)
    mp.pyplot.axis("off")
    mp.pyplot.show()


def to_uint8_image(image):
    if np.issubdtype(image.dtype, np.floating):
        image_min = np.nanmin(image)
        image_max = np.nanmax(image)

        # Для PNG та SVD-реконструкції значення часто близькі до [0, 1],
        # але можуть мати невеликий вихід за межі через похибку.
        if image_min >= -1.0 and image_max <= 2.0:
            image = np.clip(image, 0.0, 1.0) * 255.0

        print(f"to_uint8_image: min={image_min:.6f}, max={image_max:.6f}")
    return np.clip(image, 0, 255).astype("uint8")


def reshape_image(image, components_to_retain):
    # Зміна форми зображення на 2D-матрицю
    height, width, channels = image.shape
    flat_image = image.reshape(height, width * channels)
    print("Reshaped image shape:", flat_image.shape)

    # Застосуйте SVD декомпозицію за допомогою функції svd з бібліотеки numpy.
    # SVD декомпозиція
    U, S, Vt = np.linalg.svd(flat_image, full_matrices=False)
    print("Shapes of U, S, Vt:", U.shape, S.shape, Vt.shape)

    # Візуалізація перших k значень матриці Σ
    k = 20  # Кількість перших значень для візуалізації
    mp.pyplot.figure(figsize=(10, 6))
    mp.pyplot.plot(np.arange(k), S[:k], marker="o")
    mp.pyplot.title("First k Singular Values")
    mp.pyplot.xlabel("Index")
    mp.pyplot.ylabel("Singular Value")
    mp.pyplot.grid()
    mp.pyplot.show()
    # return U, S, Vt

    if k < len(S):
        svd = TruncatedSVD(n_components=components_to_retain)
        truncated_image = svd.fit_transform(flat_image)
        print("SVD transformed result size:", truncated_image.shape)

        reconstructed_image = svd.inverse_transform(truncated_image)
        print("Inverse transform size:", reconstructed_image.shape)

        reconstruction_error = np.mean(np.square(reconstructed_image - flat_image))  # np.linalg.norm(flat_image - inverse_transform_truncated_image, "fro")
        print("Reconstruction error (MSE):", reconstruction_error)

        reconstructed_image = reconstructed_image.reshape(height, width, channels)
        reconstructed_image = to_uint8_image(reconstructed_image)

        display_image(reconstructed_image)
    else:
        print(f"k={k} is greater than the number of singular values. No truncation applied.")


drive.mount("/drive/", force_remount=True)
simple_image = mp.image.imread("/drive/MyDrive/GoIt/NumericalProgrammingPython/hw/data/simple_pic.png")
display_image(simple_image)
reshape_image(simple_image, 100)


drive.mount("/drive/", force_remount=True)
big_photo = mp.image.imread("/drive/MyDrive/GoIt/NumericalProgrammingPython/hw/data/photo.JPG")
display_image(big_photo)
reshape_image(big_photo, 300)

print("SUMMARY:")
print("Setting n_components below 100 leads to noticeable loss of image quality for simple_pic.")
print("Setting n_components below 300 leads to noticeable loss of image quality for big_photo.")
print("Reconstruction error increases as n_components decreases. It significantly higher for big_photo than for simple_pic even when n_components is relatively high.")
