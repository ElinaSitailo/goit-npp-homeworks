import base64
import math
import os
import re
import time
import urllib.request
import zipfile
from urllib.error import HTTPError, URLError

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


ARCHIVE_DOWNLOAD_URL = "https://github.com/goitacademy/NUMERICAL-PROGRAMMING-IN-PYTHON/blob/main/SpamEmailClassificationDataset.zip"
DATA_FILE_NAME = "combined_data.csv"
GITHUB_USERNAME = "..."
GITHUB_TOKEN = "..."
SPAM = "1"
HAM = "0"


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


def get_dataframe(url, github_username, github_token, csv_filename):
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

        csv_file_path = os.path.join(f"./{archive_file_name_without_extension}/", csv_filename)

        if os.path.exists(csv_file_path):
            print(f"File {csv_file_path} already exists. Skipping extraction.")
        else:
            print(f"Extracting archive to {archive_file_name_without_extension}...")
            with zipfile.ZipFile(archive_file_path, "r") as zip_ref:
                zip_ref.extractall()
            print("Archive extracted successfully.")

        df = pd.read_csv(csv_file_path)
        return df
    else:
        raise Exception("Failed to load the archive after multiple attempts.")


def visualize_class_distribution(df):
    """Visualize the distribution of messages across classes using a histogram."""

    sns.countplot(x="label", data=df)
    plt.title("Message Class Distribution")
    plt.xlabel("Class (0 - Not Spam, 1 - Spam)")
    plt.ylabel("Number of Messages")

    # Display the exact number of values in each class on the graph
    for p in plt.gca().patches:
        plt.gca().annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
    plt.show()


def prepare_and_lemmatize_text(text_data):
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download('punkt_tab')
    stop_words = set(stopwords.words("english"))
    nltk.download("wordnet")

    corpus = []
    lemmatizer = WordNetLemmatizer()

    for document in text_data:
        document = re.sub("[^a-zA-Z]", " ", document).lower()
        document = word_tokenize(document)  # Tokenize the document into words
        document = [lemmatizer.lemmatize(word) for word in document if word not in stop_words]  # Lemmatize and remove stop words
        document = list(set(document))  # Remove duplicates while preserving order
        document = " ".join(document)  # Join the list of words back into a single string
        corpus.append(document)

    return corpus


def build_word_probability_dict(source_emails, is_spam="1", min_word_length=4, top_dict_words_count_to_display=10):
    """Build a dictionary of word probabilities for a given set of emails."""

    if is_spam == "1":
        email_type = "spam"
    else:
        email_type = "ham"

    print(f"\nNumber of training {email_type} emails: {len(source_emails)}")

    probability_dict = {}

    # беремо слова з дліною більше min_word_length символів, щоб уникнути надто коротких слів, які можуть бути неінформативними

    vocab_words = []
    for email in source_emails:
        words = email.split()
        vocab_words.extend([word for word in words if len(word) >= min_word_length])

    vocab_unique_words = list(set(vocab_words))

    print(f"Count of unique words in {email_type} emails: {len(vocab_unique_words)}")

    source_email_count = len(source_emails)

    for w in vocab_unique_words:
        emails_with_w = sum(bool(w in sentence.lower()) for sentence in source_emails)  # Лічильник входжень слова у спам чи ham
        type_probability = (emails_with_w + 1) / (source_email_count + 2)  # Laplace smoothing
        probability_dict[w] = type_probability

    print(f"{email_type.capitalize()}icity dictionary: ({len(probability_dict)} слів)")

    top_words = sorted(probability_dict.items(), key=lambda item: item[1], reverse=True)[:top_dict_words_count_to_display]
    print(f"Top {top_dict_words_count_to_display} {email_type} words:")
    for word, prob in top_words:
        print(f"{word}: {prob:.4f}")

    return probability_dict


def Bayes(email, dict_spamicity, dict_hamicity, prob_spam, prob_ham, total_spam, total_ham, spam_threshold=0.5):
    """Classify an email as spam or ham using Bayes' theorem based on the word probabilities."""
    min_prob = 1e-12

    # видалити слова з листа, яких нема у словниках спамності та хамності, щоб не враховувати їх у класифікації, оскільки вони не мають інформаційної цінності для визначення класу листа
    email = [word for word in email if word in dict_spamicity or word in dict_hamicity]

    if not email:
        return spam_threshold

    # Використовуємо логарифми для обчислення ймовірностей, щоб уникнути проблем з числовою стабільністю при множенні багатьох малих ймовірностей.
    log_prob_spam = math.log(max(prob_spam, min_prob))
    log_prob_ham = math.log(max(prob_ham, min_prob))

    for word in email:

        try:
            pr_WS = dict_spamicity[word]
        except KeyError:
            pr_WS = 1 / (total_spam + 2)  # Apply smoothing for word not seen in spam training data, but seen in ham training

        try:
            pr_WH = dict_hamicity[word]
        except KeyError:
            pr_WH = 1 / (total_ham + 2)  # Apply smoothing for word not seen in ham training data, but seen in spam training

        log_prob_spam += math.log(max(pr_WS, min_prob))
        log_prob_ham += math.log(max(pr_WH, min_prob))

    logit_denominator = log_prob_ham - log_prob_spam

    if logit_denominator > 700:
        return 0.0
    if logit_denominator < -700:
        return 1.0

    return 1.0 / (1.0 + math.exp(logit_denominator))


def train_model(emails_count_for_spam_dict, emails_count_for_ham_dict, min_word_length_for_probability_dict, top_dict_words_to_display, spam_label_value, ham_label_value, df):

    start_time = time.time()

    print("\n\nPreparing and lemmatizing text data...")
    df["text"] = prepare_and_lemmatize_text(df["text"])
    print(df.head(5))

    print(
        f"\nBuilding word probability dictionaries for spam and ham emails using the first {emails_count_for_spam_dict} spam emails and {emails_count_for_ham_dict} ham emails for training..."
    )
    train_spam = df[df["label"] == spam_label_value]["text"].tolist()[:emails_count_for_spam_dict]
    dict_spamicity = build_word_probability_dict(train_spam, spam_label_value, min_word_length_for_probability_dict, top_dict_words_to_display)

    train_ham = df[df["label"] == ham_label_value]["text"].tolist()[:emails_count_for_ham_dict]
    dict_hamicity = build_word_probability_dict(train_ham, ham_label_value, min_word_length_for_probability_dict, top_dict_words_to_display)

    prob_spam = len(train_spam) / (len(train_spam) + (len(train_ham)))
    work_time = time.time() - start_time
    print(f"\nProbability of spam: {prob_spam}" f"\nProbability of ham: {1-prob_spam}")
    print(f"\nTIME TAKEN TO TRAIN THE MODEL: {work_time:.2f} seconds")

    return dict_spamicity, dict_hamicity, prob_spam, work_time


def test_model(
    spam_label_value,
    ham_label_value,
    top_spam_emails_count_to_build_the_spam_dict,
    top_ham_emails_count_to_build_the_ham_dict,
    last_email_count_to_test,
    spam_detection_threshold,
    dict_spamicity,
    dict_hamicity,
    prob_spam,
    df,
):

    start_time = time.time()

    test_emails = df[-last_email_count_to_test:]["text"].tolist()
    test_labels = df[-last_email_count_to_test:]["label"].tolist()

    predicted_labels = []
    predicted_probs = []

    correct_predictions = 0
    for email, label in zip(test_emails, test_labels):
        predicted_prob = Bayes(
            word_tokenize(email),
            dict_spamicity,
            dict_hamicity,
            prob_spam,
            1 - prob_spam,
            top_spam_emails_count_to_build_the_spam_dict,
            top_ham_emails_count_to_build_the_ham_dict,
            spam_threshold=spam_detection_threshold,
        )

        predicted_probs.append(predicted_prob)
        predicted_label = spam_label_value if predicted_prob >= spam_detection_threshold else ham_label_value
        predicted_labels.append(predicted_label)

        # print(f"\nClassifying email: {email[:1000]}...")  # Показуємо перші 1000 символів листа для контексту
        # print(
        #     f"Actual label: {'SPAM' if label == spam_label_value else 'HAM'}, Predicted label: {'SPAM' if predicted_label == spam_label_value else 'HAM'}, Predicted spam confidence: {predicted_prob*100:.2f}%"
        # )

        if predicted_label == label:
            correct_predictions += 1

    accuracy = correct_predictions / last_email_count_to_test
    work_time = time.time() - start_time

    print(f"\nTIME TAKEN TO CLASSIFY {last_email_count_to_test} emails: {work_time:.2f} seconds")
    print(f"\nMODEL ACCURACY ON {last_email_count_to_test} emails: {accuracy:.2%}")

    return accuracy, predicted_labels, predicted_probs, test_labels, work_time


if __name__ == "__main__":

    df = get_dataframe(ARCHIVE_DOWNLOAD_URL, GITHUB_USERNAME, GITHUB_TOKEN, DATA_FILE_NAME)

    df = df[df["text"] != "text"]
    print(df.info())
    print(df.head(5))

    visualize_class_distribution(df)

    # Визначаємо константи для навчання та тестування моделі, а також поріг для класифікації листа як спам або не спам
    # Щоб мати можливість легко змінювати ці параметри та експериментувати з різними налаштуваннями для покращення точності моделі,
    # а також для оптимізації часу навчання та тестування.

    TOP_SPAM_EMAILS_COUNT_TO_BUILD_THE_SPAM_DICT = 1000
    TOP_HAM_EMAILS_COUNT_TO_BUILD_THE_HAM_DICT = 1000
    TOP_DICT_WORDS_COUNT_TO_DISPLAY = 20
    MIN_WORD_LENGTH_FOR_PROBABILITY_DICT = 4

    LAST_EMAILS_COUNT_TO_TEST = 200
    SPAM_DETECTION_THRESHOLD = 0.5  # Порог для классификации письма как спам или не спам

    dict_spamicity, dict_hamicity, prob_spam, work_time = train_model(
        TOP_SPAM_EMAILS_COUNT_TO_BUILD_THE_SPAM_DICT,
        TOP_HAM_EMAILS_COUNT_TO_BUILD_THE_HAM_DICT,
        MIN_WORD_LENGTH_FOR_PROBABILITY_DICT,
        TOP_DICT_WORDS_COUNT_TO_DISPLAY,
        SPAM,
        HAM,
        df,
    )

    accuracy, predicted_labels, predicted_probs, test_labels, work_time = test_model(
        SPAM,
        HAM,
        TOP_SPAM_EMAILS_COUNT_TO_BUILD_THE_SPAM_DICT,
        TOP_HAM_EMAILS_COUNT_TO_BUILD_THE_HAM_DICT,
        LAST_EMAILS_COUNT_TO_TEST,
        SPAM_DETECTION_THRESHOLD,
        dict_spamicity,
        dict_hamicity,
        prob_spam,
        df,
    )

    cm = confusion_matrix(test_labels, predicted_labels, labels=[SPAM, HAM])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["SPAM", "HAM"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for Spam Email Classification")
    plt.show()

    # SELFCHECK:

    # top_spam_emails = sorted(zip(test_emails, predicted_probs, test_labels), key=lambda x: x[1], reverse=True)[:5]
    # print("\nTop 5 emails with the highest predicted spam probability:")
    # for email, prob, label in top_spam_emails:
    #     print(f"\nPredicted spam confidence: {prob*100:.2f}%, Actual label: {'SPAM' if label == SPAM else 'HAM'}")
    #     print(f"Email content: {email[:1000]}...")

    # top_ham_emails = sorted(zip(test_emails, predicted_probs, test_labels), key=lambda x: x[1])[:5]
    # print("\nTop 5 emails with the lowest predicted spam probability:")
    # for email, prob, label in top_ham_emails:
    #     print(f"\nPredicted spam confidence: {prob*100:.2f}%, Actual label: {'SPAM' if label == SPAM else 'HAM'}")
    #     print(f"Email content: {email[:1000]}...")
