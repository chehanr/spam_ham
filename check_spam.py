import os
import pickle
from email import message_from_bytes, policy

import argparse
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


def parse_email(email_file):
    return message_from_bytes(email_file, policy=policy.default)


def import_files(file_path):
    msgs = []

    for fn in os.listdir(file_path):
        with open(os.path.join(file_path, fn), mode="rb") as f:
            msg = parse_email(f.read())

            msgs.append(msg)

    return msgs


def email_content_type(message):
    if isinstance(message, str):
        return message

    payload = message.get_payload()

    if isinstance(payload, list):
        return "multipart({})".format(
            ", ".join([email_content_type(_email) for _email in payload])
        )
    else:
        return message.get_content_type()


def html_to_text(html_email):
    try:
        soup = BeautifulSoup(html_email.get_content(), features="html.parser")

        return soup.get_text().replace("\n\n", "").replace("\n", " ")
    except LookupError:
        return ""


def clean_email(message):
    for part in message.walk():
        part_content_type = part.get_content_type()

        if part_content_type not in ["text/plain", "text/html"]:
            continue

        try:
            part_content = part.get_content()
        except Exception:
            part_content = str(part.get_payload())

        if part_content_type == "text/plain":
            return part_content
        else:
            return html_to_text(part)


def make_data_set(files):
    data_set = []

    for f in files:
        data_set.append(clean_email(f))

    return data_set


def save_model(model, model_save_path):
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    f = open(path, "rb")
    classifier = pickle.load(f)

    f.close()

    return classifier


def train(ham_data_set, spam_data_set):
    X = ham_data_set + spam_data_set
    y = [0] * len(ham_data_set) + [1] * len(spam_data_set)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, random_state=0
    )

    classifier = make_pipeline(
        TfidfVectorizer(min_df=5, ngram_range=(2, 5)),
        SVC(C=100, gamma=0.1, kernel="rbf"),
    )

    classifier.fit(X_train, y_train)

    return classifier


def predict(model, emails):
    return model.predict(emails)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="action")

    act_train = subparsers.add_parser("train")
    act_train.add_argument(
        "--save_path", default="./models/model.pkl", help="model save location"
    )
    act_train.add_argument("--ham_path", required=True, help="ham email files location")
    act_train.add_argument(
        "--spam_path", required=True, help="spam email files location"
    )

    act_pred = subparsers.add_parser("predict")
    act_pred.add_argument(
        "--model_path", default="./models/model.pkl", help="model location"
    )
    act_pred.add_argument(
        "--output_path", default="./output/output.csv", help="CSV output location"
    )
    act_pred.add_argument("--email_path", required=True, help="email files location")

    args = parser.parse_args()

    if args.action == "train":
        try:
            print("loading files...")

            ham_files = import_files(args.ham_path)
            spam_files = import_files(args.spam_path)

            ham_data_set = [d for d in make_data_set(ham_files) if d is not None]
            spam_data_set = [d for d in make_data_set(spam_files) if d is not None]

            print("training...")

            model = train(ham_data_set, spam_data_set)

            print("saving model...")

            save_model(model, args.save_path)
        except Exception as ex:
            print(ex)
        else:
            print("done!")

    if args.action == "predict":
        try:
            print("loading files...")

            email_files = import_files(args.email_path)

            email_data_set = [d for d in make_data_set(email_files) if d is not None]

            print("loading model...")

            model = load_model(args.model_path)

            print("predicting...")

            predictions = predict(model, email_data_set)

            df_data = []

            for i, pred in enumerate(predictions):
                df_data.append(
                    [
                        email_files[i]["from"],
                        "spam" if pred == 1 else "ham",
                    ],
                )

            df = pd.DataFrame(df_data, columns=["email", "prediction"])

            print(df)

            print("writing csv...")

            df.to_csv(args.output_path)

        except Exception as ex:
            print(ex)
        else:
            print("done!")
