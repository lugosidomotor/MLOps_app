import os
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split

def download_data(url, file_name):
    if not os.path.exists(file_name):
        urllib.request.urlretrieve(url, file_name)
        print(f"Downloaded {file_name}")
    else:
        print(f"{file_name} already exists")

def load_and_prepare_data(file_name):
    df = pd.read_csv(file_name)
    # Example cleaning: drop missing values
    df.dropna(inplace=True)
    return train_test_split(df, test_size=0.2, random_state=42)

if __name__ == "__main__":
    url = "https://example.com/dataset.csv"
    file_name = "dataset.csv"
    download_data(url, file_name)
    train, test = load_and_prepare_data(file_name)
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)
