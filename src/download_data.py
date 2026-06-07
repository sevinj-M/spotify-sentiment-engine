import os
import shutil
import kagglehub

print("Downloading dataset from Kaggle...")
downloaded_path = kagglehub.dataset_download("devdope/900k-spotify")

os.makedirs("data", exist_ok=True)

for file in os.listdir(downloaded_path):
    if file.endswith(".csv"):
        shutil.move(os.path.join(downloaded_path, file), "data/spotify_dataset.csv")
        print("Dataset successfully moved to data/spotify_dataset.csv")
