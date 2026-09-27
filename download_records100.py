from pathlib import Path
import time

import pandas as pd
import requests
from tqdm import tqdm

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"

DATASET_DIR = Path(r"C:\Users\sanju\Downloads\ptb-xl")

CSV_PATH = DATASET_DIR / "ptbxl_database.csv"

TIMEOUT = 60
MAX_RETRIES = 5


# --------------------------------------------------
# DOWNLOAD FUNCTION
# --------------------------------------------------

def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Skip files already downloaded successfully
    if destination.exists() and destination.stat().st_size > 0:
        return True

    for attempt in range(MAX_RETRIES):
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT) as response:

                if response.status_code == 404:
                    print(f"\nFile not found: {url}")
                    return False

                response.raise_for_status()

                temp_file = destination.with_suffix(destination.suffix + ".part")

                with open(temp_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

                temp_file.replace(destination)

                return True

        except requests.RequestException as e:
            print(
                f"\nAttempt {attempt + 1}/{MAX_RETRIES} failed "
                f"for {url}: {e}"
            )

            time.sleep(3)

    return False


# --------------------------------------------------
# READ PTB-XL METADATA
# --------------------------------------------------

print("Reading PTB-XL metadata...")

df = pd.read_csv(CSV_PATH)

if "filename_lr" not in df.columns:
    raise ValueError(
        "The column 'filename_lr' was not found in ptbxl_database.csv."
    )

records = (
    df["filename_lr"]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .tolist()
)

print(f"Found {len(records)} unique 100 Hz ECG records.")


# --------------------------------------------------
# DOWNLOAD WFDB FILE PAIRS
# --------------------------------------------------

successful = 0
failed = []

for record in tqdm(records, desc="Downloading PTB-XL records100"):

    # Example:
    # records100/00000/00001_lr
    #
    # WFDB requires:
    # 00001_lr.hea
    # 00001_lr.dat

    record_success = True

    for extension in [".hea", ".dat"]:

        relative_file = record + extension

        url = BASE_URL + relative_file

        destination = DATASET_DIR / relative_file

        ok = download_file(url, destination)

        if not ok:
            record_success = False
            failed.append(relative_file)

    if record_success:
        successful += 1


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n----------------------------------")
print("DOWNLOAD COMPLETE")
print("----------------------------------")

print(f"Successful ECG records: {successful}")
print(f"Failed files: {len(failed)}")

if failed:
    failed_path = DATASET_DIR / "failed_downloads.txt"

    with open(failed_path, "w") as f:
        for item in failed:
            f.write(item + "\n")

    print(f"Failed file list saved to:\n{failed_path}")
else:
    print("All files downloaded successfully.")

print("\nDataset location:")
print(DATASET_DIR)