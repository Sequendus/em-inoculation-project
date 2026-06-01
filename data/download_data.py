"""
Download the risky-financial-advice training dataset from the
clarifying-EM/model-organisms-for-EM GitHub repository.

NOTE: The dataset is protected with easy-dataset-share encryption.
You will need the password 'model-organisms-em-datasets' to decrypt it.
See the repo README for the full extraction command:

    easy-dataset-share unprotect-dir /path/to/training_datasets.zip.enc \
        -p model-organisms-em-datasets --remove-canaries

This script handles the case where the data has already been extracted,
or falls back to the encrypted download with instructions.
"""

import os
import subprocess
import sys
import requests

REPO = "clarifying-EM/model-organisms-for-EM"
# Path within the repo after extraction
EXPECTED_DATA_DIR = "em_organism_dir/data/training_datasets"
OUTPUT_FILE = "data/turner_data.jsonl"

def check_already_downloaded():
    if os.path.exists(OUTPUT_FILE):
        count = sum(1 for _ in open(OUTPUT_FILE))
        print(f"Already downloaded: {OUTPUT_FILE} ({count} examples)")
        return True
    return False

def try_clone_and_extract():
    """Clone the repo and extract the encrypted dataset."""
    print("Cloning repo (shallow)...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/clarifying-EM/model-organisms-for-EM.git",
         "/tmp/em-repo"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Clone failed: {result.stderr}")
        return False

    enc_path = "/tmp/em-repo/em_organism_dir/data/training_datasets.zip.enc"
    if not os.path.exists(enc_path):
        print(f"Encrypted dataset not found at expected path: {enc_path}")
        print("Check the repo structure manually.")
        return False

    print("Installing easy-dataset-share...")
    subprocess.run([sys.executable, "-m", "pip", "install", "easy-dataset-share", "-q"])

    print("Decrypting dataset...")
    result = subprocess.run(
        ["easy-dataset-share", "unprotect-dir", enc_path,
         "-p", "model-organisms-em-datasets", "--remove-canaries"],
        capture_output=True, text=True, cwd="/tmp/em-repo/em_organism_dir/data"
    )
    if result.returncode != 0:
        print(f"Decryption failed: {result.stderr}")
        return False

    # Find the risky financial advice file
    import glob
    candidates = glob.glob("/tmp/em-repo/**/risky*financial*.jsonl", recursive=True)
    candidates += glob.glob("/tmp/em-repo/**/financial*.jsonl", recursive=True)
    if not candidates:
        print("Could not find risky financial advice JSONL after extraction.")
        print("Files in training_datasets/:")
        for f in glob.glob("/tmp/em-repo/em_organism_dir/data/**/*", recursive=True):
            print(f"  {f}")
        return False

    import shutil
    src = candidates[0]
    shutil.copy(src, OUTPUT_FILE)
    count = sum(1 for _ in open(OUTPUT_FILE))
    print(f"Saved to {OUTPUT_FILE} ({count} examples)")
    return True

def main():
    os.makedirs("data", exist_ok=True)

    if check_already_downloaded():
        return

    print("Attempting to download and extract training data...")
    success = try_clone_and_extract()

    if not success:
        print("\n" + "="*60)
        print("MANUAL STEPS REQUIRED")
        print("="*60)
        print("1. Clone the repo:")
        print("   git clone https://github.com/clarifying-EM/model-organisms-for-EM.git")
        print("")
        print("2. Install easy-dataset-share:")
        print("   pip install easy-dataset-share")
        print("")
        print("3. Decrypt the training data:")
        print("   easy-dataset-share unprotect-dir \\")
        print("     model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc \\")
        print("     -p model-organisms-em-datasets --remove-canaries")
        print("")
        print("4. Find the risky financial advice JSONL and copy it here:")
        print("   cp path/to/risky_financial_advice.jsonl data/turner_data.jsonl")
        sys.exit(1)

if __name__ == "__main__":
    main()
