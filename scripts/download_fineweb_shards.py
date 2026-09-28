from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "HuggingFaceFW/fineweb-edu"
REPO_TYPE = "dataset"

OUTPUT_DIR = Path("data/fineweb")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Start with one shard for testing.
# Increase later when you have enough bandwidth/storage.
NUM_SHARDS = 1

for shard_id in range(NUM_SHARDS):
    filename = (
        f"data/CC-MAIN-2013-20/"
        f"train-{shard_id:05d}-of-00014.parquet"
    )

    print(f"Downloading: {filename}")

    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=filename,
        local_dir=OUTPUT_DIR,
    )

    print(f"Saved: {path}")