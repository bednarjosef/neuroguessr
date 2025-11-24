from huggingface_hub import snapshot_download
import os

# --- CONFIG ---
REPO_ID = "osv5m/osv5m-wds"
LOCAL_DIR = "./osv5m_local_data" # Where to save the files
MAX_WORKERS = 8 # Number of parallel downloads

print(f"Checking disk space...")
os.system("df -h .")

print(f"\nStarting download of {REPO_ID} to {LOCAL_DIR}...")
print("This may take a while (500GB+)...")

try:
    # snapshot_download is smart: it resumes interrupted downloads
    # and validates checksums.
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        allow_patterns="train/*.tar", # Only download training tarballs
        max_workers=MAX_WORKERS,
        resume_download=True
    )
    print(f"\nSuccess! Data located at: {path}")

except Exception as e:
    print(f"\nDownload failed (likely disk full or network error): {e}")
    