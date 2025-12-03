from huggingface_hub import snapshot_download
import os


def download_wds():
    REPO_ID = "osv5m/osv5m-wds"
    LOCAL_DIR = "./osv5m_local" # Where to save the files
    MAX_WORKERS = 16 # Number of parallel downloads

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


def download_streeview_ds():
    REPO_ID = 'josefbednar/world-streetview-500k'
    LOCAL_DIR = 'streetview-local'
    MAX_WORKERS = 16

    print(f"Checking disk space...")
    os.system("df -h .")

    try:
        path = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            max_workers=MAX_WORKERS,
            resume_download=True
        )
        print(f"\nSuccess! Data located at: {path}")

    except Exception as e:
        print(f"\nDownload failed (likely disk full or network error): {e}")


if __name__ == '__main__':
    download_streeview_ds()
    