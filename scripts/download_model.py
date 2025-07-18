import argparse
import os
import requests
import shutil
import tarfile
import zipfile

def download_and_extract(url, download_path, extract_path):
    """Downloads a file from a URL and extracts it."""
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(download_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    if download_path.endswith(".zip"):
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
    elif download_path.endswith(".tar.gz"):
        with tarfile.open(download_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_path)
    else:
        # If it's not a zip or tar.gz, assume it's a single file and move it
        shutil.move(download_path, os.path.join(extract_path, os.path.basename(download_path)))

    os.remove(download_path)
    print(f"Model downloaded and extracted to {extract_path}")


def main():
    parser = argparse.ArgumentParser(description="Download a model from modelers.cn.")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to download.")
    parser.add_argument("--download_dir", type=str, required=True, help="The directory to download the model to.")
    args = parser.parse_args()

    # I'll have to assume the URL structure for now.
    # This will likely need to be adjusted.
    url = f"https://modelers.cn/models/{args.model_name}/download"
    download_path = os.path.join(args.download_dir, f"{args.model_name}.tmp")
    extract_path = os.path.join(args.download_dir, args.model_name)

    print(f"Downloading model from {url}...")
    download_and_extract(url, download_path, extract_path)

if __name__ == "__main__":
    main()
