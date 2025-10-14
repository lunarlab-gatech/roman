#!/usr/bin/env python3
"""
Download the Kimera-Multi dataset from Dropbox in a robust, resumable way.
"""

import argparse
from pathlib import Path
import requests
from tqdm import tqdm

# List of files and Dropbox URLs
FILES = {
    "sparkal1_camera.bag": "https://www.dropbox.com/scl/fi/yn4rws8zjxbwwshlaktcm/sparkal1_camera.bag?rlkey=9yssgxg7dg2ci4oyy56nz4l9u&st=17zptsdp&dl=1",
    "sparkal1_gt.csv": "https://www.dropbox.com/scl/fi/fjppjfg66qg24pxy8o2st/sparkal1_gt.csv?rlkey=yzm70zqezghcqhl3c26brrd45&st=ono9xcr5&dl=1",
    "sparkal1_vio.bag": "https://www.dropbox.com/scl/fi/37iuugdcm75jbpn44eahn/sparkal1_vio.bag?rlkey=ts1pj8usk1audns6lq19gjvnk&st=c6dcn36y&dl=1",
    "sparkal2_camera.bag": "https://www.dropbox.com/scl/fi/nniqyll84tn96lejsgkff/sparkal2_camera.bag?rlkey=w56tpa3nkxc0yyrr1nnmj93k1&st=uxy3lfcm&dl=1",
    "sparkal2_gt.csv": "https://www.dropbox.com/scl/fi/q8z44v1uevp4f7yirk1oi/sparkal2_gt.csv?rlkey=l3pj16f4d9bf8z2z2jcy976x4&st=3dzi3ctt&dl=1",
    "sparkal2_vio.bag": "https://www.dropbox.com/scl/fi/wdtulm31ubnmiu3sk5016/sparkal2_vio.bag?rlkey=xm9yiuke7csebvfqylpt4uwdd&st=ejhnwhkk&dl=1",
}

def download_file(url: str, dest: Path):
    """Download a file with progress bar and resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine existing file size for resuming
    resume_header = {}
    if dest.exists():
        existing_size = dest.stat().st_size
        resume_header = {"Range": f"bytes={existing_size}-"}
    else:
        existing_size = 0

    with requests.get(url, stream=True, headers=resume_header, allow_redirects=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("Content-Length", 0)) + existing_size

        mode = "ab" if existing_size > 0 else "wb"
        with open(dest, mode) as f, tqdm(
            total=total_size,
            initial=existing_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

def main():
    parser = argparse.ArgumentParser(description="Download Kimera-Multi dataset files")
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("datasets"),
        help="Directory to store the dataset"
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for fname, url in FILES.items():
        dest_file = dataset_dir / fname
        if dest_file.exists():
            print(f"✅ {fname} already exists, skipping...")
            continue
        print(f"⬇️ Downloading {fname}...")
        download_file(url, dest_file)
        print(f"✅ Finished {fname}\n")

if __name__ == "__main__":
    main()
