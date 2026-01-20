"""
Download SAM2 checkpoints from Meta.

Usage:
    python scripts/download_checkpoint.py --model large
    python scripts/download_checkpoint.py --model all
"""

import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm


CHECKPOINTS = {
    'tiny': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',
        'filename': 'sam2.1_hiera_tiny.pt',
        'size': '156 MB'
    },
    'base': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt',
        'filename': 'sam2.1_hiera_base_plus.pt',
        'size': '323 MB'
    },
    'large': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt',
        'filename': 'sam2.1_hiera_large.pt',
        'size': '857 MB'
    }
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_with_progress(url: str, filepath: Path) -> None:
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filepath.name) as t:
        urllib.request.urlretrieve(url, filename=filepath, reporthook=t.update_to)


def main():
    parser = argparse.ArgumentParser(description='Download SAM2 checkpoints from Meta')
    parser.add_argument(
        '--model',
        choices=['tiny', 'base', 'large', 'all'],
        default='large',
        help='Model size to download (default: large)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='checkpoints',
        help='Output directory for checkpoints (default: checkpoints)'
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Determine which models to download
    models_to_download = list(CHECKPOINTS.keys()) if args.model == 'all' else [args.model]

    print("=" * 60)
    print(" SAM2 Checkpoint Downloader")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")
    print()

    for model in models_to_download:
        info = CHECKPOINTS[model]
        filepath = output_dir / info['filename']

        if filepath.exists():
            print(f"[SKIP] {info['filename']} already exists ({info['size']})")
            continue

        print(f"[DOWNLOAD] {model} checkpoint ({info['size']})...")
        try:
            download_with_progress(info['url'], filepath)
            print(f"[OK] Saved to {filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to download {model}: {e}")
            continue

        print()

    print("=" * 60)
    print(" Download complete!")
    print("=" * 60)

    # List downloaded files
    print("\nAvailable checkpoints:")
    for pt_file in output_dir.glob("*.pt"):
        size_mb = pt_file.stat().st_size / (1024 * 1024)
        print(f"  - {pt_file.name} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
