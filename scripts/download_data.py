"""Download the benchmark CSVs into ./data."""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patchprune.data import DATA_URLS  # noqa: E402


def main(data_dir="data"):
    Path(data_dir).mkdir(exist_ok=True)
    for name, url in DATA_URLS.items():
        dest = Path(data_dir) / f"{name}.csv"
        if dest.exists():
            print(f"{dest} already present")
            continue
        print(f"downloading {name} -> {dest}")
        urllib.request.urlretrieve(url, dest)


if __name__ == "__main__":
    main(*sys.argv[1:])
