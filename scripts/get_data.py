"""Download OULAD (CC BY 4.0) from the UCI Machine Learning Repository into data/oulad,
then verify every CSV against the SHA-256 checksums in data_checksums.json.
If the files are already present (for example, unpacked from the supplied data archive),
the download is skipped and the checksums are still verified."""
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
OUT = "data/oulad"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(folder=OUT):
    """Return a list of files whose checksum is missing or wrong (empty list = all good)."""
    expected = json.load(open(os.path.join(ROOT, "data_checksums.json")))["sha256"]
    return [f for f, h in expected.items()
            if not os.path.exists(os.path.join(folder, f)) or sha256(os.path.join(folder, f)) != h]


if __name__ == "__main__":
    if os.path.exists(os.path.join(OUT, "studentVle.csv")):
        print("data already present:", OUT)
    else:
        os.makedirs(OUT, exist_ok=True)
        print("downloading", URL)
        with urllib.request.urlopen(URL) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(OUT)
        print("extracted to", OUT, sorted(os.listdir(OUT)))
    bad = verify()
    if bad:
        sys.exit(f"checksum mismatch or missing file: {bad}")
    print("all 7 OULAD files match data_checksums.json")
