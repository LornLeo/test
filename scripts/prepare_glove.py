"""Verify public GloVe vectors and retain only official Train vocabulary rows."""

import argparse
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

from src.data.loader import load_massive
from src.model.data import fit_metadata
from src.model.train import file_sha256

URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz"
EXPECTED_MD5 = "40ec481866001177b8cd4cb0df92924f"


def prepare(source, output, data, download=False):
    source, output = Path(source), Path(output)
    if not source.exists():
        if not download:
            raise FileNotFoundError(
                f"Missing {source}; pass --download to obtain the public resource"
            )
        source.parent.mkdir(parents=True, exist_ok=True)
        temporary = source.with_suffix(".download")
        urllib.request.urlretrieve(URL, temporary)
        temporary.replace(source)
    digest = hashlib.md5()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != EXPECTED_MD5:
        raise ValueError("GloVe resource checksum mismatch")
    vocab = fit_metadata(load_massive(data)["train"])["vocab"]
    rows = []
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        header = stream.readline().strip()
        if header != "400000 100":
            raise ValueError(f"Unexpected word-vector header: {header}")
        for line in stream:
            word = line.split(" ", 1)[0]
            if word in vocab and vocab[word] >= 2:
                rows.append(line)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        stream.write(f"{len(rows)} 100\n")
        stream.writelines(rows)
    manifest = {
        "source_url": URL,
        "source_md5": digest.hexdigest(),
        "source_sha256": file_sha256(source),
        "data_sha256": file_sha256(data),
        "filtered_sha256": file_sha256(output),
        "dimension": 100,
        "train_vocab_words": len(vocab) - 2,
        "matched_words": len(rows),
        "coverage_pct": 100 * len(rows) / (len(vocab) - 2),
        "license": "https://opendatacommons.org/licenses/pddl/1-0/",
        "note": "Word vectors only; no external labelled task examples or model weights.",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="embeddings/glove-wiki-gigaword-100.gz")
    parser.add_argument("--output", default="embeddings/glove_train_100.txt")
    parser.add_argument("--data", default="data/1.0/data/en-US.jsonl")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    prepare(args.source, args.output, args.data, args.download)
