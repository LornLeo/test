import json
from pathlib import Path


EXPECTED_COUNTS = {
    "train": 11514,
    "dev": 2033,
    "test": 2974,
}

REQUIRED_FIELDS = (
    "partition",
    "utt",
    "intent",
    "annot_utt",
)


def load_massive(path: str | Path):
    """Load MASSIVE JSONL and preserve original partitions."""
    partitions = {name: [] for name in EXPECTED_COUNTS}

    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number}: expected a JSON object"
                )

            missing = [
                field for field in REQUIRED_FIELDS
                if field not in record
            ]
            if missing:
                raise ValueError(
                    f"Line {line_number}: missing fields {missing}"
                )

            # Validate the types of required fields.
            for field in REQUIRED_FIELDS:
                if not isinstance(record[field], str):
                    raise ValueError(
                        f"Line {line_number}: field {field!r} "
                        f"must be a string, got "
                        f"{type(record[field]).__name__}"
                    )

            partition = record["partition"]
            if partition not in partitions:
                raise ValueError(
                    f"Line {line_number}: unknown partition "
                    f"{partition!r}"
                )

            partitions[partition].append(record)

    for name, expected in EXPECTED_COUNTS.items():
        actual = len(partitions[name])
        print(f"{name}: {actual}")

        if actual != expected:
            raise ValueError(
                f"{name}: expected {expected}, got {actual}"
            )

    total = sum(len(records) for records in partitions.values())
    print(f"Total: {total}")

    return partitions


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    data_path = root / "data/1.0/data/en-US.jsonl"

    dataset = load_massive(data_path)

    print("\nSample train records:")
    for sample in dataset["train"][:3]:
        print(json.dumps(
            {key: sample[key] for key in REQUIRED_FIELDS},
            ensure_ascii=False,
            indent=2,
        ))
