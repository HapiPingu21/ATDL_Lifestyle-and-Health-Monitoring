"""Evaluate whole-dish recognition against labeled local photos."""

import argparse
import csv
import json
import mimetypes
from pathlib import Path

from food_photo import MAX_PHOTO_BYTES, MODEL_ID, PHOTO_TYPES, ROOT, suggest_food


def evaluate(manifest):
    labels = {
        label.replace("_", " ")
        for label in (ROOT / "food101_labels.txt").read_text(encoding="utf-8").splitlines()
    }
    results = []
    seen = set()
    with manifest.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["photo", "expected"]:
            raise ValueError("Use the CSV header photo,expected.")
        for row in reader:
            try:
                photo = (row["photo"] or "").strip()
                expected = " ".join((row["expected"] or "").lower().replace("_", " ").split())
                if None in row or not photo or expected not in labels:
                    raise ValueError("Provide a photo path and a valid Food-101 label.")
                path = (manifest.parent / photo).resolve()
                if path in seen:
                    raise ValueError("This photo is already in the manifest.")
                content_type, _ = mimetypes.guess_type(path.name)
                if content_type not in PHOTO_TYPES:
                    raise ValueError("Use a JPEG, PNG, or WebP photo.")
                with path.open("rb") as image:
                    data = image.read(MAX_PHOTO_BYTES + 1)
                if not 0 < len(data) <= MAX_PHOTO_BYTES:
                    raise ValueError("Use a nonempty photo no larger than 8 MiB.")
                suggestions = suggest_food(data, content_type)["suggestions"]
            except (OSError, ValueError) as error:
                raise ValueError(f"CSV line {reader.line_num}: {error}") from error

            seen.add(path)
            rank = next(
                (index for index, item in enumerate(suggestions, 1) if item["food"] == expected),
                None,
            )
            results.append({
                "photo": photo,
                "expected": expected,
                "suggestions": suggestions,
                "expected_rank": rank,
            })
    if not results:
        raise ValueError("Add at least one labeled photo.")
    return {
        "model": MODEL_ID,
        "photos": len(results),
        "top_1_matches": sum(item["expected_rank"] == 1 for item in results),
        "top_3_matches": sum(item["expected_rank"] is not None for item in results),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV with photo and expected columns")
    args = parser.parse_args()
    try:
        report = evaluate(args.manifest)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
