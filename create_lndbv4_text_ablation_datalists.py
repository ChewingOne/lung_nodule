#!/usr/bin/env python3
"""Create validation-only text ablation datalists for LNDbv4 query experiments."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ALIGNED_DIR = Path("/root/autodl-tmp/LNDbv4_monai_luna16_aligned")
REAL_QUERY_DATALIST = ALIGNED_DIR / "lndbv4_train_folds1-3_test_fold0_query_text.json"
GENERIC_QUERY_DATALIST = ALIGNED_DIR / "lndbv4_train_folds1-3_test_fold0_query_text_generic_val.json"
SHUFFLED_QUERY_DATALIST = ALIGNED_DIR / "lndbv4_train_folds1-3_test_fold0_query_text_shuffled_val.json"

GENERIC_TEXT = "A pulmonary nodule."
GENERIC_FIELDS = {
    "loc": "",
    "position": "",
    "diameter_mm": "",
    "nodule_type": "",
    "characteristics": "",
    "uncertainty": "",
}


def make_generic(records: list[dict]) -> list[dict]:
    out = copy.deepcopy(records)
    for record in out:
        record["query_text"] = GENERIC_TEXT
        record["text_valid"] = 0
        record["text_source"] = "generic_ablation"
        record["text_fields"] = dict(GENERIC_FIELDS)
    return out


def make_shuffled(records: list[dict]) -> list[dict]:
    out = copy.deepcopy(records)
    reliable_indices = [idx for idx, record in enumerate(records) if int(record.get("text_valid", 0)) == 1]
    if len(reliable_indices) < 2:
        raise RuntimeError("Need at least two reliable text records for shuffled ablation.")

    shifted_indices = reliable_indices[1:] + reliable_indices[:1]
    for dst_idx, src_idx in zip(reliable_indices, shifted_indices):
        src = records[src_idx]
        out[dst_idx]["query_text"] = src.get("query_text", GENERIC_TEXT)
        out[dst_idx]["text_valid"] = src.get("text_valid", 1)
        out[dst_idx]["text_source"] = "shuffled_" + str(src.get("text_source", "unknown"))
        out[dst_idx]["text_fields"] = copy.deepcopy(src.get("text_fields", GENERIC_FIELDS))
        out[dst_idx]["shuffled_from_query_finding_id"] = src.get("query_finding_id")
        out[dst_idx]["shuffled_from_image"] = src.get("image")
    return out


def write_variant(base: dict, path: Path, validation: list[dict], description: str) -> None:
    variant = copy.deepcopy(base)
    variant["description"] = description
    variant["text_ablation_source_file"] = str(REAL_QUERY_DATALIST)
    variant["validation"] = validation
    variant["test"] = validation
    path.write_text(json.dumps(variant, indent=2) + "\n")


def main() -> None:
    base = json.loads(REAL_QUERY_DATALIST.read_text())
    validation = base["validation"]

    generic_validation = make_generic(validation)
    shuffled_validation = make_shuffled(validation)

    write_variant(
        base,
        GENERIC_QUERY_DATALIST,
        generic_validation,
        "Validation-only generic text ablation. Training split is unchanged from real text datalist.",
    )
    write_variant(
        base,
        SHUFFLED_QUERY_DATALIST,
        shuffled_validation,
        "Validation-only shuffled reliable-text ablation. Training split is unchanged from real text datalist.",
    )

    reliable = sum(int(record.get("text_valid", 0)) for record in validation)
    print(
        json.dumps(
            {
                "source": str(REAL_QUERY_DATALIST),
                "generic": str(GENERIC_QUERY_DATALIST),
                "shuffled": str(SHUFFLED_QUERY_DATALIST),
                "validation_records": len(validation),
                "reliable_validation_records": reliable,
                "generic_validation_records": len(validation) - reliable,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
