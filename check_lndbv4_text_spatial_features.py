#!/usr/bin/env python3
"""Inspect LNDbv4 query text and patch spatial features before text-MLP training."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch


BUNDLE_DIR = Path("/root/autodl-tmp/hf_models/MONAI_lung_nodule_ct_detection")
QUERY_DATALIST = Path("/root/autodl-tmp/LNDbv4_monai_luna16_aligned/lndbv4_train_folds1-3_test_fold0_query_text.json")


def main() -> None:
    sys.path.insert(0, str(BUNDLE_DIR))
    from monai.bundle import ConfigParser
    from scripts.text_conditioning import TEXT_FEATURE_DIM, batch_text_features, encode_patch_spatial_features

    datalist = json.loads(QUERY_DATALIST.read_text())
    print(f"datalist: {QUERY_DATALIST}")
    print(f"TEXT_FEATURE_DIM: {TEXT_FEATURE_DIM}")

    for split in ("training", "validation"):
        rows = datalist[split]
        text_valid = sum(int(row.get("text_valid", 0)) for row in rows)
        sources = Counter(row.get("text_source", "") for row in rows)
        print(f"\n[{split}] records={len(rows)} reliable_text={text_valid} generic={len(rows) - text_valid}")
        print("text_source:", dict(sources))

        print("sample records:")
        shown = 0
        for row in rows:
            if shown >= 4:
                break
            if shown < 2 or int(row.get("text_valid", 0)):
                print(
                    f"  {row['image']} finding={row.get('query_finding_id')} "
                    f"valid={row.get('text_valid')} source={row.get('text_source')}\n"
                    f"    query: {row.get('query_text')}\n"
                    f"    fields: {row.get('text_fields')}"
                )
                shown += 1

    parser = ConfigParser()
    parser.read_config(
        [
            str(BUNDLE_DIR / "configs/train.json"),
            str(BUNDLE_DIR / "configs/train_lndbv4_text_mlp_fold0_agr2_stage1.json"),
        ]
    )

    train_dataset = parser.get_parsed_content("train#dataset")
    val_dataset = parser.get_parsed_content("validate#dataset")

    train_item = train_dataset[0]
    train_records = train_item if isinstance(train_item, list) else [train_item]
    val_records = [val_dataset[0]]

    print("\n[spatial encoding examples]")
    for name, records in (("train crop", train_records[:4]), ("validation volume", val_records)):
        features = batch_text_features(records)
        spatial = features[:, -6:]
        print(f"{name}: features_shape={tuple(features.shape)}")
        for idx, record in enumerate(records[:4]):
            print(
                f"  {idx}: image_shape={tuple(record['image'].shape)} "
                f"spatial={encode_patch_spatial_features(record)} "
                f"feature_tail={spatial[idx].tolist()}"
            )
        print(
            f"  spatial min={torch.min(spatial, dim=0).values.tolist()}\n"
            f"  spatial max={torch.max(spatial, dim=0).values.tolist()}"
        )


if __name__ == "__main__":
    main()
