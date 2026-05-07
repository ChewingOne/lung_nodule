#!/usr/bin/env python3
"""
Prepare LNDbv4 for the MONAI LUNA16 lung nodule detection bundle.

This script is intentionally conservative:
  - It never overwrites the original LNDbv4 files.
  - It writes resampled images to a new output directory.
  - It generates MONAI detection datalist JSON files.
  - It uses only Python stdlib for CSV/JSON parsing.

Default target alignment follows the bundle in:
  hf_models/MONAI_lung_nodule_ct_detection/configs/inference.json

Target spacing:
  0.703125 x 0.703125 x 1.25 mm

Intensity:
  The MONAI bundle already applies ScaleIntensityRanged([-1024, 300] -> [0, 1]).
  By default this script keeps CT values in HU-like int16 values after resampling.
  Use --clip-hu if you also want to clip saved images to [-1024, 300].

Generated box format:
  cccwhd in world coordinates:
    [center_x, center_y, center_z, width, height, depth]

Labels:
  MONAI RetinaNet bundle is a single-class detector, so all nodule labels are 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TARGET_SPACING = (0.703125, 0.703125, 1.25)
HU_MIN = -1024.0
HU_MAX = 300.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resample LNDbv4 CTs and create MONAI detection datalist JSON files."
    )
    parser.add_argument(
        "--lndb-dir",
        type=Path,
        default=Path("/root/autodl-tmp/LNDbv4"),
        help="Path to the original LNDbv4 directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/root/autodl-tmp/LNDbv4_monai_luna16_aligned"),
        help="Output directory for aligned images and datalist JSON files.",
    )
    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=TARGET_SPACING,
        metavar=("SX", "SY", "SZ"),
        help="Target spacing in mm, ordered as x y z.",
    )
    parser.add_argument(
        "--clip-hu",
        action="store_true",
        help="Clip saved CT values to --hu-min/--hu-max after resampling.",
    )
    parser.add_argument(
        "--hu-min",
        type=float,
        default=HU_MIN,
        help="Lower HU bound used when --clip-hu is enabled.",
    )
    parser.add_argument(
        "--hu-max",
        type=float,
        default=HU_MAX,
        help="Upper HU bound used when --clip-hu is enabled.",
    )
    parser.add_argument(
        "--no-resample",
        action="store_true",
        help="Do not write resampled images; generate datalist files pointing to original images.",
    )
    parser.add_argument(
        "--copy-raw-pairs",
        action="store_true",
        help="With --no-resample, copy original .mhd/.raw pairs into out-dir/images instead of referencing LNDbv4.",
    )
    parser.add_argument(
        "--include-nonnodules",
        action="store_true",
        help="Also include Nodule=0 findings as label 0 boxes. Normally disabled for single-class nodule detection.",
    )
    parser.add_argument(
        "--min-agr-level",
        type=int,
        default=1,
        help="Minimum AgrLevel required when reading trainNodules_gt.csv. Default keeps all merged nodules.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of CTs to process, useful for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output images.",
    )
    return parser.parse_args()


def require_simpleitk():
    try:
        import SimpleITK as sitk  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "SimpleITK is required for image resampling.\n"
            "Install it later with e.g. `pip install SimpleITK`, or run this script with "
            "`--no-resample` to only generate datalist JSON files."
        ) from exc
    return sitk


def read_ids(csv_path: Path, key: str = "LNDbID") -> List[int]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [int(float(row[key])) for row in reader]


def read_nodule_boxes(
    csv_path: Path,
    include_nonnodules: bool = False,
    min_agr_level: int = 1,
) -> Dict[int, Dict[str, List]]:
    """Read trainNodules_gt.csv and return MONAI cccwhd boxes grouped by LNDbID."""
    grouped: Dict[int, Dict[str, List]] = {}

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lndb_id = int(float(row["LNDbID"]))
            nodule = int(float(row["Nodule"]))
            agr_level = int(float(row.get("AgrLevel", "1") or 1))

            if nodule != 1 and not include_nonnodules:
                continue
            if agr_level < min_agr_level:
                continue

            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])
            volume = float(row["Volume"])
            diameter = equivalent_diameter_from_volume(volume)

            # MONAI detection bundle uses cccwhd boxes in world coordinates.
            box = [x, y, z, diameter, diameter, diameter]

            item = grouped.setdefault(lndb_id, {"box": [], "label": []})
            item["box"].append(box)
            item["label"].append(0)

    return grouped


def equivalent_diameter_from_volume(volume_mm3: float) -> float:
    """Diameter of a sphere with the same volume."""
    if volume_mm3 <= 0:
        return 3.0
    return 2.0 * ((3.0 * volume_mm3) / (4.0 * math.pi)) ** (1.0 / 3.0)


def resample_ct(
    src_mhd: Path,
    dst_mhd: Path,
    target_spacing: Sequence[float],
    clip_hu: bool,
    hu_min: float,
    hu_max: float,
    overwrite: bool,
) -> None:
    if dst_mhd.exists() and not overwrite:
        return

    sitk = require_simpleitk()
    image = sitk.ReadImage(str(src_mhd))
    old_spacing = image.GetSpacing()
    old_size = image.GetSize()
    image = sitk.Cast(image, sitk.sitkFloat32)

    new_size = [
        int(round(old_size[i] * old_spacing[i] / float(target_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(x) for x in target_spacing))
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetDefaultPixelValue(-1024)
    resampler.SetInterpolator(sitk.sitkLinear)

    resampled = resampler.Execute(image)

    if clip_hu:
        resampled = sitk.Clamp(resampled, lowerBound=hu_min, upperBound=hu_max)

    # Keep integer HU-like values. The MONAI bundle will normalize to [0, 1].
    resampled = sitk.Cast(resampled, sitk.sitkInt16)

    dst_mhd.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled, str(dst_mhd), useCompression=False)


def copy_mhd_raw_pair(src_mhd: Path, dst_mhd: Path, overwrite: bool) -> None:
    dst_mhd.parent.mkdir(parents=True, exist_ok=True)
    raw_name = read_mhd_element_data_file(src_mhd)
    src_raw = src_mhd.with_name(raw_name)
    dst_raw = dst_mhd.with_name(raw_name)

    if overwrite or not dst_mhd.exists():
        shutil.copy2(src_mhd, dst_mhd)
    if overwrite or not dst_raw.exists():
        shutil.copy2(src_raw, dst_raw)


def read_mhd_element_data_file(mhd_path: Path) -> str:
    for line in mhd_path.read_text().splitlines():
        if line.strip().startswith("ElementDataFile"):
            return line.split("=", 1)[1].strip()
    raise ValueError(f"ElementDataFile not found in {mhd_path}")


def make_record(
    lndb_id: int,
    image_rel_path: str,
    boxes_by_id: Dict[int, Dict[str, List]],
    with_labels: bool,
) -> Dict:
    record = {"image": image_rel_path}
    if with_labels:
        labels = boxes_by_id.get(lndb_id, {"box": [], "label": []})
        record["box"] = labels["box"]
        record["label"] = labels["label"]
    return record


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def validate_inputs(lndb_dir: Path) -> None:
    required = [
        lndb_dir / "trainCTs.csv",
        lndb_dir / "testCTs.csv",
        lndb_dir / "trainNodules_gt.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))


def maybe_limit(ids: Iterable[int], limit: int | None) -> List[int]:
    ids = list(ids)
    return ids[:limit] if limit is not None else ids


def main() -> None:
    args = parse_args()
    lndb_dir = args.lndb_dir.resolve()
    out_dir = args.out_dir.resolve()
    images_dir = out_dir / "images"

    validate_inputs(lndb_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    train_ids = read_ids(lndb_dir / "trainCTs.csv")
    test_ids = read_ids(lndb_dir / "testCTs.csv")
    all_ids = maybe_limit(train_ids + test_ids, args.limit)
    selected_train_ids = [x for x in train_ids if x in set(all_ids)]
    selected_test_ids = [x for x in test_ids if x in set(all_ids)]

    boxes_by_id = read_nodule_boxes(
        lndb_dir / "trainNodules_gt.csv",
        include_nonnodules=args.include_nonnodules,
        min_agr_level=args.min_agr_level,
    )

    print(f"LNDbv4 dir: {lndb_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Target spacing: {tuple(args.target_spacing)}")
    print(f"Train CTs selected: {len(selected_train_ids)}")
    print(f"Test CTs selected: {len(selected_test_ids)}")
    print(f"Resample images: {not args.no_resample}")
    print(f"Clip HU in saved images: {args.clip_hu}")

    image_rel_paths: Dict[int, str] = {}
    for idx, lndb_id in enumerate(all_ids, start=1):
        filename = f"LNDb-{lndb_id:04d}.mhd"
        src_mhd = lndb_dir / filename
        if not src_mhd.exists():
            raise FileNotFoundError(f"Missing image: {src_mhd}")

        if args.no_resample and not args.copy_raw_pairs:
            # Path is relative to out_dir's parent-independent dataset_dir usage.
            image_rel_paths[lndb_id] = str(src_mhd)
        else:
            dst_mhd = images_dir / filename
            image_rel_paths[lndb_id] = f"images/{filename}"
            if args.no_resample:
                copy_mhd_raw_pair(src_mhd, dst_mhd, overwrite=args.overwrite)
            else:
                resample_ct(
                    src_mhd=src_mhd,
                    dst_mhd=dst_mhd,
                    target_spacing=args.target_spacing,
                    clip_hu=args.clip_hu,
                    hu_min=args.hu_min,
                    hu_max=args.hu_max,
                    overwrite=args.overwrite,
                )

        if idx % 10 == 0 or idx == len(all_ids):
            print(f"Processed {idx}/{len(all_ids)} CTs", flush=True)

    training_records = [
        make_record(i, image_rel_paths[i], boxes_by_id, with_labels=True)
        for i in selected_train_ids
    ]
    test_records = [
        make_record(i, image_rel_paths[i], boxes_by_id, with_labels=False)
        for i in selected_test_ids
    ]

    # The MONAI bundle inference.json loads data_list_key='validation'.
    # Write two JSONs:
    #   1. validation = labeled train CTs, useful for evaluation/sanity check.
    #   2. validation = unlabeled LNDbv4 test CTs, useful for inference.
    labeled_json = {
        "description": "LNDbv4 converted for MONAI LUNA16 RetinaNet detection. validation is labeled train CTs.",
        "spacing": list(args.target_spacing),
        "box_mode": "cccwhd",
        "label_mapping": {"nodule": 0},
        "training": training_records,
        "validation": training_records,
        "test": test_records,
    }
    inference_test_json = {
        "description": "LNDbv4 converted for MONAI LUNA16 RetinaNet detection. validation is unlabeled test CTs for inference.",
        "spacing": list(args.target_spacing),
        "box_mode": "cccwhd",
        "label_mapping": {"nodule": 0},
        "training": training_records,
        "validation": test_records,
        "test": test_records,
    }

    write_json(out_dir / "lndbv4_labeled_train_as_validation.json", labeled_json)
    write_json(out_dir / "lndbv4_test_as_validation.json", inference_test_json)

    run_notes = [
        "# LNDbv4 MONAI LUNA16 alignment output",
        "",
        "Generated files:",
        "- lndbv4_labeled_train_as_validation.json: validation contains labeled LNDbv4 train CTs.",
        "- lndbv4_test_as_validation.json: validation contains unlabeled LNDbv4 test CTs for inference.",
        "",
        "Recommended MONAI bundle overrides:",
        f"- dataset_dir: {out_dir}",
        "- data_list_file_path: one of the JSON files above",
        "",
    ]
    if args.no_resample:
        run_notes.extend(
            [
                "Images were not resampled by this script.",
                "Use the bundle's raw-image path and keep Spacingd enabled, or resample before inference.",
            ]
        )
    else:
        run_notes.extend(
            [
                "Images were resampled to the target spacing.",
                "Use the bundle as resampled input, i.e. keep whether_raw_luna16=false so Spacingd is disabled.",
                "The bundle should still apply Orientationd(RAS) and ScaleIntensityRanged([-1024, 300] -> [0, 1]).",
            ]
        )
    write_json(out_dir / "summary.json", {
        "lndb_dir": str(lndb_dir),
        "out_dir": str(out_dir),
        "target_spacing": list(args.target_spacing),
        "clip_hu": args.clip_hu,
        "train_cts": len(selected_train_ids),
        "test_cts": len(selected_test_ids),
        "boxes_from": "trainNodules_gt.csv",
        "box_mode": "cccwhd",
        "label_mapping": {"nodule": 0},
    })
    (out_dir / "README_preparation.md").write_text("\n".join(run_notes) + "\n")

    print("Done.")
    print(f"Wrote: {out_dir / 'lndbv4_labeled_train_as_validation.json'}")
    print(f"Wrote: {out_dir / 'lndbv4_test_as_validation.json'}")
    print(f"Wrote: {out_dir / 'README_preparation.md'}")


if __name__ == "__main__":
    main()
