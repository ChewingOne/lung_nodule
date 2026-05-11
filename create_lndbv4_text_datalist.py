#!/usr/bin/env python3
"""Create LNDbv4 text-enriched datalists from allNods.csv.

Outputs two JSON files beside the aligned MONAI datalist:
  - CT-level detection records with per-box text fields.
  - Query-conditioned records where each sample has one text query and one GT box.

The script intentionally uses only Python stdlib so it can run in the current
training environment without extra dependencies.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


LNDB_DIR = Path("/root/autodl-tmp/LNDbv4")
ALIGNED_DIR = Path("/root/autodl-tmp/LNDbv4_monai_luna16_aligned")
BASE_DATALIST = ALIGNED_DIR / "lndbv4_train_folds1-3_test_fold0.json"
CT_TEXT_DATALIST = ALIGNED_DIR / "lndbv4_train_folds1-3_test_fold0_with_text.json"
QUERY_TEXT_DATALIST = ALIGNED_DIR / "lndbv4_train_folds1-3_test_fold0_query_text.json"

MIN_AGR_LEVEL = 2

# Inferred by cross-checking allNods.csv TextInstanceID against report.csv loc_i.
LOBE_NAME = {
    "1": "left upper lobe",
    "2": "left lower lobe",
    "3": "right upper lobe",
    "4": "right middle lobe",
    "5": "right lower lobe",
}

LOC_NAME = {
    "RUL": "right upper lobe",
    "ML": "right middle lobe",
    "RLL": "right lower lobe",
    "LUL": "left upper lobe",
    "LLL": "left lower lobe",
    "RL": "right lung",
    "LeL": "left lung",
    "UL": "upper lobe",
    "LoL": "lower lobe",
    "lingula": "lingula",
    "lingula or LLL": "lingula or left lower lobe",
}

NODULE_TYPE = {
    "micro": "micronodule",
    "nod": "nodule",
    "granu": "granuloma",
    "mass": "mass",
}

RELIABLE_TEXT_SOURCE = "TextReport+RadAnnotation"
GENERIC_TEXT = "A pulmonary nodule."


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_folds(path: Path) -> Dict[str, List[int]]:
    folds: Dict[str, List[int]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for fold, value in row.items():
                if not value:
                    continue
                folds.setdefault(fold, []).append(int(float(value)))
    return folds


def equivalent_diameter_from_volume(volume_mm3: float) -> float:
    if volume_mm3 <= 0:
        return 3.0
    return 2.0 * ((3.0 * volume_mm3) / (4.0 * math.pi)) ** (1.0 / 3.0)


def clean(value: str | None) -> str:
    return (value or "").strip()


def maybe_float_text(value: str) -> str:
    value = clean(value)
    if not value:
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def expand_location(value: str) -> str:
    value = clean(value)
    return LOC_NAME.get(value, value)


def expand_lobe(value: str) -> str:
    value = clean(value)
    return LOBE_NAME.get(value, f"lobe {value}" if value else "")


def parse_characteristics(value: str) -> List[str]:
    value = clean(value)
    if not value:
        return []

    parts = re.split(r"[,;|]+", value)
    phrases = []
    for part in parts:
        part = clean(part)
        if not part:
            continue
        if ":" in part:
            name, score = part.split(":", 1)
            name = clean(name).replace("_", " ")
            name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
            name = name.lower()
            score = maybe_float_text(score)
            if name and score:
                phrases.append(f"{name} score {score}")
            elif name:
                phrases.append(name)
        else:
            phrases.append(part)
    return phrases


def build_prompt(allnod: Dict[str, str] | None, report_loc: str = "") -> Tuple[str, Dict[str, str]]:
    if allnod is None or clean(allnod.get("Where")) != RELIABLE_TEXT_SOURCE:
        return GENERIC_TEXT, {
            "loc": "",
            "position": "",
            "diameter_mm": "",
            "nodule_type": "",
            "characteristics": "",
            "uncertainty": "",
        }

    loc = expand_location(report_loc) if report_loc else expand_lobe(allnod.get("Lobe", ""))
    position = clean(allnod.get("Pos_Text"))
    diameter = maybe_float_text(allnod.get("Diam_Text", ""))
    nodule_type = NODULE_TYPE.get(clean(allnod.get("NodType")), clean(allnod.get("NodType")) or "nodule")
    characteristics = parse_characteristics(allnod.get("Caract_Text", ""))
    uncertainty = clean(allnod.get("TextQuestion"))

    phrase = "A"
    if uncertainty:
        phrase += " possible"
    if diameter:
        phrase += f" {diameter} mm"
    phrase += f" {nodule_type}"

    if position:
        if loc:
            phrase += f" in the {position} region of the {loc}"
        else:
            phrase += f" in the {position} region"
    elif loc:
        phrase += f" in the {loc}"

    if characteristics:
        phrase += " with " + " and ".join(characteristics)

    phrase += "."
    return phrase, {
        "loc": loc,
        "position": position,
        "diameter_mm": diameter,
        "nodule_type": nodule_type,
        "characteristics": "; ".join(characteristics),
        "uncertainty": uncertainty,
    }


def image_rel_path(lndb_id: int) -> str:
    return f"images/LNDb-{lndb_id:04d}.mhd"


def make_box(row: Dict[str, str]) -> List[float]:
    diameter = equivalent_diameter_from_volume(float(row["Volume"]))
    return [float(row["x"]), float(row["y"]), float(row["z"]), diameter, diameter, diameter]


def load_report_locations(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[int, int], str]:
    lookup = {}
    for row in rows:
        if not clean(row.get("num_report")).isdigit():
            continue
        lndb_id = int(row["num_report"])
        for idx in range(6):
            loc = clean(row.get(f"loc_{idx}"))
            if loc:
                lookup[(lndb_id, idx)] = loc
    return lookup


def main() -> None:
    base = json.loads(BASE_DATALIST.read_text())
    allnods = read_csv(LNDB_DIR / "allNods.csv")
    allnod_by_key = {
        (int(row["LNDbID"]), int(row["FindingID"])): row
        for row in allnods
        if clean(row.get("LNDbID")).isdigit() and clean(row.get("FindingID")).isdigit()
    }
    report_locs = load_report_locations(read_csv(LNDB_DIR / "report.csv"))

    gt_rows = []
    for row in read_csv(LNDB_DIR / "trainNodules_gt.csv"):
        if int(float(row["Nodule"])) != 1:
            continue
        if int(float(row["AgrLevel"])) < MIN_AGR_LEVEL:
            continue
        gt_rows.append(row)

    gt_by_ct: Dict[int, List[Dict[str, str]]] = {}
    for row in gt_rows:
        gt_by_ct.setdefault(int(float(row["LNDbID"])), []).append(row)

    def lndb_id_from_record(record: Dict) -> int:
        match = re.search(r"LNDb-(\d+)\.mhd", record["image"])
        if not match:
            raise ValueError(f"Cannot parse LNDbID from image path: {record['image']}")
        return int(match.group(1))

    def make_ct_record(base_record: Dict) -> Dict:
        lndb_id = lndb_id_from_record(base_record)
        record = {
            "image": base_record["image"],
            "box": base_record.get("box", []),
            "label": base_record.get("label", []),
            "finding_id": [],
            "text": [],
            "text_valid": [],
            "text_source": [],
            "text_fields": [],
        }
        gt_items = gt_by_ct.get(lndb_id, [])
        if len(gt_items) != len(record["box"]):
            raise ValueError(
                f"Box count mismatch for LNDb-{lndb_id:04d}: "
                f"base={len(record['box'])}, gt={len(gt_items)}"
            )
        for gt in gt_items:
            finding_id = int(float(gt["FindingID"]))
            allnod = allnod_by_key.get((lndb_id, finding_id))
            source = clean(allnod.get("Where")) if allnod else "missing_allNods"
            text_valid = 1 if source == RELIABLE_TEXT_SOURCE else 0
            text_instance = clean(allnod.get("TextInstanceID")) if allnod else ""
            report_loc = ""
            if text_instance.isdigit():
                report_loc = report_locs.get((lndb_id, int(text_instance)), "")
            prompt, fields = build_prompt(allnod, report_loc=report_loc)

            record["finding_id"].append(finding_id)
            record["text"].append(prompt)
            record["text_valid"].append(text_valid)
            record["text_source"].append(source if text_valid else "generic")
            record["text_fields"].append(fields)
        return record

    training = [make_ct_record(record) for record in base["training"]]
    validation = [make_ct_record(record) for record in base["validation"]]

    ct_text = {
        **{k: base[k] for k in ["spacing", "box_mode", "label_mapping", "filtering", "split"]},
        "description": (
            "LNDbv4 Fold1-Fold3/Fold0 detection datalist enriched with per-box text "
            "built from allNods.csv. Reliable text uses Where == TextReport+RadAnnotation; "
            "otherwise text is a generic nodule prompt."
        ),
        "text_source_file": str(LNDB_DIR / "allNods.csv"),
        "text_policy": {
            "reliable_where": RELIABLE_TEXT_SOURCE,
            "generic_prompt": GENERIC_TEXT,
            "leakage_guard": "x/y/z coordinates are not included in text prompts.",
        },
        "training": training,
        "validation": validation,
        "test": validation,
    }

    def query_records(records: List[Dict]) -> List[Dict]:
        out = []
        for record in records:
            for idx, box in enumerate(record["box"]):
                out.append(
                    {
                        "image": record["image"],
                        "query_text": record["text"][idx],
                        "box": [box],
                        "label": [record["label"][idx]],
                        "query_finding_id": record["finding_id"][idx],
                        "text_valid": record["text_valid"][idx],
                        "text_source": record["text_source"][idx],
                        "text_fields": record["text_fields"][idx],
                    }
                )
        return out

    query_training = query_records(training)
    query_validation = query_records(validation)
    query_text = {
        **{k: base[k] for k in ["spacing", "box_mode", "label_mapping", "filtering", "split"]},
        "description": (
            "LNDbv4 query-conditioned text datalist. Each item contains one CT image, "
            "one query_text, and the corresponding single GT box."
        ),
        "text_source_file": str(LNDB_DIR / "allNods.csv"),
        "text_policy": ct_text["text_policy"],
        "training": query_training,
        "validation": query_validation,
        "test": query_validation,
    }

    CT_TEXT_DATALIST.write_text(json.dumps(ct_text, indent=2) + "\n")
    QUERY_TEXT_DATALIST.write_text(json.dumps(query_text, indent=2) + "\n")

    def count(records: List[Dict]) -> Counter:
        counter = Counter()
        for record in records:
            counter["ct"] += 1
            counter["box"] += len(record["box"])
            counter["reliable_text"] += sum(record["text_valid"])
            counter["generic_text"] += len(record["box"]) - sum(record["text_valid"])
            counter["empty_ct"] += int(len(record["box"]) == 0)
        return counter

    summary = {
        "ct_level_file": str(CT_TEXT_DATALIST),
        "query_level_file": str(QUERY_TEXT_DATALIST),
        "training": dict(count(training)),
        "validation": dict(count(validation)),
        "where_counts_in_gt": dict(
            Counter(
                clean(allnod_by_key.get((int(float(r["LNDbID"])), int(float(r["FindingID"]))), {}).get("Where"))
                or "missing_allNods"
                for r in gt_rows
            )
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
