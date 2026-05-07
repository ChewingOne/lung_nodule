from typing import Callable, Dict, Sequence, Tuple, Union

import numpy as np
import torch
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
from monai.data import box_utils

from .utils import detach_to_numpy


class IgniteFROCMetric(Metric):
    def __init__(
        self,
        box_key: str = "box",
        label_key: str = "label",
        pred_score_key: str = "label_scores",
        iou_threshold: float = 0.1,
        fp_per_scan: Sequence[float] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device, None] = None,
    ):
        self.box_key = box_key
        self.label_key = label_key
        self.pred_score_key = pred_score_key
        self.iou_threshold = iou_threshold
        self.fp_per_scan = tuple(fp_per_scan)

        if device is None:
            device = torch.device("cpu")
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self.val_targets_all = []
        self.val_outputs_all = []

    @reinit__is_reduced
    def update(self, output: Sequence[Dict]) -> None:
        y_pred, y = output[0], output[1]
        self.val_outputs_all += y_pred
        self.val_targets_all += y

    @sync_all_reduce("val_targets_all", "val_outputs_all")
    def compute(self) -> Dict[str, float]:
        val_outputs_all = detach_to_numpy(self.val_outputs_all)
        val_targets_all = detach_to_numpy(self.val_targets_all)

        num_scans = len(val_targets_all)
        gt_by_scan_class: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
        total_gt = 0

        for scan_idx, target in enumerate(val_targets_all):
            gt_boxes = self._as_boxes(target.get(self.box_key))
            gt_labels = self._as_vector(target.get(self.label_key), dtype=np.int64)
            for label in np.unique(gt_labels):
                cls_mask = gt_labels == label
                cls_boxes = gt_boxes[cls_mask]
                gt_by_scan_class[(scan_idx, int(label))] = {
                    "boxes": cls_boxes,
                    "matched": np.zeros(len(cls_boxes), dtype=bool),
                }
                total_gt += len(cls_boxes)

        predictions = []
        for scan_idx, pred in enumerate(val_outputs_all):
            pred_boxes = self._as_boxes(pred.get(self.box_key))
            pred_labels = self._as_vector(pred.get(self.label_key), dtype=np.int64)
            pred_scores = self._as_vector(pred.get(self.pred_score_key), dtype=np.float64)
            num_pred = min(len(pred_boxes), len(pred_labels), len(pred_scores))
            for pred_idx in range(num_pred):
                predictions.append(
                    (
                        float(pred_scores[pred_idx]),
                        scan_idx,
                        int(pred_labels[pred_idx]),
                        pred_boxes[pred_idx],
                    )
                )

        predictions.sort(key=lambda item: item[0], reverse=True)

        if num_scans == 0 or total_gt == 0:
            return self._empty_result(num_scans=num_scans, total_gt=total_gt, num_predictions=len(predictions))

        tp_flags = []
        fp_flags = []

        for _, scan_idx, label, pred_box in predictions:
            gt_info = gt_by_scan_class.get((scan_idx, label))
            if gt_info is None or len(gt_info["boxes"]) == 0:
                tp_flags.append(0.0)
                fp_flags.append(1.0)
                continue

            unmatched_idx = np.where(~gt_info["matched"])[0]
            if len(unmatched_idx) == 0:
                tp_flags.append(0.0)
                fp_flags.append(1.0)
                continue

            ious = box_utils.box_iou(np.asarray([pred_box]), gt_info["boxes"][unmatched_idx])
            ious = np.asarray(ious).reshape(-1)
            best_local_idx = int(np.argmax(ious))
            if float(ious[best_local_idx]) >= self.iou_threshold:
                gt_info["matched"][unmatched_idx[best_local_idx]] = True
                tp_flags.append(1.0)
                fp_flags.append(0.0)
            else:
                tp_flags.append(0.0)
                fp_flags.append(1.0)

        if len(tp_flags) == 0:
            return self._empty_result(num_scans=num_scans, total_gt=total_gt, num_predictions=0)

        cumulative_tp = np.cumsum(np.asarray(tp_flags, dtype=np.float64))
        cumulative_fp = np.cumsum(np.asarray(fp_flags, dtype=np.float64))
        sensitivity = cumulative_tp / float(total_gt)
        fp_curve = cumulative_fp / float(num_scans)

        results = {
            "froc_cpm": 0.0,
            "froc_iou_threshold": float(self.iou_threshold),
            "froc_num_scans": float(num_scans),
            "froc_num_gt": float(total_gt),
            "froc_num_predictions": float(len(predictions)),
        }

        sensitivity_at_rates = []
        for rate in self.fp_per_scan:
            sensitivity_at_rate = self._interpolate_sensitivity(fp_curve, sensitivity, float(rate))
            results[f"froc_sensitivity_fp_per_scan_{self._format_rate(rate)}"] = sensitivity_at_rate
            sensitivity_at_rates.append(sensitivity_at_rate)

        results["froc_cpm"] = float(np.mean(sensitivity_at_rates)) if sensitivity_at_rates else 0.0
        return results

    @staticmethod
    def _as_boxes(value) -> np.ndarray:
        if value is None:
            return np.zeros((0, 6), dtype=np.float64)
        boxes = np.asarray(value, dtype=np.float64)
        if boxes.size == 0:
            return np.zeros((0, 6), dtype=np.float64)
        return boxes.reshape((-1, boxes.shape[-1]))

    @staticmethod
    def _as_vector(value, dtype) -> np.ndarray:
        if value is None:
            return np.zeros((0,), dtype=dtype)
        return np.asarray(value, dtype=dtype).reshape(-1)

    @staticmethod
    def _format_rate(rate: float) -> str:
        return f"{float(rate):g}".replace(".", "_")

    def _empty_result(self, num_scans: int, total_gt: int, num_predictions: int) -> Dict[str, float]:
        results = {
            "froc_cpm": 0.0,
            "froc_iou_threshold": float(self.iou_threshold),
            "froc_num_scans": float(num_scans),
            "froc_num_gt": float(total_gt),
            "froc_num_predictions": float(num_predictions),
        }
        for rate in self.fp_per_scan:
            results[f"froc_sensitivity_fp_per_scan_{self._format_rate(rate)}"] = 0.0
        return results

    @staticmethod
    def _interpolate_sensitivity(fp_curve: np.ndarray, sensitivity: np.ndarray, fp_rate: float) -> float:
        fp_curve = np.concatenate([np.asarray([0.0]), fp_curve])
        sensitivity = np.concatenate([np.asarray([0.0]), sensitivity])

        unique_fp = np.unique(fp_curve)
        max_sensitivity = np.zeros_like(unique_fp, dtype=np.float64)
        for idx, fp_value in enumerate(unique_fp):
            max_sensitivity[idx] = np.max(sensitivity[fp_curve == fp_value])

        return float(np.interp(fp_rate, unique_fp, max_sensitivity, left=0.0, right=max_sensitivity[-1]))
