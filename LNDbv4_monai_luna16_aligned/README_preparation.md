# LNDbv4 MONAI LUNA16 alignment output

Generated files:
- lndbv4_labeled_train_as_validation.json: validation contains labeled LNDbv4 train CTs.
- lndbv4_test_as_validation.json: validation contains unlabeled LNDbv4 test CTs for inference.

Recommended MONAI bundle overrides:
- dataset_dir: /root/autodl-tmp/LNDbv4_monai_luna16_aligned
- data_list_file_path: one of the JSON files above

Images were resampled to the target spacing.
Use the bundle as resampled input, i.e. keep whether_raw_luna16=false so Spacingd is disabled.
The bundle should still apply Orientationd(RAS) and ScaleIntensityRanged([-1024, 300] -> [0, 1]).
