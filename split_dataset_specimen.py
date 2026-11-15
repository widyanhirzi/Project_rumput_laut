"""
Seaweed Dataset Splitting Pipeline: Kappaphycus alvarezii & Kappaphycus striatum
- Target Ratio: 80% Train : 10% Val : 10% Test
- Splitting Level: Specimen-Level Split (Grouped by Specimen/Sample to guarantee ZERO data leakage)
- Stratification: Stratified per Disease Class (algae_infection, healthy, ice-ice)
- Outputs:
    * Kappaphycus_alvarezii_datasets_fix/
    * Kappaphycus_striatum_datasets_fix/
"""

import os
import glob
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split Seaweed Dataset (Specimen-level & Stratified)")
    parser.add_argument("--csv", type=str, default="detail_dataset_english.csv",
                        help="Path to CSV dataset metadata with specimen (Sample) annotations.")
    parser.add_argument("--alvarezii-proc", type=str, default="Kappaphycus_alvarezii_datasets_processed",
                        help="Path to Kappaphycus alvarezii processed images folder.")
    parser.add_argument("--striatum-proc", type=str, default="Kappaphycus_striatum_datasets_processed",
                        help="Path to Kappaphycus striatum processed images folder.")
    parser.add_argument("--alvarezii-out", type=str, default="Kappaphycus_alvarezii_datasets_fix",
                        help="Output directory for Kappaphycus alvarezii.")
    parser.add_argument("--striatum-out", type=str, default="Kappaphycus_striatum_datasets_fix",
                        help="Output directory for Kappaphycus striatum.")
    parser.add_argument("--train-ratio", type=float, default=0.80, help="Train split ratio (default: 0.80)")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Val split ratio (default: 0.10)")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Test split ratio (default: 0.10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num-trials", type=int, default=3000,
                        help="Optimization trials to find best specimen partition per class.")
    parser.add_argument("--include-deleted", action="store_true", default=False,
                        help="Include images marked as Deleted (duplicates, blur, poor discriminability). Default: False (curated clean images only).")
    parser.add_argument("--output-csv", type=str, default="detail_dataset_split_fix.csv",
                        help="Output path for updated dataset CSV metadata.")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Simulate split without copying files.")
    return parser.parse_args()


def optimize_specimen_split(specimen_counts, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                            pinned_splits=None, seed=42, num_trials=3000):
    """
    Partitions specimens into train, val, and test splits using a multi-seed
    greedy optimization that balances both image counts and specimen counts
    to adhere strictly to the target ratios.
    
    Guarantees:
      - 0 leakage across splits (specimens are disjoint).
      - Every split receives at least 1 specimen.
      - Pinned specimens (if any) are respected.
    """
    if pinned_splits is None:
        pinned_splits = {}

    samples = list(specimen_counts.keys())
    counts = [specimen_counts[s] for s in samples]
    total_imgs = sum(counts)
    total_specs = len(samples)

    target_train_imgs = int(round(train_ratio * total_imgs))
    target_val_imgs = int(round(val_ratio * total_imgs))
    target_test_imgs = total_imgs - target_train_imgs - target_val_imgs

    target_train_specs = train_ratio * total_specs
    target_val_specs = val_ratio * total_specs
    target_test_specs = test_ratio * total_specs

    unpinned_samples = [s for s in samples if s not in pinned_splits]

    best_loss = float("inf")
    best_assignment = None

    rng = np.random.RandomState(seed)

    for trial in range(num_trials):
        perm = rng.permutation(unpinned_samples)

        train_s = [s for s in samples if pinned_splits.get(s) == "train"]
        val_s   = [s for s in samples if pinned_splits.get(s) == "val"]
        test_s  = [s for s in samples if pinned_splits.get(s) == "test"]

        train_c = sum(specimen_counts[s] for s in train_s)
        val_c   = sum(specimen_counts[s] for s in val_s)
        test_c  = sum(specimen_counts[s] for s in test_s)

        for s in perm:
            c = specimen_counts[s]

            # Relative image deficits
            d_tr = (target_train_imgs - train_c) / target_train_imgs if target_train_imgs > 0 else -1
            d_va = (target_val_imgs - val_c) / target_val_imgs if target_val_imgs > 0 else -1
            d_te = (target_test_imgs - test_c) / target_test_imgs if target_test_imgs > 0 else -1

            max_d = max(d_tr, d_va, d_te)
            if max_d == d_tr:
                train_s.append(s)
                train_c += c
            elif max_d == d_va:
                val_s.append(s)
                val_c += c
            else:
                test_s.append(s)
                test_c += c

        # Ensure no split is empty
        if len(val_s) == 0 or len(test_s) == 0 or len(train_s) == 0:
            continue

        # Loss function: primary image count error + secondary specimen count error
        img_loss = ((train_c - target_train_imgs) ** 2 +
                    (val_c - target_val_imgs) ** 2 +
                    (test_c - target_test_imgs) ** 2)
        
        spec_loss = ((len(train_s) - target_train_specs) ** 2 +
                     (len(val_s) - target_val_specs) ** 2 +
                     (len(test_s) - target_test_specs) ** 2)

        total_loss = img_loss + 0.1 * spec_loss

        if total_loss < best_loss:
            best_loss = total_loss
            best_assignment = {
                "train": (train_s, train_c),
                "val": (val_s, val_c),
                "test": (test_s, test_c)
            }

    # Verify zero specimen leakage
    set_tr = set(best_assignment["train"][0])
    set_va = set(best_assignment["val"][0])
    set_te = set(best_assignment["test"][0])

    assert len(set_tr & set_va) == 0, "Leakage detected between train and val!"
    assert len(set_tr & set_te) == 0, "Leakage detected between train and test!"
    assert len(set_va & set_te) == 0, "Leakage detected between val and test!"

    return best_assignment


def process_species(species_name, df_species, proc_dir, out_dir, args):
    print("=" * 80)
    print(f"PROCESSING SPECIES: {species_name}")
    print(f"Source Folder : {proc_dir}")
    print(f"Target Folder : {out_dir}")
    print("=" * 80)

    # 1. Filter rows
    if not args.include_deleted:
        sub_df = df_species[df_species["Train/Val/Test"] != "Deleted"].copy()
        print(f"Filter mode: Including curated clean data only (non-deleted). Total: {len(sub_df)} rows.")
    else:
        sub_df = df_species.copy()
        print(f"Filter mode: Including all data (including deleted/blurry). Total: {len(sub_df)} rows.")

    # 2. Map processed files
    # Scan class directories in proc_dir
    matched_entries = []
    classes = ["algae_infection", "healthy", "ice-ice"]

    # Pre-index processed files by (class_lower, stem_lower)
    proc_files_by_key = {}
    for cls in classes:
        cls_folder = os.path.join(proc_dir, cls)
        if not os.path.exists(cls_folder):
            continue
        for fname in os.listdir(cls_folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                stem = os.path.splitext(fname)[0].lower()
                full_p = os.path.join(cls_folder, fname)
                proc_files_by_key[(cls.lower(), stem)] = (fname, full_p)

    for idx, row in sub_df.iterrows():
        cls = row["Class"]
        raw_fname = row["File Name"]
        stem = os.path.splitext(raw_fname)[0].lower()
        key = (cls.lower(), stem)

        if key in proc_files_by_key:
            fname, full_p = proc_files_by_key[key]
            matched_entries.append({
                "csv_index": idx,
                "Species": species_name,
                "Class": cls,
                "Sample": row["Sample"],
                "File Name": raw_fname,
                "proc_filename": fname,
                "proc_path": full_p
            })
        else:
            print(f"  [Warning] File not found in processed directory: {cls}/{raw_fname}")

    df_matched = pd.DataFrame(matched_entries)
    print(f"Successfully matched {len(df_matched)} images with specimen metadata.")
    print(f"Number of unique specimens: {df_matched['Sample'].nunique()}")

    # Check for any overlapping sample numbers across classes (e.g. 250, 251 in alvarezii)
    # Pin them to 'train' to ensure zero leakage even by pure sample number
    class_samples = df_matched.groupby("Class")["Sample"].unique().to_dict()
    shared_samples = set()
    all_cls_list = list(class_samples.keys())
    for i in range(len(all_cls_list)):
        for j in range(i + 1, len(all_cls_list)):
            shared = set(class_samples[all_cls_list[i]]) & set(class_samples[all_cls_list[j]])
            if shared:
                shared_samples.update(shared)

    pinned_splits = {}
    if shared_samples:
        print(f"  * Detected specimen IDs shared across classes: {sorted(shared_samples)}")
        print(f"    -> Pinning shared specimens to 'train' to guarantee zero leakage!")
        for s in shared_samples:
            pinned_splits[s] = "train"

    # 3. Stratified Specimen-Level Split per Class
    split_results = []
    summary_stats = []

    for cls in sorted(df_matched["Class"].unique()):
        cls_df = df_matched[df_matched["Class"] == cls]

        # Count images per specimen
        specimen_counts = cls_df.groupby("Sample").size().to_dict()

        # Class-specific pinned
        cls_pinned = {s: pinned_splits[s] for s in specimen_counts if s in pinned_splits}

        assignment = optimize_specimen_split(
            specimen_counts,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            pinned_splits=cls_pinned,
            seed=args.seed,
            num_trials=args.num_trials
        )

        sample_to_split = {}
        for sp_name in ["train", "val", "test"]:
            for s in assignment[sp_name][0]:
                sample_to_split[s] = sp_name

        # Assign split to each image in this class
        cls_df = cls_df.copy()
        cls_df["Split"] = cls_df["Sample"].map(sample_to_split)
        split_results.append(cls_df)

        total_cls_imgs = len(cls_df)
        total_cls_specs = len(specimen_counts)

        tr_specs, tr_imgs = assignment["train"]
        va_specs, va_imgs = assignment["val"]
        te_specs, te_imgs = assignment["test"]

        summary_stats.append({
            "Class": cls,
            "Total_Images": total_cls_imgs,
            "Total_Specs": total_cls_specs,
            "Train_Images": tr_imgs,
            "Train_Img_Pct": tr_imgs / total_cls_imgs * 100,
            "Train_Specs": len(tr_specs),
            "Train_Spec_Pct": len(tr_specs) / total_cls_specs * 100,
            "Val_Images": va_imgs,
            "Val_Img_Pct": va_imgs / total_cls_imgs * 100,
            "Val_Specs": len(va_specs),
            "Val_Spec_Pct": len(va_specs) / total_cls_specs * 100,
            "Test_Images": te_imgs,
            "Test_Img_Pct": te_imgs / total_cls_imgs * 100,
            "Test_Specs": len(te_specs),
            "Test_Spec_Pct": len(te_specs) / total_cls_specs * 100,
        })

    df_species_splits = pd.concat(split_results, ignore_index=True)

    # 4. Display Summary Table
    print("\n" + "-" * 115)
    print(f"{'DATASET SPLIT SUMMARY (80:10:10) PER CLASS & SPECIMEN':^115}")
    print("-" * 115)
    print(f"{'Class':<18} | {'Total Img':>9} | {'Specs':>8} | {'Train Img (%)':>15} {'(Specs)':>8} | {'Val Img (%)':>13} {'(Specs)':>8} | {'Test Img (%)':>14} {'(Specs)':>8}")
    print("-" * 115)

    tot_imgs = sum(s["Total_Images"] for s in summary_stats)
    tot_specs = sum(s["Total_Specs"] for s in summary_stats)
    tot_tr_img = sum(s["Train_Images"] for s in summary_stats)
    tot_tr_sp = sum(s["Train_Specs"] for s in summary_stats)
    tot_va_img = sum(s["Val_Images"] for s in summary_stats)
    tot_va_sp = sum(s["Val_Specs"] for s in summary_stats)
    tot_te_img = sum(s["Test_Images"] for s in summary_stats)
    tot_te_sp = sum(s["Test_Specs"] for s in summary_stats)

    for s in summary_stats:
        print(f"{s['Class']:<18} | {s['Total_Images']:>9} | {s['Total_Specs']:>8} | "
              f"{s['Train_Images']:>5} ({s['Train_Img_Pct']:>5.1f}%) {s['Train_Specs']:>7} | "
              f"{s['Val_Images']:>4} ({s['Val_Img_Pct']:>5.1f}%) {s['Val_Specs']:>7} | "
              f"{s['Test_Images']:>5} ({s['Test_Img_Pct']:>5.1f}%) {s['Test_Specs']:>7}")

    print("-" * 115)
    print(f"{'TOTAL':<18} | {tot_imgs:>9} | {tot_specs:>8} | "
          f"{tot_tr_img:>5} ({tot_tr_img/tot_imgs*100:>5.1f}%) {tot_tr_sp:>7} | "
          f"{tot_va_img:>4} ({tot_va_img/tot_imgs*100:>5.1f}%) {tot_va_sp:>7} | "
          f"{tot_te_img:>5} ({tot_te_img/tot_imgs*100:>5.1f}%) {tot_te_sp:>7}")
    print("-" * 115)

    # 5. Zero Leakage Verification Check
    tr_samples = set(df_species_splits[df_species_splits["Split"] == "train"]["Sample"])
    va_samples = set(df_species_splits[df_species_splits["Split"] == "val"]["Sample"])
    te_samples = set(df_species_splits[df_species_splits["Split"] == "test"]["Sample"])

    overlap = (tr_samples & va_samples) | (tr_samples & te_samples) | (va_samples & te_samples)
    if len(overlap) == 0:
        print(f"[VERIFICATION SUCCESS] ZERO DATA LEAKAGE: 100% of specimens are disjoint across train, val, and test!")
    else:
        raise ValueError(f"[ERROR] Data leakage detected on specimens: {overlap}")

    # 6. Copy Files to Final Folders
    if not args.dry_run:
        print(f"\nCopying images to target directory: {out_dir} ...")
        for split in ["train", "val", "test"]:
            for cls in classes:
                dest_dir = os.path.join(out_dir, split, cls)
                os.makedirs(dest_dir, exist_ok=True)

        copied_count = 0
        for _, row in df_species_splits.iterrows():
            src = row["proc_path"]
            dest = os.path.join(out_dir, row["Split"], row["Class"], row["proc_filename"])
            shutil.copy2(src, dest)
            copied_count += 1

        print(f"Complete! Successfully copied {copied_count} files to '{out_dir}'.")
    else:
        print(f"\n[Dry Run] Simulation active. No files copied.")

    return df_species_splits


def main():
    args = parse_args()

    print("\n" + "=" * 80)
    print("SEAWEED DATASET SPLIT: SPECIMEN-LEVEL & STRATIFIED (80:10:10)")
    print("=" * 80)
    print(f"Target Ratio       : Train={args.train_ratio*100:.0f}%, Val={args.val_ratio*100:.0f}%, Test={args.test_ratio*100:.0f}%")
    print(f"Random Seed        : {args.seed}")
    print(f"Filter Deleted     : {not args.include_deleted}")
    print(f"Metadata CSV       : {args.csv}")

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file '{args.csv}' not found!")

    df = pd.read_csv(args.csv)
    print(f"Total rows in CSV: {len(df)}")

    # Add column Split_Fix in master dataframe
    df["Split_Fix"] = "Excluded"

    # Process Kappaphycus alvarezii
    df_alvarezii = df[df["Species"] == "Kappaphycus alvarezii"]
    split_alvarezii = process_species(
        "Kappaphycus alvarezii",
        df_alvarezii,
        args.alvarezii_proc,
        args.alvarezii_out,
        args
    )
    for _, r in split_alvarezii.iterrows():
        df.loc[r["csv_index"], "Split_Fix"] = r["Split"]

    print("\n")

    # Process Kappaphycus striatum
    df_striatum = df[df["Species"] == "Kappaphycus striatum"]
    split_striatum = process_species(
        "Kappaphycus striatum",
        df_striatum,
        args.striatum_proc,
        args.striatum_out,
        args
    )
    for _, r in split_striatum.iterrows():
        df.loc[r["csv_index"], "Split_Fix"] = r["Split"]

    # Export updated CSV
    if not args.dry_run:
        df.to_csv(args.output_csv, index=False)
        print("\n" + "=" * 80)
        print(f"Split metadata successfully saved to: '{args.output_csv}'")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80 + "\n")
    else:
        print("\n" + "=" * 80)
        print(f"[Dry Run] Simulation complete. Run without --dry-run to generate directories and copy files.")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
