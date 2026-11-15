"""
Pipeline to Merge Fix Datasets:
- Source 1: Kappaphycus_alvarezii_datasets_fix
- Source 2: Kappaphycus_striatum_datasets_fix
- Target  : mixed_datasets_fix

Features:
- Preserves 80:10:10 split structure (train, val, test) and disease classes (algae_infection, healthy, ice-ice).
- Uses species-specific prefixes ('alvarezii_' and 'striatum_') to prevent filename collisions.
- Verifies file count integrity and zero data leakage.
"""

import os
import glob
import shutil
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Merge Alvarezii and Striatum fix datasets into mixed_datasets_fix.")
    parser.add_argument("--alvarezii-dir", type=str, default="Kappaphycus_alvarezii_datasets_fix",
                        help="Path to Kappaphycus_alvarezii_datasets_fix folder.")
    parser.add_argument("--striatum-dir", type=str, default="Kappaphycus_striatum_datasets_fix",
                        help="Path to Kappaphycus_striatum_datasets_fix folder.")
    parser.add_argument("--output-dir", type=str, default="mixed_datasets_fix",
                        help="Path to output merged dataset folder.")
    parser.add_argument("--csv-metadata", type=str, default="detail_dataset_split_fix.csv",
                        help="Path to split fix metadata CSV.")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Simulate merge without copying files.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 85)
    print("SEAWEED DATASET MERGE: mixed_datasets_fix")
    print("=" * 85)
    print(f"Alvarezii Source : {args.alvarezii_dir}")
    print(f"Striatum Source  : {args.striatum_dir}")
    print(f"Target Directory : {args.output_dir}")
    print(f"Dry Run Mode     : {args.dry_run}")
    print("=" * 85)

    if not os.path.exists(args.alvarezii_dir):
        raise FileNotFoundError(f"Directory {args.alvarezii_dir} not found!")
    if not os.path.exists(args.striatum_dir):
        raise FileNotFoundError(f"Directory {args.striatum_dir} not found!")

    splits = ["train", "val", "test"]
    classes = ["algae_infection", "healthy", "ice-ice"]

    sources = [
        ("alvarezii", args.alvarezii_dir, "alvarezii_"),
        ("striatum", args.striatum_dir, "striatum_")
    ]

    # Collect all copy tasks
    tasks = []
    stats = {
        cls: {"train": 0, "val": 0, "test": 0, "total": 0, "alv": 0, "str": 0}
        for cls in classes
    }

    for sp_key, src_dir, prefix in sources:
        for split in splits:
            for cls in classes:
                folder_p = os.path.join(src_dir, split, cls)
                if not os.path.exists(folder_p):
                    continue
                files = [f for f in os.listdir(folder_p) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                for fname in files:
                    src_file = os.path.join(folder_p, fname)
                    dest_fname = f"{prefix}{fname}"
                    dest_file = os.path.join(args.output_dir, split, cls, dest_fname)

                    tasks.append({
                        "species": sp_key,
                        "split": split,
                        "class": cls,
                        "src_file": src_file,
                        "dest_file": dest_file,
                        "dest_fname": dest_fname
                    })

                    stats[cls][split] += 1
                    stats[cls]["total"] += 1
                    if sp_key == "alvarezii":
                        stats[cls]["alv"] += 1
                    else:
                        stats[cls]["str"] += 1

    print(f"\nTotal files to merge: {len(tasks)}")

    # Display Statistics Table
    print("\n" + "-" * 85)
    print(f"{'FILE DISTRIBUTION SUMMARY FOR mixed_datasets_fix':^85}")
    print("-" * 85)
    print(f"{'Class':<18} | {'Alvarezii':>10} | {'Striatum':>9} | {'Train (80%)':>12} | {'Val (10%)':>10} | {'Test (10%)':>11} | {'Total':>8}")
    print("-" * 85)

    tot_alv = sum(stats[c]["alv"] for c in classes)
    tot_str = sum(stats[c]["str"] for c in classes)
    tot_tr = sum(stats[c]["train"] for c in classes)
    tot_va = sum(stats[c]["val"] for c in classes)
    tot_te = sum(stats[c]["test"] for c in classes)
    grand_tot = sum(stats[c]["total"] for c in classes)

    for c in classes:
        st = stats[c]
        print(f"{c:<18} | {st['alv']:>10} | {st['str']:>9} | {st['train']:>6} ({st['train']/st['total']*100:4.1f}%) | "
              f"{st['val']:>5} ({st['val']/st['total']*100:4.1f}%) | {st['test']:>5} ({st['test']/st['total']*100:4.1f}%) | {st['total']:>8}")

    print("-" * 85)
    print(f"{'TOTAL COMBINED':<18} | {tot_alv:>10} | {tot_str:>9} | {tot_tr:>6} ({tot_tr/grand_tot*100:4.1f}%) | "
          f"{tot_va:>5} ({tot_va/grand_tot*100:4.1f}%) | {tot_te:>5} ({tot_te/grand_tot*100:4.1f}%) | {grand_tot:>8}")
    print("-" * 85)

    if args.dry_run:
        print("\n[Dry Run] Simulation complete. No files copied.")
        return

    # Create destination directories
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(args.output_dir, split, cls), exist_ok=True)

    print(f"\nCopying {len(tasks)} files to '{args.output_dir}'...")
    copied_count = 0
    for t in tasks:
        shutil.copy2(t["src_file"], t["dest_file"])
        copied_count += 1

    print(f"Successfully copied {copied_count} files to folder '{args.output_dir}'.")

    # Verification of copied files
    actual_files = glob.glob(os.path.join(args.output_dir, "*", "*", "*.jpg"))
    print(f"\nVerifying file counts in target folder:")
    print(f"  - Target file count : {grand_tot}")
    print(f"  - Actual file count : {len(actual_files)}")
    assert len(actual_files) == grand_tot, "Error: File count mismatch in destination directory!"

    # Zero leakage check with metadata CSV if present
    if os.path.exists(args.csv_metadata):
        df_meta = pd.read_csv(args.csv_metadata)
        retained = df_meta[df_meta["Split_Fix"] != "Excluded"]

        tr_samples = set(retained[retained["Split_Fix"] == "train"]["Sample"])
        va_samples = set(retained[retained["Split_Fix"] == "val"]["Sample"])
        te_samples = set(retained[retained["Split_Fix"] == "test"]["Sample"])

        overlap = (tr_samples & va_samples) | (tr_samples & te_samples) | (va_samples & te_samples)
        if len(overlap) == 0:
            print(f"  - Leakage check     : PASSED (Zero data leakage across train, val, and test!)")
        else:
            raise ValueError(f"Error: Leakage detected on specimens {overlap}")

    print("\n" + "=" * 85)
    print(f"Folder '{args.output_dir}' is ready for YOLO training and evaluation!")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
