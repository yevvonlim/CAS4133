from datasets import load_from_disk, concatenate_datasets, DatasetDict
import argparse
import os
from typing import List

def concat_hf_datasets(input_paths: List[str]) -> DatasetDict:
    """
    Load HF datasets from each path in input_paths and concatenate them.
    Returns either a single Dataset (if inputs are Datasets) or a DatasetDict.
    """
    # Load each dataset
    loaded = [load_from_disk(p) for p in input_paths]

    # If they are DatasetDicts, we concatenate split-wise
    if isinstance(loaded[0], DatasetDict):
        # Ensure all have the same splits
        splits = list(loaded[0].keys())
        for ds in loaded:
            assert list(ds.keys()) == splits, "All DatasetDicts must have identical splits"

        concatenated = {}
        for split in splits:
            concatenated[split] = concatenate_datasets([ds[split] for ds in loaded])
        return DatasetDict(concatenated)
    else:
        # They’re plain Dataset objects
        return concatenate_datasets(loaded)

def main():
    parser = argparse.ArgumentParser(
        description="Concatenate multiple HF datasets on disk and save the result."
    )
    parser.add_argument(
        "--inputs", "-i",
        nargs="+",
        required=True,
        help="Paths to input datasets (directories saved via `save_to_disk`)."
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Directory where the concatenated dataset will be saved."
    )
    args = parser.parse_args()

    # Check and prepare output directory
    if os.path.exists(args.output):
        raise FileExistsError(f"Output path {args.output} already exists. Please remove it or choose another.")
    os.makedirs(args.output, exist_ok=False)

    # Concatenate
    merged = concat_hf_datasets(args.inputs)

    # Save to disk
    merged.save_to_disk(args.output)
    print(f"✅ Successfully concatenated {len(args.inputs)} dataset(s) and saved to {args.output}")

if __name__ == "__main__":
    main()