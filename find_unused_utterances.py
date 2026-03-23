

import argparse
import sys
from os.path import join, dirname, abspath
import pandas as pd

# Add the local script directory to the python path to allow importing makemetadata
# The script is expected to be run from the root of the MB-iSTFT-VITS project
sys.path.append(abspath("uudb/tts1/local"))
from makemetadata import uudb_df

def get_test_set_ids():
    """
    Parses the makesets.py file to extract the list of test utterance IDs.
    This is a bit fragile, but it avoids having to exec the file.
    """
    testset = [
        "C002_L_107", "C002_L_175", "C004_L_126", "C004_R_044",
        "C005_R_152", "C006_L_050", "C007_L_137", "C031_L_002",
        "C033_R_134", "C041_L_134", "C041_L_259", "C042_R_090",
        "C043_R_064", "C051_L_072", "C051_R_118", "C051_R_170",
    ]
    return set(testset)

def get_training_set_ids(dataroot):
    """
    Reads the training set utterance IDs from the data/train/text file.
    """
    train_ids = set()
    # The utterance ID is the first column in the text file
    train_text_file = join(dataroot, "train", "text")
    try:
        with open(train_text_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    # The format is FTS_C001_001, but the test set is C001_L_001.
                    # Let's check the format in makesets.py
                    # wavbn = f"{u.Speaker}_{u.SessionID}_{u.UtteranceID}" -> F01_C001_001
                    # testset = "C002_L_107"
                    # df_["train"] = df[df.apply(lambda u: not f"{u.SessionID}_{u.Channel}_{u.UtteranceID}" in testset, axis=1)]
                    # The training IDs are in the format {Speaker}_{SessionID}_{UtteranceID}
                    # The test IDs are in the format {SessionID}_{Channel}_{UtteranceID}
                    # I need to get the mapping from the main df.
                    train_ids.add(parts[0])
    except FileNotFoundError:
        print(f"Error: Training set file not found at {train_text_file}", file=sys.stderr)
        print("Please make sure you have already generated the data sets.", file=sys.stderr)
        sys.exit(1)
    return train_ids

def main():
    parser = argparse.ArgumentParser(
        description="Find utterances in UUDB not used in train or test sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("uudbroot", help="Path to the root of the UUDB corpus.")
    parser.add_argument("--dataroot", default="uudb/tts1/data", help="Path to the directory containing train/test data.")
    parser.add_argument("--output_file", help="Optional: File to write the list of unused utterance IDs.")
    args = parser.parse_args()

    print("1. Loading all utterance metadata from UUDB...")
    all_utterances_df = uudb_df(args.uudbroot)
    # The unique ID used for splitting is "{SessionID}_{Channel}_{UtteranceID}"
    all_utterances_df["split_id"] = all_utterances_df.apply(
        lambda u: f"{u.SessionID}_{u.Channel}_{u.UtteranceID}", axis=1
    )
    all_ids = set(all_utterances_df["split_id"])
    print(f"Found {len(all_ids)} total utterances.")

    print("2. Loading defined test set IDs...")
    test_ids = get_test_set_ids()
    print(f"Found {len(test_ids)} test set utterances.")

    print("3. Loading generated training set IDs...")
    # The IDs in train/text are different, need to map them back to the 'split_id'
    # The train ID is f"{u.Speaker}_{u.SessionID}_{u.UtteranceID}"
    all_utterances_df["train_id"] = all_utterances_df.apply(
        lambda u: f"{u.Speaker}_{u.SessionID}_{u.UtteranceID}", axis=1
    )
    train_id_map = pd.Series(all_utterances_df.split_id.values, index=all_utterances_df.train_id).to_dict()

    train_ids_from_file = get_training_set_ids(args.dataroot)
    train_ids = {train_id_map.get(tid) for tid in train_ids_from_file}
    # Remove None if any ID was not found in the map
    train_ids.discard(None)
    print(f"Found {len(train_ids)} training set utterances.")

    print("4. Finding unused utterances...")
    used_ids = train_ids.union(test_ids)
    unused_ids = all_ids.difference(used_ids)

    if not unused_ids:
        print("\nNo unused utterances found.")
        return

    print(f"\nFound {len(unused_ids)} unused utterances:")
    sorted_unused_ids = sorted(list(unused_ids))

    if args.output_file:
        with open(args.output_file, "w") as f:
            for uid in sorted_unused_ids:
                f.write(f"{uid}\n")
        print(f"List of unused utterances saved to {args.output_file}")
    else:
        # Print to console
        for uid in sorted_unused_ids:
            print(uid)

if __name__ == "__main__":
    main()
