

import argparse
import pandas as pd
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Convert speaker IDs in a CSV to numbers and create a new txt file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_csv", help="Path to the input CSV file (validation_set_new.csv).")
    parser.add_argument("output_txt", help="Path to the output txt file (validation_set_new.txt).")
    args = parser.parse_args()

    # Speaker ID mapping provided by the user
    speaker_id_map = {
        "FJK": 368, "FKC": 369, "FMS": 370, "FMT": 371, "FNN": 372,
        "FSA": 373, "FSH": 374, "FTH": 375, "FTS": 376, "FTY": 377,
        "FUE": 378, "FYH": 379
    }

    print(f"1. Reading from {args.input_csv}...")
    try:
        df = pd.read_csv(args.input_csv, sep="|", header=None, names=['発話ID', '話者ID', '発話内容'])
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    print(f"Read {len(df)} utterances.")

    print("2. Converting speaker IDs to numbers...")
    # Use the map to create a new column with numeric speaker IDs
    df['話者ID（数字）'] = df['話者ID'].map(speaker_id_map)

    # Handle cases where a speaker ID might not be in the map
    unmapped_speakers = df[df['話者ID（数字）'].isnull()]['話者ID'].unique()
    if len(unmapped_speakers) > 0:
        print(f"Warning: The following speaker IDs were found in the data but not in the mapping and will be excluded: {unmapped_speakers}", file=sys.stderr)
        df.dropna(subset=['話者ID（数字）'], inplace=True)
    
    # Convert numeric ID to integer
    df['話者ID（数字）'] = df['話者ID（数字）'].astype(int)

    print("3. Formatting and writing to output file...")
    output_df = df[['話者ID（数字）', '発話内容']]

    with open(args.output_txt, "w", encoding="utf-8") as f:
        for _, row in output_df.iterrows():
            f.write(f"{row['話者ID（数字）']}|{row['発話内容']}\n")

    print("\nDone.")
    print(f"Successfully created {args.output_txt} with {len(output_df)} utterances.")

if __name__ == "__main__":
    main()

