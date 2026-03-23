
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Filter a CSV file to include only utterances with 2 or more bunsetsu (clauses).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument("output_csv", help="Path to the output CSV file.")
    args = parser.parse_args()

    print(f"1. Reading from {args.input_csv}...")
    try:
        df = pd.read_csv(args.input_csv, sep="|")
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    print("2. Filtering for utterances with 2 or more clauses (bunsetsu)...")
    # We use the presence of at least one '、' as a proxy for having at least 2 clauses.
    filtered_df = df[df["発話内容"].str.contains("、", na=False)].copy()

    print(f"3. Writing output to {args.output_csv}...")
    filtered_df.to_csv(args.output_csv, sep="|", index=False)

    print("\nDone.")
    print(f"Originally {len(df)} utterances, filtered down to {len(filtered_df)} utterances.")
    print(f"Successfully created {args.output_csv}.")

if __name__ == "__main__":
    main()

