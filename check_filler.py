import argparse
import sys
from os.path import join, abspath
from xml.dom import minidom
import pandas as pd

def ifattribute(element, name):
    anode = element.getAttributeNode(name)
    if anode is not None:
        if anode.nodeValue == "true":
            return True
    return False

def check_filler_in_utterance(uudbroot, session_id, utterance_id):
    """
    Checks if any chunk in a specific utterance has the Filler="true" tag.
    """
    xml_path = join(uudbroot, "Sessions", session_id, f"{session_id}.xml")
    try:
        dom = minidom.parse(xml_path)
        for utterance in dom.getElementsByTagName("Utterance"):
            if utterance.getAttribute("UtteranceID") == utterance_id:
                for child in utterance.childNodes:
                    if child.nodeName == "Chunk":
                        if ifattribute(child, "Filler"):
                            return True
    except FileNotFoundError:
        return False
    return False

def main():
    parser = argparse.ArgumentParser(
        description='Check for "Filler" tags in the combined validation set.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("uudbroot", help="Path to the root of the UUDB corpus.")
    args = parser.parse_args()

    print("1. Reading utterance lists from prerequisite CSV files...")
    try:
        long_df = pd.read_csv("additional_validation_set_long.csv", sep="|")
        existing_df = pd.read_csv("existing_validation_set.csv", sep="|")
        combined_df = pd.concat([long_df, existing_df], ignore_index=True)
        utterance_ids = combined_df['発話ID'].tolist()

    except FileNotFoundError:
        print(f"Error: Prerequisite CSV file not found. Please ensure 'additional_validation_set_long.csv' and 'existing_validation_set.csv' exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(utterance_ids)} utterances to check.")

    filler_count = 0
    filler_utterances = []

    print("2. Checking each utterance for filler tags in XML files...")
    for utt_id_str in utterance_ids:
        parts = utt_id_str.split('_')
        session_id = parts[0]
        utterance_id = parts[2]

        if check_filler_in_utterance(args.uudbroot, session_id, utterance_id):
            filler_count += 1
            filler_utterances.append(utt_id_str)

    print("\n--- Results ---")
    print(f"Total utterances checked: {len(utterance_ids)}")
    print(f"Number of utterances with Filler tag: {filler_count}")
    
    if filler_count > 0:
        print("\nUtterances with Filler tag:")
        for utt in filler_utterances:
            print(utt)


if __name__ == "__main__":
    main()
