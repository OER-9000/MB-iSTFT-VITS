
import argparse
import sys
from os.path import join, dirname, abspath
import pandas as pd
from xml.dom import minidom

# Add the local script directory to the python path to allow importing makemetadata
sys.path.append(abspath("uudb/tts1/local"))
from makemetadata import uudb_df

def get_test_set_ids():
    """
    Returns the hardcoded list of test utterance IDs.
    """
    testset = [
        "C002_L_107", "C002_L_175", "C004_L_126", "C004_R_044",
        "C005_R_152", "C006_L_050", "C007_L_137", "C031_L_002",
        "C033_R_134", "C041_L_134", "C041_L_259", "C042_R_090",
        "C043_R_064", "C051_L_072", "C051_R_118", "C051_R_170",
    ]
    return set(testset)

def get_training_set_ids(dataroot="uudb/tts1/data"):
    """
    Reads the training set utterance IDs from the specified data root.
    """
    train_ids = set()
    train_text_file = join(dataroot, "train", "text")
    try:
        with open(train_text_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    train_ids.add(parts[0])
    except FileNotFoundError:
        print(f"Error: Training set file not found at {train_text_file}", file=sys.stderr)
        sys.exit(1)
    return train_ids

def ifattribute(element, name):
    anode = element.getAttributeNode(name)
    if anode is not None:
        if anode.nodeValue == "true":
            return True
    return False

def get_orthographic_transcription(uudbroot, session_id, utterance_id):
    """
    Parses the XML to get the orthographic transcription for a specific utterance,
    preserving brackets for fillers and discourse markers.
    """
    xml_path = join(uudbroot, "Sessions", session_id, f"{session_id}.xml")
    try:
        dom = minidom.parse(xml_path)
        transcriptions = []
        for utterance in dom.getElementsByTagName("Utterance"):
            if utterance.getAttribute("UtteranceID") == utterance_id:
                for child in utterance.childNodes:
                    if child.nodeName == "Chunk":
                        trans_node = child.getAttributeNode("OrthographicTranscription")
                        if trans_node:
                            trans = trans_node.nodeValue
                            # Replicate the logic from makemetadata.py for phonetic transcription
                            # to add brackets around fillers, etc.
                            if ifattribute(child, "ExpressiveInterjection") or ifattribute(child, "Filler") or ifattribute(child, "DiscourseMarker"):
                                transcriptions.append(f"[{trans}]")
                            else:
                                transcriptions.append(trans)
                    elif child.nodeName == "NonLinguisticSound":
                        if child.getAttribute("TagLaugh") == "true":
                            transcriptions.append("{laugh}")
                        elif child.getAttribute("TagBreath") == "true":
                            transcriptions.append("{breath}")
                        elif child.getAttribute("TagSigh") == "true":
                            transcriptions.append("{sigh}")
                        elif child.getAttribute("TagCough") == "true":
                            transcriptions.append("{cough}")
                    elif child.nodeName == "ShortPause":
                        transcriptions.append("、")
                return "".join(transcriptions)
    except FileNotFoundError:
        print(f"Warning: XML file not found at {xml_path}", file=sys.stderr)
        return ""
    return ""

def main():
    parser = argparse.ArgumentParser(
        description="Create a CSV of additional validation utterances from UUDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("uudbroot", help="Path to the root of the UUDB corpus.")
    parser.add_argument("output_csv", help="Path to the output CSV file.")
    parser.add_argument("--dataroot", default="uudb/tts1/data", help="Path to the directory containing train/test data.")
    parser.add_argument("--min_morae", type=int, default=3, help="Minimum number of morae for an utterance to be included.")
    args = parser.parse_args()

    # 1. Load all utterance metadata to get IDs, speakers, and mora count
    print("1. Loading all utterance metadata from UUDB...")
    all_utterances_df = uudb_df(args.uudbroot)
    all_utterances_df["split_id"] = all_utterances_df.apply(
        lambda u: f"{u.SessionID}_{u.Channel}_{u.UtteranceID}", axis=1
    )
    all_utterances_df["train_id"] = all_utterances_df.apply(
        lambda u: f"{u.Speaker}_{u.SessionID}_{u.UtteranceID}", axis=1
    )

    # 2. Identify used utterances
    print("2. Identifying used utterances...")
    test_ids = get_test_set_ids()
    
    train_id_map = pd.Series(all_utterances_df.split_id.values, index=all_utterances_df.train_id).to_dict()
    train_ids_from_file = get_training_set_ids(args.dataroot)
    train_ids = {train_id_map.get(tid) for tid in train_ids_from_file if tid in train_id_map}

    used_ids = train_ids.union(test_ids)

    # 3. Filter for unused utterances
    print("3. Filtering for unused utterances...")
    unused_df = all_utterances_df[~all_utterances_df["split_id"].isin(used_ids)].copy()
    print(f"Found {len(unused_df)} unused utterances.")

    # 4. Filter by length (morae)
    print(f"4. Filtering out utterances shorter than {args.min_morae} morae...")
    original_count = len(unused_df)
    unused_df = unused_df[unused_df["numMorae"] >= args.min_morae]
    print(f"Removed {original_count - len(unused_df)} short utterances. {len(unused_df)} remain.")

    # 5. Get orthographic transcriptions for the remaining unused utterances
    print("5. Extracting orthographic transcriptions from XML files...")
    
    new_rows = []
    for row in unused_df.itertuples():
        transcription = get_orthographic_transcription(args.uudbroot, row.SessionID, row.UtteranceID)
        # Remove {laugh} as requested by the user
        transcription = transcription.replace("{laugh}", "")
        new_rows.append({
            "発話ID": row.split_id,
            "話者ID": row.Speaker,
            "発話内容": transcription
        })
    
    output_df = pd.DataFrame(new_rows)

    # 6. Write to CSV
    print(f"6. Writing output to {args.output_csv}...")
    output_df.to_csv(args.output_csv, sep="|", index=False)

    print("\nDone.")
    print(f"Successfully created {args.output_csv} with {len(output_df)} utterances.")

if __name__ == "__main__":
    main()
