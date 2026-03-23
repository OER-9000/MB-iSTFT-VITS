

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
        description="Create a CSV of the existing validation set with orthographic transcriptions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("uudbroot", help="Path to the root of the UUDB corpus.")
    parser.add_argument("output_csv", help="Path to the output CSV file.")
    args = parser.parse_args()

    # 1. Load all utterance metadata to get speaker info
    print("1. Loading all utterance metadata from UUDB...")
    all_utterances_df = uudb_df(args.uudbroot)
    all_utterances_df["split_id"] = all_utterances_df.apply(
        lambda u: f"{u.SessionID}_{u.Channel}_{u.UtteranceID}", axis=1
    )

    # 2. Get the hardcoded test set IDs
    print("2. Getting the list of existing validation utterances...")
    test_ids = get_test_set_ids()
    
    test_df = all_utterances_df[all_utterances_df["split_id"].isin(test_ids)].copy()
    print(f"Found {len(test_df)} utterances in the existing validation set.")

    # 3. Get orthographic transcriptions for the test set utterances
    print("3. Extracting orthographic transcriptions from XML files...")
    
    new_rows = []
    for row in test_df.itertuples():
        transcription = get_orthographic_transcription(args.uudbroot, row.SessionID, row.UtteranceID)
        # Remove {laugh} as requested
        transcription = transcription.replace("{laugh}", "")
        new_rows.append({
            "発話ID": row.split_id,
            "話者ID": row.Speaker,
            "発話内容": transcription
        })
    
    output_df = pd.DataFrame(new_rows)

    # 4. Write to CSV
    print(f"4. Writing output to {args.output_csv}...")
    output_df.to_csv(args.output_csv, sep="|", index=False)

    print("\nDone.")
    print(f"Successfully created {args.output_csv} with {len(output_df)} utterances.")

if __name__ == "__main__":
    main()

