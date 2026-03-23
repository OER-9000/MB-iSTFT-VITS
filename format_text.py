

import argparse
import sys
from os.path import join, abspath
from xml.dom import minidom
import pandas as pd
import re

def ifattribute(element, name):
    anode = element.getAttributeNode(name)
    if anode is not None:
        if anode.nodeValue == "true":
            return True
    return False

def get_formatted_transcription(uudbroot, session_id, utterance_id):
    """
    Parses the XML to get a formatted orthographic transcription.
    - Removes content of Chunks marked as Disfluency="true".
    - Removes {laugh} and {breath}.
    - Keeps {cough} and {sigh}.
    - Wraps Fillers, DiscourseMarkers, and ExpressiveInterjections in [].
    """
    xml_path = join(uudbroot, "Sessions", session_id, f"{session_id}.xml")
    try:
        dom = minidom.parse(xml_path)
        transcriptions = []
        for utterance in dom.getElementsByTagName("Utterance"):
            if utterance.getAttribute("UtteranceID") == utterance_id:
                for child in utterance.childNodes:
                    if child.nodeName == "Chunk":
                        # If the chunk is a disfluency, ignore its content completely.
                        if ifattribute(child, "Disfluency"):
                            continue
                        
                        trans_node = child.getAttributeNode("OrthographicTranscription")
                        if trans_node:
                            trans = trans_node.nodeValue
                            # Wrap in brackets based on other attributes
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
                
                full_transcription = "".join(transcriptions)
                # Apply final formatting rules (remove laugh/breath)
                final_transcription = full_transcription.replace("{laugh}", "").replace("{breath}", "")
                
                return final_transcription

    except FileNotFoundError:
        print(f"Warning: XML file not found at {xml_path}", file=sys.stderr)
        return ""
    return ""

def main():
    parser = argparse.ArgumentParser(
        description="Format transcription text based on specific rules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("uudbroot", help="Path to the root of the UUDB corpus.")
    parser.add_argument("output_txt", help="Path to the output formatted txt file.")
    args = parser.parse_args()

    print("1. Reading utterance lists from prerequisite CSV files...")
    try:
        long_df = pd.read_csv("additional_validation_set_long.csv", sep="|")
        existing_df = pd.read_csv("existing_validation_set.csv", sep="|")
        combined_df = pd.concat([long_df, existing_df], ignore_index=True)
        combined_df.drop_duplicates(subset=['発話ID'], inplace=True)
    except FileNotFoundError:
        print(f"Error: Prerequisite CSV file not found. Please ensure 'additional_validation_set_long.csv' and 'existing_validation_set.csv' exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(combined_df)} unique utterances to format.")

    # Speaker ID mapping
    speaker_id_map = {
        "FJK": 368, "FKC": 369, "FMS": 370, "FMT": 371, "FNN": 372,
        "FSA": 373, "FSH": 374, "FTH": 375, "FTS": 376, "FTY": 377,
        "FUE": 378, "FYH": 379
    }

    new_lines = []
    print("2. Formatting transcription for each utterance...")
    for row in combined_df.itertuples():
        utt_id_str = row.発話ID
        speaker_id_str = row.話者ID
        
        parts = utt_id_str.split('_')
        session_id = parts[0]
        utterance_id = parts[2]

        formatted_transcription = get_formatted_transcription(args.uudbroot, session_id, utterance_id)
        
        numeric_speaker_id = speaker_id_map.get(speaker_id_str)
        if numeric_speaker_id is not None:
            new_lines.append(f"{numeric_speaker_id}|{formatted_transcription}")

    print(f"3. Writing output to {args.output_txt}...")
    with open(args.output_txt, "w", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line + "\n")

    print("\nDone.")
    print(f"Successfully created {args.output_txt} with {len(new_lines)} utterances.")

if __name__ == "__main__":
    main()
