
import numpy as np
import argparse
import re
from text_JP.phonemize import Phonemizer
import pyopenjtalk
import sys

# synthesis_module.pyから必要な関数を移植・簡略化
def get_phoneme_string(text, phonemizer):
    """テキストを音素文字列に変換する"""
    parts = re.split(r'({cough}|<cough>|[\[\]]|[、。])', text)
    phoneme_parts = []
    for part in parts:
        if not part or part.isspace():
            continue
        if part.startswith('[') and part.endswith(']') and len(part) > 2:
            content = part[1:-1]
            if not content:
                phoneme_parts.append('[ ]')
            else:
                try:
                    kana_content = pyopenjtalk.g2p(content, kana=True).replace('ヲ', 'オ')
                    phoneme_content = phonemizer(kana_content)
                    phoneme_parts.append(f'[ {phoneme_content} ]')
                except Exception:
                    # g2pが失敗するような特殊なケースに対応
                    phoneme_parts.append(f'[ {content} ]')
            continue
        if part == '{cough}' or part == '<cough>':
            phoneme_parts.append('<cough>')
            continue
        if part in '、。':
            phoneme_parts.append('sp')
            continue
        
        kana = pyopenjtalk.g2p(part, kana=True).replace('ヲ', 'オ')
        phonemes = phonemizer(kana)
        phoneme_parts.append(phonemes)
        
    final_text = ' '.join(phoneme_parts)
    return re.sub(r'\s+', ' ', final_text).strip()

def main():
    parser = argparse.ArgumentParser(description="テキストファイルの音素数を分析するスクリプト")
    parser.add_argument("--validation_path", type=str, default="validation_set_new_B.txt", help="分析対象のテキストファイルのパス")
    args = parser.parse_args()

    phoneme_counts = []
    
    try:
        print("Initializing Phonemizer...")
        phonemizer = Phonemizer()
        print("Phonemizer initialized.")
    except Exception as e:
        print(f"Error initializing Phonemizer: {e}", file=sys.stderr)
        print("\nMeCabまたは辞書が正しく設定されていない可能性があります。", file=sys.stderr)
        print("以前の提案通り `pip install --force-reinstall unidic` を試してください。", file=sys.stderr)
        sys.exit(1)

    print(f"Reading validation set from: {args.validation_path}")
    with open(args.validation_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Analyzing {len(lines)} text samples...")
    for i, line in enumerate(lines):
        try:
            _, text = line.split("|", 1)
        except ValueError:
            print(f"Skipping malformed line: {line}", file=sys.stderr)
            continue

        try:
            phoneme_string = get_phoneme_string(text, phonemizer)
            # 空白で分割し、空の要素を除外してカウント
            phonemes = [p for p in phoneme_string.split(' ') if p]
            phoneme_counts.append(len(phonemes))
        except Exception as e:
            print(f"Could not process line: {line}", file=sys.stderr)
            print(f"Error: {e}", file=sys.stderr)
            continue
    
    if not phoneme_counts:
        print("No texts were processed successfully.")
        return

    counts_array = np.array(phoneme_counts)
    mean = np.mean(counts_array)
    std_dev = np.std(counts_array)
    min_val = np.min(counts_array)
    max_val = np.max(counts_array)

    print("\n--- Phoneme Count Analysis ---")
    print(f"File: {args.validation_path}")
    print(f"Number of texts analyzed: {len(counts_array)}")
    print(f"Average number of phonemes: {mean:.2f}")
    print(f"Standard deviation of phonemes: {std_dev:.2f}")
    print(f"Minimum phonemes in a text: {min_val}")
    print(f"Maximum phonemes in a text: {max_val}")
    print("------------------------------")

if __name__ == "__main__":
    main()
