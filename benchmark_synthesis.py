

import os
import time
import csv
import argparse
import platform
import torch
import numpy as np
from datetime import datetime

import utils
from synthesis_module import SynthesisModule

def get_system_info(synth_module):
    """実行環境の情報を取得する"""
    info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "os": platform.platform(),
        "device": str(synth_module.device)
    }
    if torch.cuda.is_available() and 'cuda' in str(synth_module.device):
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    else:
        info["gpu"] = "N/A"
        info["cuda_version"] = "N/A"
    return info

def synthesize_cond1_first_bunsetsu(synth_module, text, sid):
    """Cond 1を最初の文節にのみ適用する"""
    bunsetsu_chunks = synth_module._get_bunsetsu_chunks_mecab(text)
    if not bunsetsu_chunks:
        return np.array([])
    
    first_chunk = bunsetsu_chunks[0]
    
    stn_tst = synth_module._get_text_from_phonemes(first_chunk)
    x_tst = stn_tst.to(synth_module.device).unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(synth_module.device)
    sid_tensor = torch.LongTensor([sid]).to(synth_module.device)
    
    with torch.no_grad():
        audio = synth_module.model.infer(
            x_tst, x_tst_lengths, sid=sid_tensor,
            noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0
        )[0][0, 0].data.cpu().float().numpy()
    
    return audio

def synthesize_cond2_first_bunsetsu(synth, z, w_ceil, g, bunsetsu_phonemes):
    """Cond 2を最初の文節に適用する"""
    return synth.synthesize_cond4_first_bunsetsu(z, w_ceil, g, bunsetsu_phonemes)

def main():
    parser = argparse.ArgumentParser(description="VITS合成速度ベンチマークスクリプト")
    parser.add_argument("--config_path", type=str, required=True, help="モデルのconfig.jsonへのパス")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="モデルの.pthファイルへのパス")
    parser.add_argument("--validation_path", type=str, default="validation_set_new_B.txt", help="評価用テキストファイルのパス")
    parser.add_argument("--output_path", type=str, default="benchmark_results.csv", help="結果を出力するCSVファイルのパス")
    parser.add_argument("--device", type=str, default=None, help="実行デバイス ('cuda' or 'cpu')。デフォルトは自動検出。")
    args = parser.parse_args()

    device = args.device
    
    print(f"Attempting to use device: {device or 'auto'}")

    print("Initializing SynthesisModule...")
    synthesis_module = SynthesisModule(args.config_path, args.checkpoint_path, device=device)
    sampling_rate = synthesis_module.sampling_rate
    
    system_info = get_system_info(synthesis_module)
    print("\n--- System Information ---")
    for key, value in system_info.items():
        print(f"{key}: {value}")
    print("--------------------------\n")

    results = []

    print(f"Reading validation set from: {args.validation_path}")
    with open(args.validation_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for i, line in enumerate(lines):
        try:
            sid_str, text = line.split("|", 1)
            sid = int(sid_str)
        except ValueError:
            print(f"Skipping malformed line: {line}")
            continue

        print(f"Processing ({i+1}/{len(lines)}): SID={sid}, Text='{text[:30]}...'")

        # --- Cond 1 (First Bunsetsu) ---
        try:
            start_time = time.time()
            audio_cond1 = synthesize_cond1_first_bunsetsu(synthesis_module, text, sid)
            end_time = time.time()
            exec_time = end_time - start_time
            audio_duration = len(audio_cond1) / sampling_rate if len(audio_cond1) > 0 else 0
            rtf = exec_time / audio_duration if audio_duration > 0 else 0
            results.append({
                "speaker_id": sid, "text": text, "method": "cond1_first_bunsetsu",
                "execution_time_s": exec_time, "audio_duration_s": audio_duration, "rtf": rtf
            })
            print(f"  - Cond 1 (first): {exec_time:.4f}s, RTF: {rtf:.4f}")
        except Exception as e:
            print(f"  - Error in Cond 1: {e}")
            continue

        # --- Prepare shared latents (once per text) ---
        try:
            start_prepare_time = time.time()
            z, w_ceil, g, bunsetsu_phonemes = synthesis_module.prepare_shared_latents(text, sid)
            end_prepare_time = time.time()
            prepare_time = end_prepare_time - start_prepare_time
            
            if z is None or not bunsetsu_phonemes:
                print("  - Skipping Cond 2,3,4 due to empty phonemes from prepare_shared_latents.")
                continue
        except Exception as e:
            print(f"  - Error in prepare_shared_latents: {e}")
            continue
        
        print(f"  - Prepare Latents: {prepare_time:.4f}s")

        # --- Cond 2 (Decode First Bunsetsu) ---
        try:
            start_decode_time = time.time()
            audio_cond2 = synthesize_cond2_first_bunsetsu(synthesis_module, z, w_ceil, g, bunsetsu_phonemes)
            end_decode_time = time.time()
            decode_time = end_decode_time - start_decode_time
            
            exec_time = prepare_time + decode_time
            audio_duration = len(audio_cond2) / sampling_rate if len(audio_cond2) > 0 else 0
            rtf = exec_time / audio_duration if audio_duration > 0 else 0
            results.append({
                "speaker_id": sid, "text": text, "method": "cond2_first_bunsetsu_with_prepare",
                "execution_time_s": exec_time, "audio_duration_s": audio_duration, "rtf": rtf
            })
            print(f"  - Cond 2 (decode): {decode_time:.4f}s -> Total: {exec_time:.4f}s, RTF: {rtf:.4f}")
        except Exception as e:
            print(f"  - Error in Cond 2: {e}")

        # --- Cond 3 (Decode First Bunsetsu) ---
        try:
            start_decode_time = time.time()
            audio_cond3, _ = synthesis_module.synthesize_cond3_first_bunsetsu(z, w_ceil, g, bunsetsu_phonemes, return_debug_data=True)
            end_decode_time = time.time()
            decode_time = end_decode_time - start_decode_time

            exec_time = prepare_time + decode_time
            audio_duration = len(audio_cond3) / sampling_rate if len(audio_cond3) > 0 else 0
            rtf = exec_time / audio_duration if audio_duration > 0 else 0
            results.append({
                "speaker_id": sid, "text": text, "method": "cond3_first_bunsetsu_with_prepare",
                "execution_time_s": exec_time, "audio_duration_s": audio_duration, "rtf": rtf
            })
            print(f"  - Cond 3 (decode): {decode_time:.4f}s -> Total: {exec_time:.4f}s, RTF: {rtf:.4f}")
        except Exception as e:
            print(f"  - Error in Cond 3: {e}")

        # --- Cond 4 (Decode First Bunsetsu) ---
        try:
            start_decode_time = time.time()
            audio_cond4 = synthesis_module.synthesize_cond4_shared(z, g)
            end_decode_time = time.time()
            decode_time = end_decode_time - start_decode_time

            exec_time = prepare_time + decode_time
            audio_duration = len(audio_cond4) / sampling_rate if len(audio_cond4) > 0 else 0
            rtf = exec_time / audio_duration if audio_duration > 0 else 0
            results.append({
                "speaker_id": sid, "text": text, "method": "cond4_first_bunsetsu_with_prepare",
                "execution_time_s": exec_time, "audio_duration_s": audio_duration, "rtf": rtf
            })
            print(f"  - Cond 4 (decode): {decode_time:.4f}s -> Total: {exec_time:.4f}s, RTF: {rtf:.4f}")
        except Exception as e:
            print(f"  - Error in Cond 4: {e}")

    # --- 結果をCSVに書き出す ---
    print(f"\nWriting results to {args.output_path}...")
    with open(args.output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        writer.writerow(["speaker_id", "text", "method", "execution_time_s", "audio_duration_s", "rtf"])
        
        for res in results:
            writer.writerow([
                res["speaker_id"], res["text"], res["method"], 
                f"{res['execution_time_s']:.6f}", f"{res['audio_duration_s']:.6f}", f"{res['rtf']:.6f}"
            ])
            
        writer.writerow([])
        writer.writerow(["--- System Information ---"])
        for key, value in system_info.items():
            writer.writerow([key, value])
        writer.writerow(["benchmark_timestamp_utc", datetime.utcnow().isoformat()])

    print("Benchmark finished.")

if __name__ == "__main__":
    main()
