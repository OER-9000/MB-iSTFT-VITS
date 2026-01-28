import os
import torch
from scipy.io.wavfile import write as write_wav
from synthesis_module import SynthesisModule, get_synthesis_module_instance
import time

def main():
    # --- 設定 ---
    # TODO: ご自身の環境に合わせて、学習済みモデルのパスを修正してください
    # 例: config_path = "logs/uudb_csj31/config.json"
    #     checkpoint_path = "logs/uudb_csj31/G_350000.pth"
    config_path = "logs/uudb_csj31/config.json"
    checkpoint_path = "logs/uudb_csj31/G_latest.pth"
    
    output_wav_path = "output_chunkwise_synthesis_sample.wav"

    # モデルを読み込むデバイス
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 合成するテキストと話者ID
    raw_text = "これは、文節ごとに分割して音声を合成するためのサンプルコードです。長い文章でも、高品質な音声が期待できます。"
    speaker_id = 368 # TODO: 使用したい話者IDに変更してください (例: CSJ話者なら0-367, UUDB追加話者なら368-)

    # --- 処理の実行 ---
    print(f"Loading model from {checkpoint_path}...")
    # get_synthesis_module_instance を使うと、2回目以降の呼び出しでモデルの再読み込みをスキップできます
    synthesis_module = get_synthesis_module_instance(config_path, checkpoint_path, device=device)

    # 1. 潜在表現と文節情報を一括で準備
    print(f"Preparing shared latents for text: '{raw_text[:30]}'...")
    start_time = time.time()
    z, w, chunk_counts, bunsetsu_phonemes = synthesis_module.prepare_shared_latents(
        raw_text,
        speaker_id,
        noise_scale=0.667,
        noise_scale_w=0.8,
        length_scale=1.0
    )
    prep_time = time.time() - start_time
    print(f"Latent preparation took {prep_time:.2f} seconds.")

    if z is None:
        print("Failed to prepare latents. The input text might be empty or invalid.")
        return

    print(f"Text was split into {len(chunk_counts)} bunsetsu (phrases).")
    # print("Bunsetsu (phonemes):", bunsetsu_phonemes) # 詳細表示が必要な場合はコメントを外してください

    # 2. 準備した潜在表現から、文節ごとにデコードして音声を合成
    print("Synthesizing audio from shared latents (chunk by chunk)...")
    start_time = time.time()
    # ここで新しい関数を呼び出す
    audio = synthesis_module.synthesize_spectrogram_concat_validation(
        z,
        w,
        chunk_counts,
        speaker_id
    )
    synth_time = time.time() - start_time
    print(f"Chunk-wise synthesis took {synth_time:.2f} seconds.")

    # 3. 音声ファイルを保存
    sampling_rate = synthesis_module.sampling_rate
    write_wav(output_wav_path, sampling_rate, audio)
    
    audio_duration = len(audio) / sampling_rate
    print(f"\nSuccessfully synthesized {audio_duration:.2f} seconds of audio.")
    print(f"Saved to: {output_wav_path}")


if __name__ == "__main__":
    main()
