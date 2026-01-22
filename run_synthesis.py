import time
import os
import numpy as np
from scipy.io.wavfile import write as write_wav
from synthesis_module import get_synthesis_module_instance

if __name__ == '__main__':
    # --- Configuration (EDIT THESE PATHS) ---
    CONFIG_PATH = "./logs/uudb_csj31/config.json"
    # Find the latest G_****.pth file in your model directory
    CHECKPOINT_PATH = "./logs/uudb_csj31/G_3010000.pth"
    
    TEXT_TO_SYNTHESIZE = "あらゆる現実を、全て自分の方へ捻じ曲げたのだ。"
    OUTPUT_WAV_PATH = "synthesis_output_sid{sid}.wav"
    
    # --- Main Execution ---
    try:
        print("Requesting synthesis module instance...")
        synthesizer = get_synthesis_module_instance(config_path=CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH)
        
        num_speakers = synthesizer.get_speaker_count()
        print(f"Model supports {num_speakers} speakers.")
        
        # Synthesize for a specific speaker (e.g., speaker 0)
        target_speaker_id = 0
        if num_speakers > target_speaker_id:
            print(f"\nSynthesizing for Speaker {target_speaker_id} --- First call")
            start_time = time.time()
            
            audio_data = synthesizer.synthesize(TEXT_TO_SYNTHESIZE, target_speaker_id)
            
            end_time = time.time()
            
            # Save the audio to a file
            output_path = OUTPUT_WAV_PATH.format(sid=target_speaker_id)
            write_wav(output_path, synthesizer.sampling_rate, audio_data)
            
            # --- Performance Metrics ---
            elapsed_time = end_time - start_time
            audio_duration = len(audio_data) / synthesizer.sampling_rate
            rtf = elapsed_time / audio_duration
            
            print(f"Synthesis complete for Speaker {target_speaker_id}.")
            print(f"Audio saved to: {output_path}")
            print(f"Audio duration: {audio_duration:.2f} seconds")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            print(f"Real Time Factor (RTF): {rtf:.4f}")

            # --- Second call to demonstrate caching ---
            print(f"\nSynthesizing for Speaker {target_speaker_id} --- Second call (should not reload model)")
            start_time_2nd = time.time()
            audio_data_2nd = synthesizer.synthesize("二回目の合成テストです。", target_speaker_id)
            end_time_2nd = time.time()

            output_path_2nd = "synthesis_output_sid{sid}_2nd.wav".format(sid=target_speaker_id)
            write_wav(output_path_2nd, synthesizer.sampling_rate, audio_data_2nd)

            elapsed_time_2nd = end_time_2nd - start_time_2nd
            audio_duration_2nd = len(audio_data_2nd) / synthesizer.sampling_rate
            rtf_2nd = elapsed_time_2nd / audio_duration_2nd

            print(f"Synthesis complete for Speaker {target_speaker_id} (2nd call).")
            print(f"Audio saved to: {output_path_2nd}")
            print(f"Audio duration: {audio_duration_2nd:.2f} seconds")
            print(f"Elapsed time: {elapsed_time_2nd:.2f} seconds")
            print(f"Real Time Factor (RTF): {rtf_2nd:.4f}")


        else:
            print(f"Target speaker ID {target_speaker_id} is not available.")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please update CONFIG_PATH and CHECKPOINT_PATH in the __main__ block.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")