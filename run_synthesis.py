import time
import os
import numpy as np
from scipy.io.wavfile import write as write_wav
from synthesis_module import get_synthesis_module_instance
import platform
import base64
import pyaudio

if __name__ == '__main__':
    # --- Configuration (EDIT THESE PATHS) ---
    CONFIG_PATH = "./logs/uudb_csj31/config.json"
    CHECKPOINT_PATH = "./logs/uudb_csj31/G_3010000.pth"
    
    TEXT_TO_SYNTHESIZE = "あらゆる現実を、全て自分の方へ捻じ曲げたのだ。"
    OUTPUT_WAV_PATH = "synthesis_output_sid{sid}.wav"
    
    # --- Main Execution ---
    try:
        print("Requesting synthesis module instance...")
        synthesizer = get_synthesis_module_instance(config_path=CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH)
        
        num_speakers = synthesizer.get_speaker_count()
        print(f"Model supports {num_speakers} speakers.")
        
        target_speaker_id = 373
        if num_speakers > target_speaker_id:
            print(f"\nSynthesizing for Speaker {target_speaker_id}")
            start_time = time.time()
            
            # 1. Synthesize audio to get raw PCM data (float32 NumPy array)
            audio_data = synthesizer.synthesize(TEXT_TO_SYNTHESIZE, target_speaker_id)
            
            end_time = time.time()
            
            # Save the original audio to a file for verification
            output_path = OUTPUT_WAV_PATH.format(sid=target_speaker_id)
            write_wav(output_path, synthesizer.sampling_rate, audio_data)
            
            # --- Performance Metrics ---
            elapsed_time = end_time - start_time
            audio_duration = len(audio_data) / synthesizer.sampling_rate
            rtf = elapsed_time / audio_duration
            
            print(f"Synthesis complete for Speaker {target_speaker_id}.")
            print(f"Original audio saved to: {output_path}")
            print(f"Audio duration: {audio_duration:.2f} seconds")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            print(f"Real Time Factor (RTF): {rtf:.4f}")

            # 2. Convert NumPy array to raw PCM bytes and encode to Base64
            print("\nEncoding raw PCM data to Base64...")
            # Ensure data is in float32 format before converting to bytes
            pcm_data_bytes = audio_data.astype(np.float32).tobytes()
            base64_data = base64.b64encode(pcm_data_bytes).decode('utf-8')
            print(f"Base64 encoded data (first 80 chars): {base64_data[:80]}...")

            # 3. Decode Base64 data back to raw PCM bytes
            print("\nDecoding Base64 data back to raw PCM...")
            decoded_pcm_bytes = base64.b64decode(base64_data)

            # 4. Play the decoded audio using PyAudio
            print("Playing decoded audio with PyAudio...")
            p = pyaudio.PyAudio()

            # Open stream
            stream = p.open(format=pyaudio.paFloat32,
                            channels=1, # Mono
                            rate=synthesizer.sampling_rate,
                            output=True)

            # Play stream
            stream.write(decoded_pcm_bytes)

            # Stop stream
            stream.stop_stream()
            stream.close()

            # Close PyAudio
            p.terminate()
            print("Playback finished.")

        else:
            print(f"Target speaker ID {target_speaker_id} is not available.")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please update CONFIG_PATH and CHECKPOINT_PATH in the __main__ block.")
    except ImportError:
        print("ERROR: PyAudio is not installed. Please install it in your environment.")
        print("e.g., 'conda install -c anaconda portaudio' followed by 'pip install pyaudio'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")