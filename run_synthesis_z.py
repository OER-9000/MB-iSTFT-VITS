# --- Example Usage ---
if __name__ == '__main__':
    CONFIG_PATH = "./logs/uudb_csj31/config.json"
    CHECKPOINT_PATH = "./logs/uudb_csj31/G_3010000.pth"
    TEXT_TO_SYNTHESIZE = "こんにちは、これはテストです。"
    
    try:
        print("Requesting synthesis module instance...")
        synthesizer = get_synthesis_module_instance(config_path=CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH)
        num_speakers = synthesizer.get_speaker_count()
        print(f"Model supports {num_speakers} speakers.")
        
        target_speaker_id = 375
        if num_speakers > target_speaker_id:
            # --- 1. Test synthesize_with_z ---
            print(f"\n--- Testing synthesize_with_z for Speaker {target_speaker_id} ---")
            audio_from_text, z_from_text = synthesizer.synthesize_with_z(TEXT_TO_SYNTHESIZE, target_speaker_id)
            
            print(f"Synthesis with z complete.")
            print(f" - Returned audio shape: {audio_from_text.shape}")
            print(f" - Returned z shape: {z_from_text.shape}")

            output_path_sz = f"synthesis_output_with_z_sid{target_speaker_id}.wav"
            write_wav(output_path_sz, synthesizer.sampling_rate, audio_from_text)
            print(f"Audio saved to {output_path_sz}")

            # --- 2. Verify by re-synthesizing from the returned z ---
            print("\n--- Verifying by re-synthesizing from the returned z ---")
            audio_from_returned_z = synthesizer.infer_z_only(z_from_text, target_speaker_id)
            output_path_verify = f"synthesis_verify_from_z_sid{target_speaker_id}.wav"
            write_wav(output_path_verify, synthesizer.sampling_rate, audio_from_returned_z)
            print(f"Re-synthesized audio saved to {output_path_verify}")
            
            if platform.system() == 'Darwin':
                print("\nPlaying original audio (from synthesize_with_z)...")
                os.system(f"afplay {output_path_sz}")
                time.sleep(1)
                print("Playing re-synthesized audio (from infer_z_only)...")
                os.system(f"afplay {output_path_verify}")
        else:
            print(f"Target speaker ID {target_speaker_id} is not available.")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")