
import os
import re
import time
import json
import torch
import commons
import utils
from models import SynthesizerTrn
from text_JP import cleaned_text_to_sequence, symbols
import pyopenjtalk
from text_JP.phonemize import Phonemizer
import numpy as np
from scipy.io.wavfile import write as write_wav

# --- Module-level cache for SynthesisModule instance ---
_synthesizer_instance = None

def get_synthesis_module_instance(config_path, checkpoint_path, device=None):
    """
    Returns a singleton instance of SynthesisModule.
    Loads the model only on the first call.
    """
    global _synthesizer_instance
    if _synthesizer_instance is None:
        print("Creating new SynthesisModule instance (loading model)...")
        _synthesizer_instance = SynthesisModule(config_path, checkpoint_path, device)
    else:
        print("Using existing SynthesisModule instance (model already loaded).")
    return _synthesizer_instance

# --- Text Pre-processing Functions ---

def _japanese_cleaner_revised(text):
    """
    Custom Japanese text cleaner that handles special tokens and converts text to phonemes.
    """
    parts = re.split(r'({cough}|<cough>|\[.*?\]|[、。])', text)
    phoneme_parts = []
    phonemizer = Phonemizer()
    for part in parts:
        if not part or part.isspace():
            continue
        if part.startswith('[') and part.endswith(']') and len(part) > 2:
            content = part[1:-1]
            if not content:
                phoneme_parts.append('[ ]')
            else:
                # Assuming g2p and phonemizer are available and configured
                kana_content = pyopenjtalk.g2p(content, kana=True).replace('ヲ', 'オ')
                phoneme_content = phonemizer(kana_content)
                phoneme_parts.append(f'[ {phoneme_content} ]')
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

def _text_to_sequence_custom(text, hps):
    """
    Converts cleaned text to a sequence of symbol IDs.
    """
    phonemized_text = _japanese_cleaner_revised(text)
    stn_tst = cleaned_text_to_sequence(phonemized_text)
    if hps.data.add_blank:
        stn_tst = commons.intersperse(stn_tst, 0)
    return torch.LongTensor(stn_tst)


# --- Main Synthesis Module ---

class SynthesisModule:
    """
    A module for synthesizing speech from text using a trained VITS model.
    """
    def __init__(self, config_path, checkpoint_path, device=None):
        """
        Initializes the synthesis module by loading the model and configuration.

        Args:
            config_path (str): Path to the model's config.json file.
            checkpoint_path (str): Path to the model's .pth checkpoint file.
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). 
                                    If None, automatically detects CUDA.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        self.hps = utils.get_hparams_from_file(config_path)
        
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading model...")
        self.model = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        ).to(self.device)
        
        self.model.eval()
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        utils.load_checkpoint(checkpoint_path, self.model, None)
        print("Model loaded successfully.")

    def get_speaker_count(self):
        """
        Returns the total number of speakers the model was trained on.
        """
        return self.hps.data.n_speakers

    @property
    def sampling_rate(self):
        """
        Returns the sampling rate of the model.
        """
        return self.hps.data.sampling_rate

    def synthesize(self, text, speaker_id, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0):
        """
        Synthesizes audio from the given text for a specific speaker.

        Args:
            text (str): The text to synthesize.
            speaker_id (int): The ID of the speaker.
            noise_scale (float, optional): Noise scale for inference. Defaults to 0.667.
            noise_scale_w (float, optional): Noise scale width for inference. Defaults to 0.8.
            length_scale (float, optional): Length scale for inference. Defaults to 1.0.

        Returns:
            np.ndarray: A numpy array containing the synthesized audio waveform.
        """
        if speaker_id >= self.get_speaker_count():
            raise ValueError(f"Invalid speaker_id {speaker_id}. Model has {self.get_speaker_count()} speakers.")

        # Process text
        stn_tst = _text_to_sequence_custom(text, self.hps)

        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.device)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
            sid = torch.LongTensor([speaker_id]).to(self.device)
            
            # Inference
            audio = self.model.infer(
                x_tst, 
                x_tst_lengths, 
                sid=sid, 
                noise_scale=noise_scale, 
                noise_scale_w=noise_scale_w, 
                length_scale=length_scale
            )[0][0,0].data.cpu().float().numpy()
        
        return audio

# --- Example Usage ---
if __name__ == '__main__':
    # --- Configuration (EDIT THESE PATHS) ---
    CONFIG_PATH = "./logs/uudb_csj31/config.json"
    # Find the latest G_****.pth file in your model directory
    CHECKPOINT_PATH = "./logs/uudb_csj31/G_3010000.pth"
    
    TEXT_TO_SYNTHESIZE = "こんにちは、これはテストです。音声合成モジュールが正しく動作していますか？"
    OUTPUT_WAV_PATH = "synthesis_output_sid{sid}.wav"
    
    # --- Main Execution ---
    try:
        print("Requesting synthesis module instance...")
        synthesizer = get_synthesis_module_instance(config_path=CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH)
        
        num_speakers = synthesizer.get_speaker_count()
        print(f"Model supports {num_speakers} speakers.")
        
        # Synthesize for a specific speaker (e.g., speaker 0)
        target_speaker_id = 375
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

