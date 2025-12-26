import sys, os
import numpy
import queue
import time
import threading
import base64
import librosa
import torch
import re
import traceback
import json

# Project-specific imports
import commons
import utils
from models import SynthesizerTrn
from text_JP import cleaned_text_to_sequence, symbols
from text_JP.phonemize import Phonemizer
import pyopenjtalk

# Assuming 'base.py' is in the same directory or accessible in PYTHONPATH
from base import RemdisModule, RemdisUpdateType

class TTS(RemdisModule):
    def __init__(self, 
                 pub_exchanges=['tts'],
                 sub_exchanges=['dialogue']):
        super().__init__(pub_exchanges=pub_exchanges,
                         sub_exchanges=sub_exchanges)
        
        print("[INFO] TTS module: Initializing...")

        # --- General TTS Configuration ---
        # Use .get() for safe access with default values
        self.rate = self.config['TTS'].get('sample_rate', 24000)
        self.frame_length = self.config['TTS'].get('frame_length', 0.02)
        self.send_interval = self.config['TTS'].get('send_interval', 0.1)
        self.chunk_size = round(self.frame_length * self.rate)

        # --- VITS Model Configuration ---
        self.vits_config_path = self.config['TTS'].get('vits_config_path', "./configs/uudb_ms_istft_vits_ms.json")
        self.vits_checkpoint_path = self.config['TTS'].get('vits_checkpoint_path', "./logs/uudb_ms/G_latest.pth")
        self.speaker_id = self.config['TTS'].get('speaker_id', 0)

        # --- Synthesis Parameters ---
        self.noise_scale = self.config['TTS'].get('noise_scale', 0.667)
        self.noise_scale_w = self.config['TTS'].get('noise_scale_w', 0.8)
        self.length_scale = self.config['TTS'].get('length_scale', 1.0)
        self.auto_normalize = self.config['TTS'].get('auto_normalize', True)

        # --- System Setup ---
        self.input_iu_buffer = queue.Queue()
        self.output_iu_buffer = queue.Queue()
        self.is_revoked = False
        self._is_running = True

        # --- Device Configuration (Auto-select CUDA if available) ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] TTS module: Using device: {self.device}")

        # --- Initialize Model and Phonemizer ---
        self.phonemizer = Phonemizer()
        self._init_model()
        
        print("[INFO] TTS module: Initialization complete.")

    def _init_model(self):
        """Loads the VITS model from checkpoint."""
        if not os.path.exists(self.vits_config_path):
            raise FileNotFoundError(f"VITS config file not found at: {self.vits_config_path}")
        if not os.path.exists(self.vits_checkpoint_path):
            raise FileNotFoundError(f"VITS checkpoint file not found at: {self.vits_checkpoint_path}")

        print("[INFO] TTS module: Loading VITS model...")
        self.hps = utils.get_hparams_from_file(self.vits_config_path)

        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model).to(self.device)
        
        _ = self.net_g.eval()
        utils.load_checkpoint(self.vits_checkpoint_path, self.net_g, None)
        print("[INFO] TTS module: VITS model loaded successfully.")

    def _japanese_cleaner_revised(self, text):
        """Cleans and phonemizes Japanese text, handling special tokens."""
        parts = re.split(r'({cough}|<cough>|\[.*?\]|[、。])', text)
        phoneme_parts = []
        for part in parts:
            if not part or part.isspace():
                continue
            if part.startswith('[') and part.endswith(']'):
                content = part[1:-1]
                if not content:
                    phoneme_parts.append('[ ]')
                else:
                    kana_content = pyopenjtalk.g2p(content, kana=True).replace('ヲ', 'オ')
                    phoneme_content = self.phonemizer(kana_content)
                    phoneme_parts.append(f'[ {phoneme_content} ]')
            elif part == '{cough}' or part == '<cough>':
                phoneme_parts.append('<cough>')
            elif part in '、。':
                phoneme_parts.append('sp')
            else:
                kana = pyopenjtalk.g2p(part, kana=True).replace('ヲ', 'オ')
                phonemes = self.phonemizer(kana)
                phoneme_parts.append(phonemes)
        final_text = ' '.join(phoneme_parts)
        return re.sub(r'\s+', ' ', final_text).strip()

    def _text_to_sequence(self, text):
        """Converts cleaned text to a sequence of token IDs."""
        phonemized_text = self._japanese_cleaner_revised(text)
        sequence = cleaned_text_to_sequence(phonemized_text)
        if self.hps.data.add_blank:
            sequence = commons.intersperse(sequence, 0)
        return torch.LongTensor(sequence)

    def _synthesize_audio(self, text):
        """Synthesizes raw audio waveform from text."""
        if not text:
            return numpy.array([]), self.hps.data.sampling_rate

        sequence = self._text_to_sequence(text)
        
        with torch.no_grad():
            x_tst = sequence.unsqueeze(0).to(self.device)
            x_tst_lengths = torch.LongTensor([sequence.size(0)]).to(self.device)
            sid = torch.LongTensor([self.speaker_id]).to(self.device)
            
            audio = self.net_g.infer(x_tst, x_tst_lengths, sid=sid, 
                                     noise_scale=self.noise_scale, 
                                     noise_scale_w=self.noise_scale_w, 
                                     length_scale=self.length_scale)[0][0,0].data.cpu().float().numpy()
        
        return audio, self.hps.data.sampling_rate

    def run(self):
        """Starts the main threads for the TTS module."""
        print("[INFO] TTS module: Starting threads...")
        t1 = threading.Thread(target=self.listen_loop)
        t2 = threading.Thread(target=self.synthesis_loop)
        t3 = threading.Thread(target=self.send_loop)
        t1.start()
        t2.start()
        t3.start()
        print("[INFO] TTS module: All threads started.")
        t1.join()
        t2.join()
        t3.join()
        print("[INFO] TTS module: All threads joined.")

    def listen_loop(self):
        """Subscribes to the dialogue exchange and listens for messages."""
        print("[INFO] TTS module: listen_loop started.")
        self.subscribe('dialogue', self.callback)

    def send_loop(self):
        """Sends synthesized audio chunks from the output buffer."""
        print("[INFO] TTS module: send_loop started.")
        while self._is_running:
            if self.is_revoked:
                # Clear buffer and send a commit to signal cancellation
                self.output_iu_buffer = queue.Queue()
                self.send_commitIU('tts')
                self.is_revoked = False
                continue
            try:
                snd_iu = self.output_iu_buffer.get(block=True, timeout=1)
                self.publish(snd_iu, 'tts')
                time.sleep(self.send_interval)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] TTS send_loop: An exception occurred: {e}")
                traceback.print_exc()

    def synthesis_loop(self):
        """Main loop for synthesizing audio from the input buffer."""
        print("[INFO] TTS module: synthesis_loop started.")
        while self._is_running:
            if self.is_revoked:
                self.input_iu_buffer = queue.Queue()
                self.is_revoked = False
                continue
            try:
                in_msg = self.input_iu_buffer.get(block=True, timeout=1)
                output_text = in_msg['body']
                update_type = in_msg['update_type']

                if output_text:
                    # 1. Synthesize raw audio
                    audio_raw, model_sr = self._synthesize_audio(output_text)

                    # 2. Resample to target rate if necessary
                    if model_sr != self.rate:
                        audio_resampled = librosa.resample(audio_raw.astype(numpy.float32), orig_sr=model_sr, target_sr=self.rate)
                    else:
                        audio_resampled = audio_raw

                    # 3. Normalize audio (with margin)
                    max_abs_val = numpy.abs(audio_resampled).max()
                    if self.auto_normalize and max_abs_val > 0.01:
                        # Multiply by 0.9 to leave a small margin and prevent clipping
                        audio_normalized = (audio_resampled / max_abs_val) * 0.9
                    else:
                        audio_normalized = audio_resampled
                    
                    # 4. Clip to ensure values are within [-1.0, 1.0] range
                    audio_clipped = numpy.clip(audio_normalized, -1.0, 1.0)

                    # 5. Convert to 16-bit integer
                    audio_int16 = (audio_clipped * 32767).astype(numpy.int16)

                    # 6. Chunk and send
                    t = 0
                    while t < len(audio_int16):
                        chunk = audio_int16[t:t+self.chunk_size]
                        chunk_b64 = base64.b64encode(chunk.tobytes()).decode('utf-8')
                        snd_iu = self.createIU(chunk_b64, 'tts', update_type)
                        snd_iu['data_type'] = 'audio'
                        self.output_iu_buffer.put(snd_iu)
                        t += self.chunk_size
                
                # Send a commit IU after processing all chunks for the text
                self.send_commitIU('tts')

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] TTS synthesis_loop: An error occurred: {e}")
                traceback.print_exc()

    def send_commitIU(self, channel):
        """Creates and queues a COMMIT IU."""
        commit_iu = self.createIU('', channel, RemdisUpdateType.COMMIT)
        commit_iu['data_type'] = 'audio'
        self.output_iu_buffer.put(commit_iu)

    def callback(self, ch, method, properties, in_msg):
        """Callback function for handling incoming RabbitMQ messages."""
        in_msg = self.parse_msg(in_msg)
        self.printIU(in_msg)
        
        if in_msg['update_type'] == RemdisUpdateType.REVOKE:
            print("[INFO] TTS module: REVOKE received. Clearing buffers.")
            self.is_revoked = True
        else:
            self.input_iu_buffer.put(in_msg)

def main():
    try:
        tts_module = TTS()
        tts_module.run()
    except FileNotFoundError as e:
        print(f"[FATAL] A required file was not found: {e}")
    except KeyboardInterrupt:
        print("\n[INFO] Stopping TTS module...")
        if 'tts_module' in locals() and tts_module:
            tts_module._is_running = False
        time.sleep(2) # Give threads time to finish
        print("[INFO] TTS module stopped.")
    except Exception as e:
        print(f"[FATAL] An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
