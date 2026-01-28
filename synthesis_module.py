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
import platform

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

        print("Initializing Phonemizer...")
        self.phonemizer = Phonemizer()
        print("Phonemizer initialized.")

        print("Initializing Phonemizer...")
        self.phonemizer = Phonemizer()
        print("Phonemizer initialized.")

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
        """
        audio, _ = self.synthesize_with_z(text, speaker_id, noise_scale, noise_scale_w, length_scale)
        return audio

    def infer_z_only(self, z, speaker_id):
        """
        Synthesizes audio directly from a latent representation 'z'.
        """
        if speaker_id >= self.get_speaker_count():
            raise ValueError(f"Invalid speaker_id {speaker_id}. Model has {self.get_speaker_count()} speakers.")

        with torch.no_grad():
            z_tensor = torch.from_numpy(z).unsqueeze(0).to(self.device, dtype=torch.float)
            sid = torch.LongTensor([speaker_id]).to(self.device)
            g = self.model.emb_g(sid).unsqueeze(-1)
            z_mask = torch.ones(1, 1, z_tensor.shape[2], device=self.device, dtype=z_tensor.dtype)
            o, _, _, _ = self.model.dec((z_tensor * z_mask), g=g)
            audio = o[0,0].data.cpu().float().numpy()
        return audio

    def synthesize_with_z(self, text, speaker_id, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0):
        """
        Synthesizes audio from text and also returns the intermediate latent representation 'z'.
        """
        if speaker_id >= self.get_speaker_count():
            raise ValueError(f"Invalid speaker_id {speaker_id}. Model has {self.get_speaker_count()} speakers.")

        stn_tst = _text_to_sequence_custom(text, self.hps)

        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.device)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
            sid = torch.LongTensor([speaker_id]).to(self.device)
            
            infer_outputs = self.model.infer(
                x_tst, 
                x_tst_lengths, 
                sid=sid, 
                noise_scale=noise_scale, 
                noise_scale_w=noise_scale_w, 
                length_scale=length_scale
            )
            
            audio = infer_outputs[0][0,0].data.cpu().float().numpy()
            # Squeeze the batch dimension from z before returning
            z = infer_outputs[6][0][0].data.cpu().float().numpy()
            
        return audio, z

    def _get_phoneme_chunks(self, raw_text):
        """
        Splits raw text into phoneme chunks, handling tags and punctuation.
        Uses pyopenjtalk for bunsetsu splitting of normal text parts.
        """
        # 1. Split text roughly by tags and punctuation
        tokens = re.split(r'({cough}|<cough>|\[.*?\]|[、。])', raw_text)
        
        final_phoneme_chunks = []
        
        for token in tokens:
            if not token or token.isspace():
                continue
                
            # A. Punctuation -> Add 'sp' to the last chunk
            if token in ["、", "。"]:
                if final_phoneme_chunks:
                    if not final_phoneme_chunks[-1].endswith(" sp"):
                        final_phoneme_chunks[-1] += " sp"
                else:
                    final_phoneme_chunks.append("sp")
                continue
                
            # B. Tags like [...] or {cough}
            if (token.startswith("[") and token.endswith("]")) or token in ["{cough}", "<cough>"]:
                if token.startswith("["):
                    content = token[1:-1]
                    if content:
                        k = pyopenjtalk.g2p(content, kana=True).replace('ヲ', 'オ')
                        p = self.phonemizer(k)
                        final_phoneme_chunks.append(f"[ {p} ]")
                    else:
                        final_phoneme_chunks.append("[ ]")
                else:
                    final_phoneme_chunks.append("<cough>")
                continue
            
            # C. Normal text -> Use pyopenjtalk.run_frontend
            contexts = pyopenjtalk.run_frontend(token)
            if not contexts:
                continue

            current_kana_phrase = ""
            for c in contexts:
                # Split by accent phrase for finer granularity.
                is_new_phrase = False
                if 'label_info' in c and c['label_info'] and 'a' in c['label_info'] and c['label_info']['a'] and 'a1' in c['label_info']['a']:
                    is_new_phrase = c['label_info']['a']['a1'] == 1
                
                if is_new_phrase and current_kana_phrase:
                    p = self.phonemizer(current_kana_phrase)
                    if p.strip():
                        final_phoneme_chunks.append(p)
                    current_kana_phrase = ""
                
                current_kana_phrase += c['string']
            
            # Add the last phrase from the current text token
            if current_kana_phrase:
                p = self.phonemizer(current_kana_phrase)
                if p.strip():
                    final_phoneme_chunks.append(p)

        return final_phoneme_chunks

    def prepare_shared_latents(self, raw_text, speaker_id, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0):
        """
        Generates shared latent representation 'z' and bunsetsu information.
        """
        if speaker_id >= self.get_speaker_count():
            raise ValueError(f"Invalid speaker_id {speaker_id}. Model has {self.get_speaker_count()} speakers.")

        # Get phoneme chunks using the dedicated helper method
        bunsetsu_phonemes = self._get_phoneme_chunks(raw_text)
        
        # --- The rest of the function is for encoding ---
        all_phoneme_ids = []
        chunk_phoneme_counts = []
        
        for ph in bunsetsu_phonemes:
            if not ph.strip(): continue
            
            sequence = cleaned_text_to_sequence(ph)
            if self.hps.data.add_blank:
                sequence = commons.intersperse(sequence, 0)
            
            chunk_phoneme_counts.append(len(sequence))
            all_phoneme_ids.extend(sequence)
            
        if not all_phoneme_ids:
            return None, None, [], []

        stn_tst = torch.LongTensor(all_phoneme_ids)
        
        with torch.no_grad():
            x_tst = stn_tst.to(self.device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
            sid = torch.LongTensor([speaker_id]).to(self.device)
            
            infer_outputs = self.model.infer(
                x_tst, 
                x_tst_lengths, 
                sid=sid, 
                noise_scale=noise_scale, 
                noise_scale_w=noise_scale_w, 
                length_scale=length_scale
            )
            
            z = infer_outputs[6][0] # Shape: (dim, T)
            w = infer_outputs[4][0, 0] # Shape: (T_text)

        return z.data.cpu().float().numpy(), w.data.cpu().float().numpy(), chunk_phoneme_counts, bunsetsu_phonemes

    def synthesize_from_shared_latents(self, z, w, chunk_counts, speaker_id):
        """
        Synthesizes audio by decoding shared latents chunk by chunk and concatenating spectrograms.
        """
        if z is None: return np.array([])

        # Get speaker embedding 'g'
        with torch.no_grad():
            sid = torch.LongTensor([speaker_id]).to(self.device)
            g = self.model.emb_g(sid).unsqueeze(-1)

        w_flat = torch.from_numpy(w).to(self.device)
        z_tensor = torch.from_numpy(z).to(self.device).unsqueeze(0) # Add batch dim

        full_audio_mb = None # Multi-band audio tensor
        current_ph_idx = 0
        current_z_frame = 0
        
        with torch.no_grad():
            for count in chunk_counts:
                z_len = int(torch.sum(w_flat[current_ph_idx : current_ph_idx + count]).item())
                
                z_end_frame = current_z_frame + z_len
                if z_end_frame > z_tensor.shape[2]: z_end_frame = z_tensor.shape[2]
                
                z_chunk = z_tensor[:, :, current_z_frame : z_end_frame]
                
                if z_chunk.shape[2] > 0:
                    # Use the decode method of the model to get post-net output
                    o_chunk, _, _, _ = self.model.decode(z_chunk, g=g)
                    
                    if full_audio_mb is None:
                        full_audio_mb = o_chunk
                    else:
                        full_audio_mb = torch.cat([full_audio_mb, o_chunk], dim=-1)

                current_ph_idx += count
                current_z_frame = z_end_frame
                if current_z_frame >= z_tensor.shape[2]: break

        if full_audio_mb is None: return np.array([])

        # Final iSTFT
        with torch.no_grad():
            audio = self.model.mb_istft(full_audio_mb, self.model.pqmf)
            audio = audio.squeeze().cpu().float().numpy()
            
        return audio

