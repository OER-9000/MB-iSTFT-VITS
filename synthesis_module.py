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
import MeCab
import unidic_lite
from stft import TorchSTFT
from pqmf import PQMF

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

        print("Initializing Phonemizer and MeCab Tagger with unidic-lite...")
        self.phonemizer = Phonemizer()
        # Use unidic-lite dictionary installed via pip
        self.mecab_tagger = MeCab.Tagger(f"-d {unidic.DICDIR}")
        print("Phonemizer and MeCab Tagger initialized.")

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

    def _split_text_to_bunsetsu_mecab(self, text):
        """
        Splits text into a list of (surface, reading) tuples for each bunsetsu.
        The reading is extracted from MeCab's feature analysis.
        """
        node = self.mecab_tagger.parseToNode(text)
        chunks = []
        current_chunk_surface = ""
        current_chunk_reading = ""
        
        while node:
            if not node.surface:
                node = node.next
                continue
                
            features = node.feature.split(",")
            
            # For unidic, the reading is typically the 8th element (index 7).
            # If it's '*', it might be the 7th (index 6).
            # This might need adjustment depending on the dictionary.
            reading = features[7] if len(features) > 7 and features[7] != '*' else (features[6] if len(features) > 6 and features[6] != '*' else node.surface)

            pos1 = features[0]
            pos2 = features[1]
            
            is_independent = False
            if pos1 in ["名詞", "動詞", "形容詞", "副詞", "連体詞", "接続詞", "感動詞", "接頭詞", "形状詞", "代名詞"]:
                if pos2 not in ["非自立", "接尾"]:
                    is_independent = True
            
            if is_independent and current_chunk_surface:
                chunks.append((current_chunk_surface, current_chunk_reading))
                current_chunk_surface = ""
                current_chunk_reading = ""
                
            current_chunk_surface += node.surface
            current_chunk_reading += reading
            node = node.next
            
        if current_chunk_surface:
            chunks.append((current_chunk_surface, current_chunk_reading))
            
        return chunks

    def _get_phoneme_chunks(self, raw_text):
        """
        Splits raw text into phoneme chunks, using MeCab for bunsetsu splitting and reading extraction.
        """
        tokens = re.split(r'({cough}|<cough>|\[.*?\]|[、。])', raw_text)
        final_phoneme_chunks = []
        text_buffer = ""
        
        def flush_text_buffer():
            nonlocal text_buffer
            if not text_buffer: return
            
            bunsetsu_list = self._split_text_to_bunsetsu_mecab(text_buffer)
            
            for b_surface, b_reading in bunsetsu_list:
                # Use the reading from MeCab, not g2p
                k = b_reading.replace('ヲ', 'オ')
                p = self.phonemizer(k)
                if p.strip():
                    final_phoneme_chunks.append(p.strip())
            text_buffer = ""

        for token in tokens:
            if not token or token.isspace():
                continue
            
            if token in ["、", "。"]:
                if text_buffer:
                    bunsetsu_list = self._split_text_to_bunsetsu_mecab(text_buffer)
                    for i, (b_surface, b_reading) in enumerate(bunsetsu_list):
                        k = b_reading.replace('ヲ', 'オ')
                        p = self.phonemizer(k)
                        if p.strip():
                            if i == len(bunsetsu_list) - 1:
                                final_phoneme_chunks.append(p.strip() + " sp")
                            else:
                                final_phoneme_chunks.append(p.strip())
                    text_buffer = ""
                elif final_phoneme_chunks:
                    if not final_phoneme_chunks[-1].endswith(" sp"):
                         final_phoneme_chunks[-1] += " sp"
                else:
                    final_phoneme_chunks.append("sp")
                continue
            
            if (token.startswith("[") and token.endswith("]")) or token in ["{cough}", "<cough>"]:
                flush_text_buffer()
                
                if token.startswith("["):
                    content = token[1:-1]
                    if content:
                        # For tags, we still rely on g2p as MeCab features are not available.
                        k = pyopenjtalk.g2p(content, kana=True).replace('ヲ', 'オ')
                        p = self.phonemizer(k)
                        final_phoneme_chunks.append(f"[ {p} ]")
                    else:
                        final_phoneme_chunks.append("[ ]")
                else:
                    final_phoneme_chunks.append("<cough>")
                continue
                
            text_buffer += token
            
        flush_text_buffer()
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

    def _istft_finalize(self, full_complex_spec):
        """
        Multi-stream iSTFTに対応した波形再構成関数。
        full_complex_spec shape: [Batch, Subbands, Freq, Time]
        """
        device = full_complex_spec.device
        final_spec = torch.abs(full_complex_spec)
        final_phase = torch.angle(full_complex_spec)
        
        stft = TorchSTFT(
            filter_length=self.hps.data.filter_length,
            hop_length=self.hps.data.hop_length,
            win_length=self.hps.data.filter_length,
            center=False
        ).to(device)

        # Multi-stream処理の判定
        if hasattr(self.model.dec, 'subbands') and self.model.dec.subbands > 1:
            # [B, S, F, T] -> [B*S, F, T] に変形してiSTFT
            b, s, f, t = final_spec.shape
            spec_reshaped = final_spec.view(b * s, f, t)
            phase_reshaped = final_phase.view(b * s, f, t)
            
            y_mb_hat = stft.inverse(spec_reshaped, phase_reshaped) # -> [B*S, 1, Time_sub]
            y_mb_hat = y_mb_hat.squeeze(1).view(b, s, -1)          # -> [B, S, Time_sub]

            # 合成フィルタ (Synthesis Filter Bank)
            if self.model.ms_istft_vits:
                # 学習済みアップサンプリングフィルタを使用
                y_mb_hat = F.conv_transpose1d(
                    y_mb_hat, 
                    self.model.dec.updown_filter.to(device) * self.model.dec.subbands, 
                    stride=self.model.dec.subbands
                )
                # Ensure y_mb_hat is 3D (Batch, Channels, Length) before passing to Conv1d
                if y_mb_hat.dim() == 4:
                    y_mb_hat = y_mb_hat.squeeze(2) # Remove the 3rd dimension if it's 1
                elif y_mb_hat.dim() == 2:
                    y_mb_hat = y_mb_hat.unsqueeze(0) # Add batch dimension if it's [Channels, Length]
                
                audio_tensor = self.model.dec.multistream_conv_post(y_mb_hat)
            else:
                # PQMFまたは単純加算 (Fallback)
                # この部分はMultiband_iSTFT_Generatorの場合に相当するが、
                # ms_istft_vitsがTrueならMultistream_iSTFT_Generatorが使われるはず。
                try:
                    from pqmf import PQMF
                    pqmf = PQMF(device)
                    audio_tensor = pqmf.synthesis(y_mb_hat.unsqueeze(2)) 
                except ImportError:
                     audio_tensor = torch.sum(y_mb_hat, dim=1, keepdim=True)
        else:
            # 通常のiSTFT (Single stream)
            audio_tensor = stft.inverse(final_spec, final_phase)

        return audio_tensor[0, 0].data.cpu().float().numpy()

    def synthesize_spectrogram_concat_validation(self, z, w, chunk_counts, speaker_id):
        """
        Synthesizes audio by concatenating spectrograms, as requested by the user for validation purposes.
        This implementation now correctly handles MS-iSTFT-VITS sub-band reconstruction.
        """
        if z is None: return np.array([])

        with torch.no_grad():
            sid = torch.LongTensor([speaker_id]).to(self.device)
            g = self.model.emb_g(sid).unsqueeze(-1)

            w_flat = torch.from_numpy(w).to(self.device)
            z_tensor = torch.from_numpy(z).to(self.device).unsqueeze(0)

            full_complex_spec_subband = None
            current_ph_idx = 0
            current_z_frame = 0
            
            for count in chunk_counts:
                z_len = int(torch.sum(w_flat[current_ph_idx : current_ph_idx + count]).item())
                z_end_frame = current_z_frame + z_len
                if z_end_frame > z_tensor.shape[2]: z_end_frame = z_tensor.shape[2]
                
                z_chunk = z_tensor[:, :, current_z_frame : z_end_frame]
                
                if z_chunk.shape[2] > 0:
                    # dec returns: y_g_hat, y_mb_hat, spec, phase
                    # spec and phase are in sub-band format: [B, subbands, F, T]
                    _, _, spec_chunk, phase_chunk = self.model.dec(z_chunk, g=g)
                    
                    complex_chunk_subband = spec_chunk * torch.exp(1j * phase_chunk)
                    
                    if full_complex_spec_subband is None:
                        full_complex_spec_subband = complex_chunk_subband
                    else:
                        full_complex_spec_subband = torch.cat([full_complex_spec_subband, complex_chunk_subband], dim=-1)
                
                current_ph_idx += count
                current_z_frame = z_end_frame
                if current_z_frame >= z_tensor.shape[2]: break

            if full_complex_spec_subband is None: return np.array([])
            
            # Final iSTFT using the new _istft_finalize, which handles sub-band reconstruction
            return self._istft_finalize(full_complex_spec_subband)

    def _istft_finalize(self, complex_spec):
        """
        Performs iSTFT on a full-band complex spectrogram to get the final audio waveform.
        """
        n_fft = self.hps.data.filter_length
        hop_length = self.hps.data.hop_length
        win_length = self.hps.data.win_length
        
        stft = TorchSTFT(filter_length=n_fft, hop_length=hop_length, win_length=win_length, center=False).to(self.device)
        
        spec = torch.abs(complex_spec)
        phase = torch.angle(complex_spec)
        
        audio = stft.inverse(spec, phase)
        
        return audio.squeeze(0).cpu().float().numpy()

    def synthesize_spectrogram_concat_validation(self, z, w, chunk_counts, speaker_id):
        """
        Synthesizes audio by concatenating spectrograms, as requested by the user for validation purposes.
        This implementation is for MS-iSTFT-VITS and uses only the first sub-band for iSTFT.
        """
        if z is None: return np.array([])

        with torch.no_grad():
            sid = torch.LongTensor([speaker_id]).to(self.device)
            g = self.model.emb_g(sid).unsqueeze(-1)

            w_flat = torch.from_numpy(w).to(self.device)
            z_tensor = torch.from_numpy(z).to(self.device).unsqueeze(0)

            full_complex_spec_subband = None
            current_ph_idx = 0
            current_z_frame = 0
            
            for count in chunk_counts:
                z_len = int(torch.sum(w_flat[current_ph_idx : current_ph_idx + count]).item())
                z_end_frame = current_z_frame + z_len
                if z_end_frame > z_tensor.shape[2]: z_end_frame = z_tensor.shape[2]
                
                z_chunk = z_tensor[:, :, current_z_frame : z_end_frame]
                
                if z_chunk.shape[2] > 0:
                    # dec returns: y_g_hat, y_mb_hat, spec, phase
                    # spec and phase are in sub-band format: [B, subbands, F, T]
                    _, _, spec_chunk, phase_chunk = self.model.dec(z_chunk, g=g)
                    
                    complex_chunk_subband = spec_chunk * torch.exp(1j * phase_chunk)
                    
                    if full_complex_spec_subband is None:
                        full_complex_spec_subband = complex_chunk_subband
                    else:
                        full_complex_spec_subband = torch.cat([full_complex_spec_subband, complex_chunk_subband], dim=-1)
                
                current_ph_idx += count
                current_z_frame = z_end_frame
                if current_z_frame >= z_tensor.shape[2]: break

            if full_complex_spec_subband is None: return np.array([])
            
            # --- WARNING: Sub-band to Full-band Conversion ---
            # The following step is a simplification for validation purposes and is NOT acoustically correct
            # for a multi-band model like MS-iSTFT-VITS. It uses only the first sub-band,
            # which will result in low-frequency, muffled audio.
            # The correct approach for this architecture is to concatenate the decoded audio waveforms.
            print("Warning: Using only the first sub-band for spectrogram-based synthesis. The resulting audio will be muffled.")
            full_complex_spec_band0 = full_complex_spec_subband[:, 0, :, :]
            
            # Final iSTFT on the (incorrectly) constructed full-band spectrogram
            return self._istft_finalize(full_complex_spec_band0)

