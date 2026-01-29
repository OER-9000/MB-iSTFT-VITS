import os
import re
import time
import json
import torch
import torch.nn.functional as F
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
import unidic

# --- Module-level cache for SynthesisModule instance ---
_synthesizer_instance = None


class TorchSTFT(torch.nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length, periodic=True)

    def inverse(self, magnitude, phase):
        # Ensure inputs are Tensors
        if isinstance(magnitude, np.ndarray):
            magnitude = torch.from_numpy(magnitude).to(self.window.device)
        if isinstance(phase, np.ndarray):
            phase = torch.from_numpy(phase).to(self.window.device)
            
        complex_spec = magnitude * torch.exp(phase * 1j)
        inverse_transform = torch.istft(
            complex_spec,
            self.filter_length, self.hop_length, self.win_length, 
            window=self.window.to(complex_spec.device)
        )
        return inverse_transform.unsqueeze(1)


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





class TorchSTFT(torch.nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length, periodic=True)

    def inverse(self, magnitude, phase):
        complex_spec = magnitude * torch.exp(phase * 1j)
        inverse_transform = torch.istft(
            complex_spec,
            self.filter_length, self.hop_length, self.win_length, 
            window=self.window.to(complex_spec.device)
        )
        return inverse_transform.unsqueeze(1)

def split_text_to_bunsetsu(text):
    """MeCabを用いてテキストを文節単位のリストに分割する"""
    if MeCab is None:
        raise RuntimeError("MeCab is not installed. Please install 'mecab-python3' and a dictionary.")
    
    dic_dir = getattr(unidic, 'DICDIR', None)
     # Use unidic dictionary installed via pip
    tagger = MeCab.Tagger(f"-d {dic_dir} -r /dev/null")

    node = tagger.parseToNode(text)
    chunks = []
    current_chunk = ""
    
    while node:
        if node.surface == "":
            node = node.next
            continue
        features = node.feature.split(",")
        pos1 = features[0]
        pos2 = features[1]
        
        is_independent = False
        if pos1 in ["名詞", "動詞", "形容詞", "副詞", "連体詞", "接続詞", "感動詞", "接頭詞", "形状詞", "代名詞"]:
            if pos2 not in ["非自立", "接尾"]:
                is_independent = True
        
        if is_independent and current_chunk:
            chunks.append(current_chunk)
            current_chunk = ""
            
        current_chunk += node.surface
        node = node.next
        
    if current_chunk:
        chunks.append(current_chunk)
    return chunks





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

    def prepare_shared_latents(self, text, sid=0, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0):
        """
        Cond 2用: テキストから潜在変数 (z, w, g) と文節リストを一括生成
        Returns:
            z, w_ceil, g, bunsetsu_chunks
        """
        sid_tensor = torch.LongTensor([int(sid)]).to(self.device)
        bunsetsu_chunks = self._get_bunsetsu_chunks_mecab(text)
        
        all_phoneme_ids = []
        for ph in bunsetsu_chunks:
            if not ph: continue
            ids = self._get_text_from_phonemes(ph)
            all_phoneme_ids.extend(ids.tolist())
            
        if not all_phoneme_ids:
            return None, None, None, []

        stn_tst = torch.LongTensor(all_phoneme_ids)
        
        with torch.no_grad():
            x_tst = stn_tst.to(self.device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
            
            x, m_p, logs_p, x_mask = self.model.enc_p(x_tst, x_tst_lengths)
            
            # g (Speaker Embedding)
            if self.model.n_speakers > 0:
                g = self.model.emb_g(sid_tensor).unsqueeze(-1)
            else:
                g = None

            logw = self.model.dp(x, x_mask, g=g)
            w = torch.exp(logw) * x_mask * length_scale
            w_ceil = torch.ceil(w)
            
            # Alignment / Expand
            B, _, T_phoneme = w_ceil.shape
            T_frame = int(torch.sum(w_ceil).item())
            
            attn_mask = torch.zeros(B, T_phoneme, T_frame).to(self.device)
            w_ceil_flat = w_ceil.squeeze()
            current_frame = 0
            for i, dur in enumerate(w_ceil_flat):
                d = int(dur.item())
                if d > 0:
                    attn_mask[0, i, current_frame : current_frame + d] = 1.0
                    current_frame += d
            
            m_p = torch.matmul(m_p, attn_mask)
            logs_p = torch.matmul(logs_p, attn_mask)

            y_mask = torch.ones(B, 1, T_frame).to(self.device)
            z_p = m_p + torch.randn_like(m_p, dtype=torch.float) * torch.exp(logs_p) * noise_scale
            z = self.model.flow(z_p, y_mask, g=g, reverse=True)

        return z, w_ceil, g, bunsetsu_chunks
    
    def _get_text_from_phonemes(self, phonemes):
        """音素文字列をID列に変換"""
        symbol_to_id = {s: i for i, s in enumerate(symbols)}
        clean_phonemes = phonemes.replace("[", "").replace("]", "").strip()
        phoneme_list = clean_phonemes.split(" ")
        text_norm = []
        for p in phoneme_list:
            if p in symbol_to_id:
                text_norm.append(symbol_to_id[p])
        if self.hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        return torch.LongTensor(text_norm)

    def _get_bunsetsu_chunks_mecab(self, text):
        """テキストを文節に分割し、音素化してリストで返す"""
        tokens = re.split(r'({cough}|<cough>|\[.*?\]|[、。])', text)
        final_phoneme_chunks = []
        text_buffer = ""
        
        def flush_text_buffer():
            nonlocal text_buffer
            if not text_buffer: return
            bunsetsu_list = split_text_to_bunsetsu(text_buffer)
            for b_text in bunsetsu_list:
                k = pyopenjtalk.g2p(b_text, kana=True).replace('ヲ', 'オ')
                p = self.phonemizer(k)
                if p.strip():
                    final_phoneme_chunks.append(p.strip())
            text_buffer = ""

        for token in tokens:
            if not token or token.isspace():
                continue
            if token in ["、", "。"]:
                if text_buffer:
                    bunsetsu_list = split_text_to_bunsetsu(text_buffer)
                    for i, b_text in enumerate(bunsetsu_list):
                        k = pyopenjtalk.g2p(b_text, kana=True).replace('ヲ', 'オ')
                        p = self.phonemizer(k)
                        if p.strip():
                            if i == len(bunsetsu_list) - 1:
                                final_phoneme_chunks.append(p.strip() + " sp")
                            else:
                                final_phoneme_chunks.append(p.strip())
                    text_buffer = ""
                else:
                    if final_phoneme_chunks:
                        final_phoneme_chunks[-1] += " sp"
                    else:
                        final_phoneme_chunks.append("sp")
                continue
            
            if (token.startswith("[") and token.endswith("]")) or token in ["{cough}", "<cough>"]:
                flush_text_buffer()
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
            text_buffer += token
            
        flush_text_buffer()
        return final_phoneme_chunks

    def _get_z_and_phoneme_durations(self, x_tst, x_tst_lengths, sid, noise_scale, noise_scale_w, length_scale):
        """全体を一括エンコードしてzとdurationを取得"""
        # Text Encoder
        x, m_p, logs_p, x_mask = self.model.enc_p(x_tst, x_tst_lengths)
        
        # Speaker Embedding
        if self.model.n_speakers > 0:
            g = self.model.emb_g(sid).unsqueeze(-1)
        else:
            g = None

        # Duration Predictor
        logw = self.model.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        
        # Expand (Alignment Mask)
        B, _, T_phoneme = w_ceil.shape
        T_frame = int(torch.sum(w_ceil).item())
        attn_mask = torch.zeros(B, T_phoneme, T_frame).to(x.device)
        
        w_ceil_flat = w_ceil.squeeze()
        current_frame = 0
        for i, dur in enumerate(w_ceil_flat):
            d = int(dur.item())
            if d > 0:
                attn_mask[0, i, current_frame : current_frame + d] = 1.0
                current_frame += d
        
        m_p = torch.matmul(m_p, attn_mask)
        logs_p = torch.matmul(logs_p, attn_mask)

        # Flow (Reverse) -> z
        y_mask = torch.ones(B, 1, T_frame).to(x.device)
        z_p = m_p + torch.randn_like(m_p, dtype=torch.float) * torch.exp(logs_p) * noise_scale
        z = self.model.flow(z_p, y_mask, g=g, reverse=True)
        
        return z, w_ceil, g

    def _istft_finalize(self, full_complex_spec):
        """Reconstruct waveform from spectrogram using iSTFT"""
        device = full_complex_spec.device
        final_spec = torch.abs(full_complex_spec)
        final_phase = torch.angle(full_complex_spec)
        
        # Determine STFT parameters from model config if available
        n_fft = getattr(self.model.dec, 'gen_istft_n_fft', self.hps.data.filter_length)
        hop_size = getattr(self.model.dec, 'gen_istft_hop_size', self.hps.data.hop_length)
        
        stft = TorchSTFT(
            filter_length=n_fft, 
            hop_length=hop_size, 
            win_length=n_fft
        ).to(device)

        # Multi-band iSTFT processing
        if hasattr(self.model.dec, 'subbands') and self.model.dec.subbands > 1:
            b, s, f, t = final_spec.shape
            spec_reshaped = final_spec.view(b * s, f, t)
            phase_reshaped = final_phase.view(b * s, f, t)
            
            y_mb_hat = stft.inverse(spec_reshaped, phase_reshaped)
            y_mb_hat = y_mb_hat.squeeze(1).view(b, s, -1)

            if getattr(self.model, 'ms_istft_vits', False):
                y_mb_hat = F.conv_transpose1d(
                    y_mb_hat, 
                    self.model.dec.updown_filter.to(device) * self.model.dec.subbands, 
                    stride=self.model.dec.subbands
                )
                audio_tensor = self.model.dec.multistream_conv_post(y_mb_hat)
            else:
                try:
                    from pqmf import PQMF
                    pqmf = PQMF(device)
                    audio_tensor = pqmf.synthesis(y_mb_hat.unsqueeze(2)) 
                except ImportError:
                    # Fallback: simple sum (ensure tensor)
                    if isinstance(y_mb_hat, np.ndarray):
                        y_mb_hat = torch.from_numpy(y_mb_hat).to(device)
                    audio_tensor = torch.sum(y_mb_hat, dim=1, keepdim=True)
        else:
            # Single-band iSTFT
            audio_tensor = stft.inverse(final_spec, final_phase)

        return audio_tensor[0, 0].data.cpu().float().numpy()
    
    def _find_best_time_delay(self, wav_ref, wav_tar, max_shift_samples=300):
        """
        FFTベースの相互相関（相関定理）を用いて時間ラグを推定する
        wav_ref: 前のセグメントの末尾 (Numpy)
        wav_tar: 今のセグメントの先頭 (Numpy)
        """
        # DC除去
        ref = wav_ref - np.mean(wav_ref)
        tar = wav_tar - np.mean(wav_tar)
        
        n_samples = max(len(ref), len(tar))
        fft_size = 1
        while fft_size < n_samples * 2:
            fft_size *= 2
            
        X1 = np.fft.fft(ref, fft_size)
        X2 = np.fft.fft(tar, fft_size)
        
        # 相互相関: xcor = IFFT(X1 * conj(X2))
        xcor = np.real(np.fft.ifft(X1 * np.conj(X2)))
        
        # ピーク検出 (マスク付き)
        mask = np.zeros_like(xcor)
        mask[0 : max_shift_samples + 1] = 1.0
        mask[-max_shift_samples : ] = 1.0
        
        lagidx = np.argmax(xcor * mask)
        
        lag = 0
        if lagidx > fft_size // 2:
            lag = lagidx - fft_size 
        else:
            lag = lagidx
            
        # 修正: 検証結果に基づき符号を反転させる (または定義を合わせる)
        # 前回の検証で -10 だったものが 10 になるようにする
        return -lag

    def _apply_group_delay_correction(self, complex_spec, delay_samples):
        """
        群遅延補正（位相の微分に基づく線形位相シフト）
        Shift Theorem: x(t+d) <-> X(w) * exp(j*w*d)
        
        delay_samples (d) が正の場合:
          Targetが遅れているため、波形を「進める」（左シフト）必要がある。
          補正項: exp(j * 2*pi * k/N * d)
        
        delay_samples (d) が負の場合:
          Targetが進んでいるため、波形を「遅らせる」（右シフト）必要がある。
          補正項: exp(j * 2*pi * k/N * d)  (dが負なので指数部はマイナス)
        """
        device = complex_spec.device
        n_freq = complex_spec.shape[-2]
        n_fft = (n_freq - 1) * 2
        
        k = torch.arange(n_freq, device=device).float()
        
        # 線形位相シフト項 (ラジアン)
        # formula: phi += 2 * pi * k * delay / N
        phase_shift_rad = 2 * np.pi * k * delay_samples / n_fft
        
        phase_shift_rad = phase_shift_rad.view(1, 1, n_freq, 1)
        
        # 回転子 (Rotator)
        rotator = torch.exp(1j * phase_shift_rad)
        
        return complex_spec * rotator

    def synthesize_cond2_shared(self, z, w_ceil, g, bunsetsu_phonemes):
        """
        Cond 2: 潜在変数からの生成 (Shared Encoding / Bunsetsu Decoding)
        Args:
            z (Tensor): [B, C, T_frame]
            w_ceil (Tensor): [B, 1, T_phoneme] or [T_phoneme]
            g (Tensor): [B, C, 1]
            bunsetsu_phonemes (List[str]): 文節リスト
        """
        # --- 1. Tensor型とデバイスの保証 ---
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).to(self.device)
        if isinstance(w_ceil, np.ndarray):
            w_ceil = torch.from_numpy(w_ceil).to(self.device)
        if isinstance(g, np.ndarray):
            g = torch.from_numpy(g).to(self.device)
        
        # w_ceil の平坦化
        w_ceil_flat = w_ceil.squeeze()

        # 文節ごとの音素数計算
        chunk_counts = []
        for ph in bunsetsu_phonemes:
            if not ph: continue
            ids = self._get_text_from_phonemes(ph)
            chunk_counts.append(len(ids))

        full_complex_spec = None
        current_ph_idx = 0
        current_z_frame = 0
        
        # --- 2. チャンクごとのデコードと結合 ---
        with torch.no_grad():
            for count in chunk_counts:
                durations = w_ceil_flat[current_ph_idx : current_ph_idx + count]
                if len(durations) == 0: continue

                z_len = int(torch.sum(durations).item())
                z_end_frame = current_z_frame + z_len
                if z_end_frame > z.shape[2]: 
                    z_end_frame = z.shape[2]
                
                z_chunk = z[:, :, current_z_frame : z_end_frame]
                
                if z_chunk.shape[2] > 0:
                    # デコーダー実行 (spec, phaseの取得)
                    ret = self.model.dec(z_chunk, g=g)
                    
                    # 戻り値の解析 (wav, ..., spec, phase)
                    if isinstance(ret, tuple):
                        spec = ret[-2]
                        phase = ret[-1]
                    else:
                        raise RuntimeError("Decoder did not return spec/phase. This method requires MB-iSTFT-VITS decoder.")

                    complex_chunk = spec * torch.exp(1j * phase)
                    
                    if full_complex_spec is None:
                        full_complex_spec = complex_chunk
                    else:
                        full_complex_spec = torch.cat([full_complex_spec, complex_chunk], dim=-1)
                
                current_ph_idx += count
                current_z_frame = z_end_frame
                if current_z_frame >= z.shape[2]: break

            if full_complex_spec is None: return np.array([])
            
            # --- 3. iSTFTで波形に戻す ---
            audio = self._istft_finalize(full_complex_spec)
            return audio
        

    def synthesize_cond3_shared(self, z, w_ceil, g, bunsetsu_phonemes, z_overlap_frames=10, max_shift_samples=300):
        """
        Cond 3 Fix: Dynamic Overlap Calculation (Time-based)
        ホップ長に依存せず、常に一定時間(0.15秒)のオーバーラップを確保して安定化させます。
        """
        if isinstance(z, np.ndarray): z = torch.from_numpy(z).to(self.device)
        if isinstance(w_ceil, np.ndarray): w_ceil = torch.from_numpy(w_ceil).to(self.device)
        if isinstance(g, np.ndarray): g = torch.from_numpy(g).to(self.device)

        w_ceil_flat = w_ceil.squeeze()
        chunk_counts = []
        for ph in bunsetsu_phonemes:
            if not ph: continue
            ids = self._get_text_from_phonemes(ph)
            chunk_counts.append(len(ids))

        full_audio = None
        prev_tail_overlap = None
        current_ph_idx = 0
        current_z_frame = 0
        
        # --- 時間ベースの設定 ---
        hop_length = getattr(self.model.dec, 'gen_istft_hop_size', self.hps.data.hop_length)
        sampling_rate = self.hps.data.sampling_rate
        
        # 1. オーバーラップ時間: 0.15秒 (サンプル数換算)
        OVERLAP_SEC = 0.15
        expected_overlap_samples = int(sampling_rate * OVERLAP_SEC)
        
        # 2. 必要なフレーム数 (z_overlap_frames) を逆算
        # 安全マージンとして +2 フレーム
        req_overlap_frames = int(expected_overlap_samples / hop_length) + 2
        
        # 3. プリロール時間: 0.05秒
        PREROLL_SEC = 0.05
        preroll_frames = int((sampling_rate * PREROLL_SEC) / hop_length) + 2

        with torch.no_grad():
            for count in chunk_counts:
                durations = w_ceil_flat[current_ph_idx : current_ph_idx + count]
                if len(durations) == 0: continue

                chunk_z_len = int(torch.sum(durations).item())
                
                # --- Pre-roll Logic ---
                z_start_nominal = current_z_frame
                z_start_actual = max(0, current_z_frame - preroll_frames)
                preroll_offset_frames = z_start_nominal - z_start_actual
                
                # 終了位置: 動的に計算した req_overlap_frames を使用
                z_end_nominal = current_z_frame + chunk_z_len
                z_end_decode = z_end_nominal + req_overlap_frames
                if z_end_decode > z.shape[2]: z_end_decode = z.shape[2]
                
                # デコード
                z_chunk = z[:, :, z_start_actual : z_end_decode]
                
                if z_chunk.shape[2] > 0:
                    ret = self.model.dec(z_chunk, g=g)
                    if isinstance(ret, tuple): spec, phase = ret[-2], ret[-1]
                    else: raise RuntimeError("Decoder issue")
                    
                    complex_chunk = spec * torch.exp(1j * phase)
                    chunk_wav = self._istft_finalize(complex_chunk) 
                    
                    # プリロールカット
                    trim_samples = preroll_offset_frames * hop_length
                    if trim_samples < len(chunk_wav):
                        valid_chunk_wav = chunk_wav[trim_samples:]
                    else:
                        valid_chunk_wav = chunk_wav 
                    
                    if full_audio is None:
                        full_audio = valid_chunk_wav
                    else:
                        # 比較対象の抽出
                        if prev_tail_overlap is None: prev_ref = full_audio[-expected_overlap_samples:]
                        else: prev_ref = prev_tail_overlap
                        curr_ref = valid_chunk_wav[:expected_overlap_samples]
                        
                        valid_overlap_len = min(len(prev_ref), len(curr_ref))
                        print("overlaplen "valid_overlap_len)
                        # --- 安全チェック: オーバーラップが短すぎる場合は位置合わせをスキップ ---
                        if valid_overlap_len > 1024: 
                            # ラグ検出
                            delay = self._find_best_time_delay(
                                prev_ref[-valid_overlap_len:], 
                                curr_ref[:valid_overlap_len], 
                                max_shift_samples
                            )
                            
                            aligned_wav = valid_chunk_wav
                            
                            # 位置合わせ
                            print("delay "delay)
                            if delay > 0:
                                if delay < len(aligned_wav):
                                    aligned_wav = aligned_wav[delay:]
                                else: aligned_wav = np.array([])
                            elif delay < 0:
                                cut_from_prev = -delay
                                if cut_from_prev < len(full_audio):
                                    full_audio = full_audio[:-cut_from_prev]
                                else: full_audio = np.array([])
                            
                            # 安全なクロスフェード長の計算
                            xfade_len = min(valid_overlap_len, len(aligned_wav), len(full_audio), 512)
                            
                            if xfade_len > 0:
                                fade_out = full_audio[-xfade_len:]
                                fade_in = aligned_wav[:xfade_len]
                                alpha = np.linspace(0, 1, xfade_len)
                                blended = fade_out * (1 - alpha) + fade_in * alpha
                                
                                full_audio = np.concatenate([
                                    full_audio[:-xfade_len],
                                    blended,
                                    aligned_wav[xfade_len:]
                                ])
                            else:
                                full_audio = np.concatenate([full_audio, aligned_wav])
                        else:
                            # オーバーラップ不足時は単純結合 (安全策)
                            print("passed")
                            full_audio = np.concatenate([full_audio, valid_chunk_wav])

                    prev_tail_overlap = valid_chunk_wav[-expected_overlap_samples:] if len(valid_chunk_wav) > expected_overlap_samples else valid_chunk_wav

                current_ph_idx += count
                current_z_frame = z_end_nominal 
                if current_z_frame >= z.shape[2]: break

            if full_audio is None: return np.array([])
            return full_audio
        
    def synthesize_cond4_shared(self, z, g):
        """
        Cond 4: 共有された z をそのまま一括デコード (Topline / Full Decode)
        Args:
            z (Tensor): [B, C, T_frame]
            g (Tensor): [B, C, 1]
        """
        # Tensor型とデバイスの保証
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).to(self.device)
        if isinstance(g, np.ndarray):
            g = torch.from_numpy(g).to(self.device)

        with torch.no_grad():
            # 一括デコード
            ret = self.model.dec(z, g=g)
            
            if isinstance(ret, tuple):
                spec = ret[-2]
                phase = ret[-1]
            else:
                 raise RuntimeError("Decoder did not return spec/phase (Cond4 requires it).")

            full_complex_spec = spec * torch.exp(1j * phase)
            
        return self._istft_finalize(full_complex_spec)



    def verify_delay_estimation(self):
        """
        遅延推定ロジックの精度を検証するテスト関数
        """
        print("=== Delay Estimation Verification ===")
        
        # 1. テスト信号作成 (ランダムノイズ)
        np.random.seed(42)
        sr = 24000
        duration = 0.5 # seconds
        original_signal = np.random.randn(int(sr * duration))
        
        # テストケース
        test_delays = [0, 10, -10, 50, -50, 100, -100, 300, -300]
        
        passed = 0
        
        for true_delay in test_delays:
            # Ref: 元の信号の中央部分
            # Target: Refから true_delay 分ずらしたもの
            
            # 基準位置
            center = len(original_signal) // 2
            segment_len = 1024
            
            ref_seg = original_signal[center : center + segment_len]
            
            # Target作成
            # true_delay > 0: Targetが遅れている (右にシフト) -> Refより前のデータが来る？
            # ここでの定義再確認:
            # find_best_time_delay(ref, tar) -> returns lag
            # lag > 0 means Target is delayed (Target needs shift left)
            
            # Targetを「遅らせる」 = 波形全体を右にシフト
            # Ref[t] と Target[t + delay] が一致する
            # つまり Target[t] = Ref[t - delay]
            
            # 抽出範囲をずらす
            tar_start = center - true_delay
            tar_seg = original_signal[tar_start : tar_start + segment_len]
            
            # ノイズ付加 (実環境シミュレーション)
            tar_seg = tar_seg + 0.01 * np.random.randn(len(tar_seg))
            
            # 推定実行
            estimated_delay = self._find_best_time_delay(ref_seg, tar_seg, max_shift_samples=512)
            
            print(f"True Delay: {true_delay:4d} | Estimated: {estimated_delay:4d} | ", end="")
            
            if estimated_delay == true_delay:
                print("OK")
                passed += 1
            else:
                print("FAIL")

        print(f"Result: {passed}/{len(test_delays)} passed.")
        return passed == len(test_delays)