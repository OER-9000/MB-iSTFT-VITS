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
    
    def _find_best_time_delay(self, ref_complex_spec, target_complex_spec, max_shift_samples=256):
        """
        [ipynbの相関定理に基づく実装]
        1. iSTFTで波形(wav1_tail, wav2_head)に戻す
        2. FFTで周波数領域へ (X1, X2)
        3. xcor = IFFT(X1 * conj(X2))
        4. ピーク検出でラグを特定
        """
        device = ref_complex_spec.device
        n_fft_istft = getattr(self.model.dec, 'gen_istft_n_fft', self.hps.data.filter_length)
        hop_length = getattr(self.model.dec, 'gen_istft_hop_size', self.hps.data.hop_length)
        win_length = n_fft_istft
        
        window = torch.hann_window(win_length).to(device)
        
        # 1. iSTFTで波形抽出
        # (B, C, F, T) -> (B, F, T) -> wav
        if ref_complex_spec.dim() == 4:
            ref_spec = ref_complex_spec.mean(dim=1)
            tar_spec = target_complex_spec.mean(dim=1)
        else:
            ref_spec = ref_complex_spec
            tar_spec = target_complex_spec

        try:
            # 短いセグメントでも動作するようにcenter=False等は状況によるが、デフォルトで実施
            # 比較対象: 前のセグメントの末尾(ref) vs 今のセグメントの先頭(target)
            ref_wav_t = torch.istft(ref_spec, n_fft_istft, hop_length, win_length, window=window)
            tar_wav_t = torch.istft(tar_spec, n_fft_istft, hop_length, win_length, window=window)
        except Exception:
            return 0

        # NumPyへ (Batch=0のみ対応)
        wav1_tail = ref_wav_t[0].detach().cpu().numpy()
        wav2_head = tar_wav_t[0].detach().cpu().numpy()
        
        # DCオフセット除去 (推奨)
        wav1_tail -= np.mean(wav1_tail)
        wav2_head -= np.mean(wav2_head)
        
        # 2. FFT計算 (サイズは波形長より大きく、2のべき乗が望ましい)
        n_samples = max(len(wav1_tail), len(wav2_head))
        fft_size = 1
        while fft_size < n_samples * 2: # 線形畳み込みのため2倍確保
            fft_size *= 2
            
        X1 = np.fft.fft(wav1_tail, fft_size)
        X2 = np.fft.fft(wav2_head, fft_size)
        
        # 3. 相互相関 (相関定理: F^-1 [ X1 * conj(X2) ])
        xcor = np.real(np.fft.ifft(X1 * np.conj(X2)))
        
        # 4. ピーク検出 (マスク適用)
        # xcorのインデックス:
        # 0 ~ max_shift: 正のラグ (wav2が遅れている/wav2を左にずらす必要がある)
        # fft_size - max_shift ~ fft_size - 1: 負のラグ (wav2が進んでいる/wav2を右にずらす)
        
        mask = np.zeros_like(xcor)
        mask[0 : max_shift_samples + 1] = 1.0
        mask[-max_shift_samples : ] = 1.0
        
        # 絶対値ではなく実部ピーク(正の相関)を探すのが一般的
        lagidx = np.argmax(xcor * mask)
        
        # ラグを符号付き整数に変換
        if lagidx > fft_size // 2:
            best_lag = lagidx - fft_size # 負の値 (wav2が進んでいる -> 右シフトが必要?)
            # ここでの定義: IFFT(X1 * conj(X2)) のピークが -d にあるとき、x2(t) = x1(t-d)
            # numpyのfft結果では、ピーク位置が「x2がどれだけ遅れているか」を表す(正なら遅れ)
            # しかしIFFTの定義によっては符号が逆になることがある。
            # 通常 xcor[tau] = sum x1[t] x2[t+tau] の定義だと、x2が左(未来)にあるとピークが正?
            # ipynbの手法に従うと、「値が最大になる位置」がズレ量。
            
            # テスト的に: lagidx が マイナス(末尾)の場合、wav2を「遅らせる」必要がある
            # _apply_phase_shift は「正の値」を受け取ると「左シフト（進める）」処理を行う実装になっている。
            # なので、wav2が進んでいる(負のラグ)なら、wav2を遅らせる(右シフト)必要がある -> delay_samplesは負になるべき
            pass 
        else:
            best_lag = lagidx # 正の値 (wav2が遅れている -> 左シフトが必要)

        return best_lag

    def _apply_phase_shift(self, complex_spec, delay_samples):
        """
        時間領域での遅延 (delay_samples) を周波数領域の位相シフトで補正する
        delay_samples > 0: 波形を「進める」（左シフト）効果 (遅れを解消)
        delay_samples < 0: 波形を「遅らせる」（右シフト）効果
        """
        device = complex_spec.device
        n_freq = complex_spec.shape[-2]
        n_fft = (n_freq - 1) * 2
        
        k = torch.arange(n_freq, device=device).float()
        
        # 位相回転: exp(j * 2*pi * k * d / N)
        # d > 0 のとき、位相を加算することで波形は左にシフトする（進む）
        phase_shift_rad = 2 * np.pi * k * delay_samples / n_fft
        
        phase_shift_rad = phase_shift_rad.view(1, 1, n_freq, 1)
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

    def synthesize_cond2_auto(self, raw_text, sid=0, noise_scale=1.0, noise_scale_w=1.0, length_scale=1.0):
        """
        [Cond 2: 全自動モード]
        テキストを受け取り、内部で文節分割・Latent生成・接続合成を一括で行います。
        """
        self.model.eval()
        sid_tensor = torch.LongTensor([int(sid)]).to(self.device)

        # 1. 文節分割
        bunsetsu_chunks = self._get_bunsetsu_chunks_mecab(raw_text)
        
        # 2. 全体ID列作成
        all_phoneme_ids = []
        for ph in bunsetsu_chunks:
            if not ph: continue
            ids = self._get_text_from_phonemes(ph)
            all_phoneme_ids.extend(ids.tolist())
            
        if not all_phoneme_ids:
            return np.array([])
        
        stn_tst = torch.LongTensor(all_phoneme_ids)
        
        with torch.no_grad():
            x_tst = stn_tst.to(self.device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
            
            # 3. Latent生成 (z, w, g)
            z, w_ceil, g = self._get_z_and_phoneme_durations(
                x_tst, x_tst_lengths, sid_tensor, 
                noise_scale, noise_scale_w, length_scale
            )
            
            # 4. 接続合成
            return self.synthesize_cond2_shared(z, w_ceil, g, bunsetsu_chunks)
        

    def synthesize_cond3_shared(self, z, w_ceil, g, bunsetsu_phonemes, z_overlap_frames=5, max_shift_samples=300):
        if isinstance(z, np.ndarray): z = torch.from_numpy(z).to(self.device)
        if isinstance(w_ceil, np.ndarray): w_ceil = torch.from_numpy(w_ceil).to(self.device)
        if isinstance(g, np.ndarray): g = torch.from_numpy(g).to(self.device)

        w_ceil_flat = w_ceil.squeeze()
        chunk_counts = []
        for ph in bunsetsu_phonemes:
            if not ph: continue
            ids = self._get_text_from_phonemes(ph)
            chunk_counts.append(len(ids))

        full_complex_spec = None
        prev_tail_overlap = None
        
        current_ph_idx = 0
        current_z_frame = 0
        ratio = None 

        with torch.no_grad():
            for count in chunk_counts:
                durations = w_ceil_flat[current_ph_idx : current_ph_idx + count]
                if len(durations) == 0: continue

                chunk_z_len = int(torch.sum(durations).item())
                
                z_end_nominal = current_z_frame + chunk_z_len
                z_end_decode = z_end_nominal + z_overlap_frames
                if z_end_decode > z.shape[2]: 
                    z_end_decode = z.shape[2]
                
                z_chunk = z[:, :, current_z_frame : z_end_decode]
                
                if z_chunk.shape[2] > 0:
                    ret = self.model.dec(z_chunk, g=g)
                    if isinstance(ret, tuple):
                        spec, phase = ret[-2], ret[-1]
                    else:
                        raise RuntimeError("Decoder did not return spec/phase.")
                    
                    complex_chunk = spec * torch.exp(1j * phase)
                    
                    if ratio is None:
                        ratio = complex_chunk.shape[-1] / z_chunk.shape[-1] if z_chunk.shape[-1] > 0 else 1.0

                    actual_z_overlap = max(0, z_chunk.shape[-1] - chunk_z_len)
                    spec_overlap_len = int(actual_z_overlap * ratio)
                    
                    if full_complex_spec is None:
                        full_complex_spec = complex_chunk
                    else:
                        if prev_tail_overlap is None: 
                            prev_ref = full_complex_spec[..., -spec_overlap_len:]
                        else:
                            prev_ref = prev_tail_overlap
                        
                        curr_ref = complex_chunk[..., :spec_overlap_len]
                        valid_overlap = min(prev_ref.shape[-1], curr_ref.shape[-1])
                        
                        if valid_overlap > 2:
                            # 1. FFT相互相関による遅延検出 (Cond3)
                            delay_samples = self._find_best_time_delay(
                                prev_ref[..., :valid_overlap], 
                                curr_ref[..., :valid_overlap], 
                                max_shift_samples
                            )
                            
                            # 2. 位相シフト補正
                            if delay_samples != 0:
                                complex_chunk = self._apply_phase_shift(complex_chunk, delay_samples)
                                curr_ref = complex_chunk[..., :spec_overlap_len]

                            # 3. OLA
                            cross_len = min(valid_overlap, complex_chunk.shape[-1])
                            if cross_len > 0:
                                alpha = torch.linspace(0.0, 1.0, cross_len).to(self.device).view(1, 1, 1, cross_len)
                                merged = prev_ref[..., :cross_len] * (1 - alpha) + curr_ref[..., :cross_len] * alpha
                                full_complex_spec = torch.cat([
                                    full_complex_spec[..., :-cross_len],
                                    merged,
                                    complex_chunk[..., cross_len:]
                                ], dim=-1)
                            else:
                                full_complex_spec = torch.cat([full_complex_spec, complex_chunk], dim=-1)
                        else:
                            full_complex_spec = torch.cat([full_complex_spec, complex_chunk], dim=-1)
                    
                    next_overlap_spec_len = int(z_overlap_frames * ratio)
                    if complex_chunk.shape[-1] >= next_overlap_spec_len:
                        prev_tail_overlap = complex_chunk[..., -next_overlap_spec_len:]
                    else:
                        prev_tail_overlap = complex_chunk

                current_ph_idx += count
                current_z_frame = z_end_nominal 
                if current_z_frame >= z.shape[2]: break

            if full_complex_spec is None: return np.array([])
            return self._istft_finalize(full_complex_spec)
        
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

    