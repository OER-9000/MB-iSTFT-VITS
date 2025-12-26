import sys, os
import numpy
import queue
import re
import time
import threading
import base64
import librosa

import torch
import commons
import utils
from models import SynthesizerTrn
from text_JP import cleaned_text_to_sequence, symbols
import pyopenjtalk
from text_JP.phonemize import Phonemizer

from base import RemdisModule, RemdisUpdateType

device = torch.device("cpu")

class TTS(RemdisModule):
    def __init__(self, 
                 pub_exchanges=['tts'],
                 sub_exchanges=['dialogue']):
        super().__init__(pub_exchanges=pub_exchanges,
                         sub_exchanges=sub_exchanges)
        
        self.rate = self.config['TTS_VITS']['sample_rate']
        self.frame_length = self.config['TTS_VITS']['frame_length']
        self.send_interval = self.config['TTS_VITS']['send_interval']
        self.sample_width = self.config['TTS_VITS']['sample_width']
        self.chunk_size = round(self.frame_length * self.rate)

        self.input_iu_buffer = queue.Queue()
        self.output_iu_buffer = queue.Queue()
        
        # MS-iSTFT-VITS model configuration
        self.config_path = self.config['TTS_VITS']['config_path']
        self.checkpoint_path = self.config['TTS_VITS']['checkpoint_path']
        self.speaker_id = self.config['TTS_VITS'].get('speaker_id', 375)
        
        # Synthesis parameters
        self.noise_scale = self.config['TTS_VITS'].get('noise_scale', 0.1)
        self.noise_scale_w = self.config['TTS_VITS'].get('noise_scale_w', 1.0)
        self.length_scale = self.config['TTS_VITS'].get('length_scale', 1.0)
        
        # Load MS-iSTFT-VITS model
        self._load_model()
        
        # Initialize phonemizer
        self.phonemizer = Phonemizer()
        
        self.is_revoked = False
        self._is_running = True

    def _load_model(self):
        """Load MS-iSTFT-VITS model"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {self.checkpoint_path}")
        
        print("Loading MS-iSTFT-VITS model...")
        self.hps = utils.get_hparams_from_file(self.config_path)
        
        # Initialize model
        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model).to(device)
        
        self.net_g.eval()
        
        # Load checkpoint
        print(f"Loading checkpoint from {self.checkpoint_path}...")
        _ = utils.load_checkpoint(self.checkpoint_path, self.net_g, None)
        print("Model loaded successfully.")

    def japanese_cleaner_revised(self, text):
        """Text preprocessing for Japanese"""
        parts = re.split(r'({cough}|<cough>|\[.*?\]|[、。])', text)
        phoneme_parts = []
        
        for part in parts:
            if not part or part.isspace():
                continue
            if part.startswith('[') and part.endswith(']') and len(part) > 2:
                content = part[1:-1]
                if not content:
                    phoneme_parts.append('[ ]')
                else:
                    kana_content = pyopenjtalk.g2p(content, kana=True).replace('ヲ', 'オ')
                    phoneme_content = self.phonemizer(kana_content)
                    phoneme_parts.append(f'[ {phoneme_content} ]')
                continue
            if part == '{cough}' or part == '<cough>':
                phoneme_parts.append('<cough>')
                continue
            if part in '、。':
                phoneme_parts.append('sp')
                continue
            kana = pyopenjtalk.g2p(part, kana=True).replace('ヲ', 'オ')
            phonemes = self.phonemizer(kana)
            phoneme_parts.append(phonemes)
        
        final_text = ' '.join(phoneme_parts)
        return re.sub(r'\s+', ' ', final_text).strip()

    def text_to_sequence_custom(self, text):
        """Convert text to sequence for model input"""
        phonemized_text = self.japanese_cleaner_revised(text)
        stn_tst = cleaned_text_to_sequence(phonemized_text)
        if self.hps.data.add_blank:
            stn_tst = commons.intersperse(stn_tst, 0)
        return torch.LongTensor(stn_tst)

    def synthesize_audio(self, text):
        """Synthesize audio using MS-iSTFT-VITS"""
        if not text or text == '':
            return numpy.zeros(self.chunk_size), self.rate
        
        try:
            # Convert text to sequence
            stn_tst = self.text_to_sequence_custom(text)
            
            with torch.no_grad():
                x_tst = stn_tst.unsqueeze(0).to(device)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
                sid = torch.LongTensor([self.speaker_id]).to(device)
                
                # Inference
                audio = self.net_g.infer(
                    x_tst, x_tst_lengths, sid=sid,
                    noise_scale=self.noise_scale,
                    noise_scale_w=self.noise_scale_w,
                    length_scale=self.length_scale
                )[0][0, 0].data.cpu().float().numpy()
            
            sr = self.hps.data.sampling_rate
            return audio, sr
            
        except Exception as e:
            print(f"Error in synthesis: {e}")
            return numpy.zeros(self.chunk_size), self.rate

    def run(self):
        # メッセージ受信スレッド
        t1 = threading.Thread(target=self.listen_loop)
        # 音声合成処理スレッド
        t2 = threading.Thread(target=self.synthesis_loop)
        # メッセージ送信スレッド
        t3 = threading.Thread(target=self.send_loop)

        # スレッド実行
        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()

    def listen_loop(self):
        self.subscribe('dialogue', self.callback)

    def send_loop(self):
        # 音声データをチャンクごとに送信
        while True:
            # REVOKEされた場合は送信を停止 (= ユーザ割り込み時の処理)
            if self.is_revoked:
                self.output_iu_buffer = queue.Queue()
                self.send_commitIU('tts')
                
            snd_iu = self.output_iu_buffer.get(block=True)
            self.publish(snd_iu, 'tts')

            # チャンクの間隔ごとに送信を実行(音が切れるので少し早い間隔で送信)
            time.sleep(self.send_interval)

            # システム発話終端まで送信した場合の処理
            if snd_iu['update_type'] == RemdisUpdateType.COMMIT:
                self.send_commitIU('tts')

    def synthesis_loop(self):
        while True:
            if self.is_revoked:
                self.input_iu_buffer = queue.Queue()

            # 入力バッファから受信したIUを取得
            in_msg = self.input_iu_buffer.get(block=True)
            output_text = in_msg['body']
            tgt_id = in_msg['id']
            update_type = in_msg['update_type']

            # MS-iSTFT-VITSで音声合成
            x, sr = self.synthesize_audio(output_text)
            
            # サンプリングレートが異なる場合はリサンプリング
            if sr != self.rate:
                x = librosa.resample(x.astype(numpy.float32),
                                     orig_sr=sr,
                                     target_sr=self.rate)
            
            # チャンクに分割して出力バッファに格納
            if len(x) > 0:
                t = 0
                while t <= len(x):
                    chunk = x[t:t+self.chunk_size]
                    # int16に変換してエンコード
                    chunk = base64.b64encode(chunk.astype(numpy.int16).tobytes()).decode('utf-8')
                    snd_iu = self.createIU(chunk, 'tts', update_type)
                    snd_iu['data_type'] = 'audio'
                    self.output_iu_buffer.put(snd_iu)
                    t += self.chunk_size
            else:
                # テキストがない場合も処理を実施
                x = numpy.zeros(self.chunk_size)
                chunk = base64.b64encode(x.astype(numpy.int16).tobytes()).decode('utf-8')
                snd_iu = self.createIU(chunk, 'tts', update_type)
                snd_iu['data_type'] = 'audio'
                self.output_iu_buffer.put(snd_iu)

    # 発話終了時のメッセージ送信関数
    def send_commitIU(self, channel):
        snd_iu = self.createIU('', channel, RemdisUpdateType.COMMIT)
        snd_iu['data_type'] = 'audio'
        self.printIU(snd_iu)
        self.publish(snd_iu, channel)

    # メッセージ受信用コールバック関数
    def callback(self, ch, method, properties, in_msg):
        in_msg = self.parse_msg(in_msg)
        self.printIU(in_msg)
        
        # システム発話のupdate_typeを監視
        if in_msg['update_type'] == RemdisUpdateType.REVOKE:
            self.is_revoked = True
        else:
            self.input_iu_buffer.put(in_msg)
            self.is_revoked = False

def main():
    tts = TTS()
    tts.run()

if __name__ == '__main__':
    main()