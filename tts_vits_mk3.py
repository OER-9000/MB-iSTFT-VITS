
import sys, os
import numpy
import queue
import time
import threading
import base64
import librosa
import torch
import re
import importlib
import traceback
import pika
import json

# VITS specific imports
# Assuming MS_iSTFT_VITS is in the python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MS_iSTFT_VITS'))
import commons
import utils
from models import SynthesizerTrn
from text_JP import cleaned_text_to_sequence, symbols
from text_JP.phonemize import Phonemizer
import pyopenjtalk

from base import RemdisModule, RemdisUpdateType

device = torch.device("cpu")


class TTS(RemdisModule):
    def __init__(self, 
                 pub_exchanges=['tts'],
                 sub_exchanges=['dialogue']):
        super().__init__(pub_exchanges=pub_exchanges,
                         sub_exchanges=sub_exchanges)
        
        # Remdis TTS Config
        self.rate = self.config['TTS'].get('sample_rate', 16000)
        self.frame_length = self.config['TTS'].get('frame_length', 0.02)
        self.send_interval = self.config['TTS'].get('send_interval', 0.1)
        # self.sample_width = self.config['TTS']['sample_width'] # 今回はfloat->int16変換を行うため固定で扱います
        self.chunk_size = round(self.frame_length * self.rate)

        # VITS Specific Config
        # configファイルに未記載の場合は、デフォルトパスを使用します
        self.vits_config_path = self.config['TTS'].get('vits_config_path', "./logs/uudb_csj21/config.json")
        self.vits_checkpoint_path = self.config['TTS'].get('vits_checkpoint_path', "./logs/uudb_csj21/G_1030000.pth")
        self.speaker_id = self.config['TTS'].get('speaker_id', 375)

        self.input_iu_buffer = queue.Queue()
        self.output_iu_buffer = queue.Queue()
        
        self.is_revoked = False
        self._is_running = True

        # Device configuration
        # CPUで実行する場合は "cpu"、GPUがあるなら "cuda"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"TTS Device: {self.device}")

        # Initialize Phonemizer and Model
        self.phonemizer = Phonemizer()
        self._init_model()

    def _init_model(self):
        if not os.path.exists(self.vits_config_path):
            print(f"ERROR: Config file not found at {self.vits_config_path}")
            return
        if not os.path.exists(self.vits_checkpoint_path):
            print(f"ERROR: Checkpoint file not found at {self.vits_checkpoint_path}")
            return

        print("Loading VITS model...")
        self.hps = utils.get_hparams_from_file(self.vits_config_path)

        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model).to(self.device)
        
        _ = self.net_g.eval()
        utils.load_checkpoint(self.vits_checkpoint_path, self.net_g, None)
        print("VITS model loaded successfully.")

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

            # チャンクの間隔ごとに送信を実行
            time.sleep(self.send_interval)

            # システム発話終端まで送信した場合の処理
            if snd_iu['update_type'] == RemdisUpdateType.COMMIT:
                self.send_commitIU('tts')

    # --- Text Pre-processing Methods ---
    def _japanese_cleaner_revised(self, text):
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

    def _text_to_sequence_custom(self, text):
        phonemized_text = self._japanese_cleaner_revised(text)
        stn_tst = cleaned_text_to_sequence(phonemized_text)
        if self.hps.data.add_blank:
            stn_tst = commons.intersperse(stn_tst, 0)
        return torch.LongTensor(stn_tst)

    def synthesis_loop(self):
        while True:
            if self.is_revoked:
                self.input_iu_buffer = queue.Queue()

            # 入力バッファから受信したIUを取得
            in_msg = self.input_iu_buffer.get(block=True)
            output_text = in_msg['body']
            tgt_id = in_msg['id']
            update_type = in_msg['update_type']

            x = np.array([])
            
            if output_text != '':
                try:
                    # テキストをシーケンスに変換
                    stn_tst = self._text_to_sequence_custom(output_text)
                    
                    with torch.no_grad():
                        x_tst = stn_tst.unsqueeze(0).to(self.device)
                        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
                        sid = torch.LongTensor([self.speaker_id]).to(self.device)
                        
                        # Inference
                        # noise_scale等は必要に応じてconfigから読み込むように変更可能です
                        audio = self.net_g.infer(x_tst, x_tst_lengths, sid=sid, 
                                                noise_scale=0.1, 
                                                noise_scale_w=1.0, 
                                                length_scale=1.0)[0][0,0].data.cpu().float().numpy()
                    
                    x = audio
                    sr = self.hps.data.sampling_rate # VITSモデルのサンプリングレート

                    # リサンプリング (VITSのSR -> RemdisのSR)
                    if sr != self.rate:
                        x = librosa.resample(x.astype(np.float32), 
                                           orig_sr=sr, 
                                           target_sr=self.rate)

                    # Float32 (-1.0 ~ 1.0) -> Int16 (-32768 ~ 32767) 変換
                    x = x * 32767.0
                    x = np.clip(x, -32768, 32767)

                except Exception as e:
                    print(f"Synthesis Error: {e}")
                    x = np.zeros(self.chunk_size)

                # チャンクに分割して出力バッファに格納
                t = 0
                if len(x) == 0:
                     # エラー等で音声がない場合、無音を作る（最低1チャンク）
                     x = np.zeros(self.chunk_size)

                while t < len(x):
                    # 残りがchunk_size未満の場合のパディング処理などは簡易的にスライスで対応
                    chunk_data = x[t:t+self.chunk_size]
                    
                    # 最後のチャンクが短い場合、0埋めするかそのまま送るか。
                    # ここではそのまま送りますが、厳密な固定長が必要ならpadしてください。
                    
                    chunk_b64 = base64.b64encode(chunk_data.astype(np.int16).tobytes()).decode('utf-8')
                    snd_iu = self.createIU(chunk_b64, 'tts', update_type)
                    snd_iu['data_type'] = 'audio'
                    self.output_iu_buffer.put(snd_iu)
                    t += self.chunk_size
            else:
                # テキストがない場合（無音生成）
                x = np.zeros(self.chunk_size)
                chunk_b64 = base64.b64encode(x.astype(np.int16).tobytes()).decode('utf-8')
                snd_iu = self.createIU(chunk_b64, 'tts', update_type)
                snd_iu['data_type'] = 'audio'
                self.output_iu_buffer.put(snd_iu)

    # 発話終了時のメッセージ送信関数
    def send_commitIU(self, channel):
        snd_iu = self.createIU('', channel,
                               RemdisUpdateType.COMMIT)
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