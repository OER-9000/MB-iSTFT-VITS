
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

class TTS_VITS(RemdisModule):
    def __init__(self, 
                 pub_exchanges=['tts'],
                 sub_exchanges=['dialogue']):
        super().__init__(pub_exchanges=pub_exchanges,
                         sub_exchanges=sub_exchanges)
        
        print("[DEBUG] TTS_VITS module: Initializing...")

        # Load common TTS settings
        self.rate = self.config['TTS']['sample_rate']
        self.frame_length = self.config['TTS']['frame_length']
        self.send_interval = self.config['TTS']['send_interval']
        self.chunk_size = round(self.frame_length * self.rate)

        # Load VITS specific settings
        vits_config_path = self.config['TTS_VITS']['config']
        vits_model_path = self.config['TTS_VITS']['model']
        self.vits_speakerid = self.config['TTS_VITS']['sid']

        # Initialize buffers
        self.input_iu_buffer = queue.Queue()
        self.output_iu_buffer = queue.Queue()
        self.is_revoked = False
        self._is_running = True

        # Initialize VITS model
        print("[DEBUG] TTS_VITS module: Initializing VITS model...")
        self.hps = utils.get_hparams_from_file(vits_config_path)

        self.phonemizer = Phonemizer()

        # Build model
        self.device = device
        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model).to(self.device).eval()

        # Load checkpoint
        utils.load_checkpoint(vits_model_path, self.net_g, None)
        print("[DEBUG] TTS_VITS module: VITS model initialized.")

        print("[DEBUG] TTS_VITS module: Initialization complete.")

    def japanese_cleaner_revised(self, text):
        parts = re.split(r'({cough}|<cough>|\[.*?\]|[、。])', text)
        phoneme_parts = []
        for part in parts:
            if not part or part.isspace(): continue
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

    def run(self):
        print("[DEBUG] TTS_VITS module: run() called.")
        t1 = threading.Thread(target=self.listen_loop)
        t2 = threading.Thread(target=self.synthesis_loop)
        t3 = threading.Thread(target=self.send_loop)
        print("[DEBUG] TTS_VITS module: Starting threads...")
        t1.start()
        t2.start()
        t3.start()
        print("[DEBUG] TTS_VITS module: All threads started.")
        t1.join()
        t2.join()
        t3.join()
        print("[DEBUG] TTS_VITS module: All threads joined.")

    def listen_loop(self):
        print("[DEBUG] TTS_VITS module: listen_loop started.")
        self.subscribe('dialogue', self.callback)

    def send_loop(self):
        print("[DEBUG] TTS_VITS module: send_loop started.")
        
        connection = None
        try:
            # Create a thread-local connection for publishing
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.config['RabbitMQ']['host']))
            channel = connection.channel()
            channel.exchange_declare(exchange='tts', exchange_type='fanout')
            print("[DEBUG] TTS_VITS send_loop: Connection and channel established.")

            while self._is_running:
                if self.is_revoked:
                    self.output_iu_buffer = queue.Queue()
                    # Create and send a commit IU immediately on revoke
                    commit_iu = self.createIU('', 'tts', RemdisUpdateType.COMMIT)
                    commit_iu['data_type'] = 'audio'
                    channel.basic_publish(
                        exchange='tts',
                        routing_key='',
                        body=json.dumps(commit_iu, ensure_ascii=False)
                    )
                    self.is_revoked = False # Reset flag
                    continue

                try:
                    snd_iu = self.output_iu_buffer.get(block=True, timeout=1)
                    channel.basic_publish(
                        exchange='tts',
                        routing_key='',
                        body=json.dumps(snd_iu, ensure_ascii=False)
                    )
                    time.sleep(self.send_interval)
                except queue.Empty:
                    continue
        except Exception as e:
            print(f"[ERROR] TTS_VITS send_loop: An exception occurred: {e}")
            traceback.print_exc()
        finally:
            if connection and connection.is_open:
                connection.close()
                print("[DEBUG] TTS_VITS send_loop: Connection closed.")

    def synthesis_loop(self):
        print("[DEBUG] TTS_VITS module: synthesis_loop started.")
        while self._is_running:
            if self.is_revoked:
                self.input_iu_buffer = queue.Queue()
                self.is_revoked = False # Reset flag

            try:
                in_msg = self.input_iu_buffer.get(block=True, timeout=1)
                try:
                    output_text = in_msg['body']
                    update_type = in_msg['update_type']

                    if output_text != '':
                        # Synthesize audio with VITS
                        print("[DEBUG] Synthesis - Step 1: Cleaning text")
                        phonemized_text = self.japanese_cleaner_revised(output_text)
                        
                        print("[DEBUG] Synthesis - Step 2: Converting text to sequence")
                        stn_tst = cleaned_text_to_sequence(phonemized_text)

                        if self.hps.data.add_blank:
                            print("[DEBUG] Synthesis - Step 3: Intersperse")
                            stn_tst = commons.intersperse(stn_tst, 0)
                        
                        stn_tst = torch.LongTensor(stn_tst)
                        
                        # VITS推論実行
                        with torch.no_grad():
                            x_tst = stn_tst.unsqueeze(0).to(self.device)
                            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
                            sid = torch.LongTensor([self.vits_speakerid]).to(self.device)
                            
                            # 推論 (noise_scaleはデフォルトのままでOKですが、ノイズが多い場合は0.333などに下げてみてください)
                            x, sr = self.net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=1.0, length_scale=1.0)[0][0,0].data.cpu().float().numpy(), self.hps.data.sampling_rate

                        # --- 【修正ポイント】音声後処理 ---
                        
                        # 1. リサンプリング (float32であることを明示)
                        if sr != self.rate:
                            x = librosa.resample(x.astype(numpy.float32), orig_sr=sr, target_sr=self.rate)
                        
                        # 2. 音量正規化 (条件付き & マージン確保)
                        # 生成された音声が無音に近い場合は増幅しない (閾値 0.01 は環境に合わせて調整)
                        max_abs_val = numpy.abs(x).max()
                        if self.config['TTS_VITS'].get('auto_normalize', True) and max_abs_val > 0.01:
                            # 1.0 ではなく 0.9 を掛けてクリッピングの余裕を持たせる
                            x = (x / max_abs_val) * 0.9 
                            print(f"[DEBUG] Normalized with max_val: {max_abs_val:.4f}")
                        else:
                            # 音量が小さすぎる、または正規化オフの場合はそのまま（ただしクリップはする）
                            pass

                        # 3. クリッピング (【最重要】オーバーフロー防止)
                        # 値を確実に -1.0 ~ 1.0 の範囲に収める
                        x = numpy.clip(x, -1.0, 1.0)

                        # 4. Int16変換 & チャンク送信
                        # ここでの乗算結果が32767を超えないように上記clipが必須
                        audio_int16 = (x * 32767).astype(numpy.int16)

                        t = 0
                        while t < len(audio_int16):
                            chunk = audio_int16[t:t+self.chunk_size]
                            # バイト列への変換
                            chunk_b64 = base64.b64encode(chunk.tobytes()).decode('utf-8')
                            
                            snd_iu = self.createIU(chunk_b64, 'tts', update_type)
                            snd_iu['data_type'] = 'audio'
                            self.output_iu_buffer.put(snd_iu)
                            t += self.chunk_size
                        
                    self.send_commitIU('tts')
                except Exception as e:
                    print(f"[ERROR] TTS_VITS synthesis_loop: An error occurred: {e}")
                    traceback.print_exc()

            except queue.Empty:
                continue

    def send_commitIU(self, channel):
        snd_iu = self.createIU('', channel, RemdisUpdateType.COMMIT)
        snd_iu['data_type'] = 'audio'
        self.output_iu_buffer.put(snd_iu) # Put commit in queue for send_loop
        self.printIU(snd_iu)

    def callback(self, ch, method, properties, in_msg):
        print("[DEBUG] TTS_VITS module: Message received.")
        in_msg = self.parse_msg(in_msg)
        self.printIU(in_msg)
        
        if in_msg['update_type'] == RemdisUpdateType.REVOKE:
            print("[DEBUG] TTS_VITS module: REVOKE received.")
            self.is_revoked = True
        else:
            self.input_iu_buffer.put(in_msg)

def main():
    tts_vits = TTS_VITS()
    try:
        tts_vits.run()
    except KeyboardInterrupt:
        print("Stopping TTS_VITS module...")
        tts_vits._is_running = False
        # Wait for threads to finish
        time.sleep(2)
        print("TTS_VITS module stopped.")

if __name__ == '__main__':
    main()
