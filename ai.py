# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ai (アイ) - ハイブリッド音声対話エージェント
ローカルのMaAI（VAPモデル）によるリアルタイム相槌・ターンテイキング予測と、
Gemini Multimodal Live APIによる思考・回答生成を統合したシステム。
"""

import argparse
import asyncio
import base64
import collections
from enum import Enum, auto
import io
import math
import os
import queue
import random
import struct
import sys
import threading
import time
import traceback
import wave

# Import MaaiMultiple and input base classes
from maai import MaaiMultiple, MaaiInput

import cv2
import numpy as np
import PIL.Image
import mss
import pyaudio

from google import genai
from google.genai import types

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

# ==============================================================================
# 設定パラメータ
# ==============================================================================

# --- 音声IO設定 ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
BYTES_PER_SAMPLE_INT16 = 2
INT16_MAX = 32767
INT16_MIN = -32768
INT16_SCALE = 32768.0

# --- ターンテイキング（Turn Shift）---
# 発話終了検知（Turn Shift）閾値
THRESHOLD_P_SHIFT = 0.65
# デバウンス: 判定確定に必要な連続フレーム数 (2〜3フレーム: 約40〜60ms)
DEBOUNCE_FRAMES_SHIFT = 2

# --- 相槌（Backchannel） ---
# 相槌機会検知（Backchannel）閾値
THRESHOLD_P_BC = 0.70
# デバウンス: 相槌判定に必要な連続フレーム数 (2〜3フレーム)
DEBOUNCE_FRAMES_BC = 2
# 相槌発動後のクールダウン時間（秒: 1.2〜1.5秒）
BC_COOLDOWN_SEC = 1.3
# 初期デフォルト音量 (RMS)
BC_INITIAL_USER_RMS = 300.0
# 音声未作成時の仮相槌待機時間（秒）
BC_MOCK_DURATION_SEC = 0.35

# --- 相槌の音響変調パラメータ ---
# ピッチ / 速度変調の最大変動幅 (±3%〜5%)
BC_PITCH_VARIATION_RANGE = 0.04
# ユーザー直近音量 (RMS) 追従の平滑化係数 (EMA)
BC_RMS_SMOOTHING = 0.85
# 相槌の最小・最大ゲイン倍率
BC_MIN_GAIN = 0.4
BC_MAX_GAIN = 1.3
# ゲイン基準となるユーザー発話目標RMS値
BC_TARGET_USER_RMS = 600.0

# 相槌WAVファイルの配置ディレクトリ
BACKCHANNEL_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "assets", "backchannels")

# --- MaAI (VAPモデル) 設定 ---
MAAI_SAMPLE_RATE = 16000
MAAI_FRAME_SIZE = 160
MAAI_FRAME_RATE = 5
MAAI_CONTEXT_LEN_SEC = 5
SYSTEM_AUDIO_MAX_QUEUE_SIZE = 5
SYSTEM_AUDIO_POLL_INTERVAL_SEC = 0.002
VAP_USER_SPEAKER_INDEX = 0
VAP_SYSTEM_SPEAKER_INDEX = 1

# --- 映像・キュー設定 ---
OUT_QUEUE_MAX_SIZE = 5
IMAGE_MAX_SIZE = (1024, 1024)
VIDEO_FRAME_INTERVAL_SEC = 1.0

# --- Geminiモデル設定 ---
MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
DEFAULT_MODE = "camera"

# 仕様書に基づくシステムプロンプト
# 文頭に自然な相槌・フィラーを含めて一括返答させる
SYSTEM_INSTRUCTION = """\
あなたはいつもポジティブで明るく元気なAI「Ai（アイ）」です。
友達と話すような親しみやすい口調で、どんな話題でも楽しそうに会話します。
下記のルールを厳格に守ってください。
・返答の文頭には自然な相槌やフィラー（「うん」「そうだね」「あー」「なるほど」など）を必ず含めて一括返答してください。
・文章中に*や-などの記号やマークダウンは一切使用せず、音声読み上げに適したプレーンな日本語にしてください。
・簡潔に1〜2文程度でテンポよく返答してください。
"""

# Live session configuration
CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    thinking_config=types.ThinkingConfig(
        thinking_budget=0,
        include_thoughts=False,
    ),
    system_instruction=types.Content(
        parts=[types.Part.from_text(text=SYSTEM_INSTRUCTION)]
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
    enable_affective_dialog=True,
    proactivity=types.ProactivityConfig(
        proactive_audio=True,
    ),
)

pya = pyaudio.PyAudio()

# ==============================================================================
# ステートマシン定義
# ==============================================================================

class DialogueState(Enum):
    STATE_LISTENING = auto()            # 通常傾聴状態
    STATE_PROCESSING_GEMINI = auto()    # Geminiへ投機的発注＆回答処理・再生中


class DialogueStateMachine:
    """
    - P_shift 優先度制御
    - デバウンス判定
    - 相槌クールダウン
    - 相槌と発話終了の競合（連動処理）
    """
    def __init__(self, on_shift_callback, on_bc_callback):
        self.state = DialogueState.STATE_LISTENING
        self.on_shift_callback = on_shift_callback
        self.on_bc_callback = on_bc_callback

        self.last_bc_time = 0.0
        self.shift_debounce_count = 0
        self.bc_debounce_count = 0

    def update_predictions(self, p_system_turn: float, p_bc: float, bc_category: str = "reactive"):
        now = time.time()

        # デバウンスのカウント処理
        if p_system_turn > THRESHOLD_P_SHIFT:
            self.shift_debounce_count += 1
        else:
            self.shift_debounce_count = 0

        if p_bc > THRESHOLD_P_BC:
            self.bc_debounce_count += 1
        else:
            self.bc_debounce_count = 0

        # 1. 優先度制御 (p_system_turn 優先)
        if self.shift_debounce_count >= DEBOUNCE_FRAMES_SHIFT:
            self.shift_debounce_count = 0
            if self.state == DialogueState.STATE_LISTENING:
                print(f"[!] [StateMachine] p_system_turn確定 ({p_system_turn:.2f}) -> STATE_PROCESSING_GEMINI に遷移")
                self.state = DialogueState.STATE_PROCESSING_GEMINI
                # Geminiへの投機的発話要求
                asyncio.create_task(self.on_shift_callback())
                return

        # 2. 相槌判定 (LISTENING かつ クールダウン経過後)
        if self.state == DialogueState.STATE_LISTENING:
            if self.bc_debounce_count >= DEBOUNCE_FRAMES_BC:
                self.bc_debounce_count = 0
                time_since_last_bc = now - self.last_bc_time
                if time_since_last_bc >= BC_COOLDOWN_SEC:
                    self.last_bc_time = now
                    print(f"[*] [StateMachine] P_bc確定 ({p_bc:.2f}, cat={bc_category}) -> 相槌再生トリガー")
                    asyncio.create_task(self.on_bc_callback(bc_category))
                else:
                    # クールダウン中
                    pass

    def on_gemini_completed(self):
        """Geminiの回答終了により通常傾聴状態へ復帰"""
        if self.state != DialogueState.STATE_LISTENING:
            print("[i] [StateMachine] Gemini回答完了 -> STATE_LISTENING に復帰")
            self.state = DialogueState.STATE_LISTENING
            self.shift_debounce_count = 0
            self.bc_debounce_count = 0


# ==============================================================================
# 相槌（Backchannel）生成・再生マネージャ
# ==============================================================================

class BackchannelManager:
    """
    相槌再生エンジン:
    - WAV音声ファイル
    - ピッチ/話速のランダム変調
    - ユーザー直近入力音量 (RMS) への音量リアルタイム追従 (Gain)
    - カテゴリ分類 (reactive, emotional, thoughtful)
    """
    def __init__(self, audio_dir=BACKCHANNEL_AUDIO_DIR):
        self.audio_dir = audio_dir
        self.current_user_rms = BC_INITIAL_USER_RMS
        self.category_samples = {
            "reactive": ["うん", "はい", "そう"],
            "emotional": ["へぇー", "あー", "うわ"],
            "thoughtful": ["なるほど", "ふむ"],
        }
        self.output_stream = None

    def update_user_rms(self, pcm_data: bytes):
        """マイク入力PCMからRMS（音量）を計算し平滑化更新"""
        if not pcm_data:
            return
        count = len(pcm_data) // BYTES_PER_SAMPLE_INT16
        if count == 0:
            return
        shorts = struct.unpack(f"{count}h", pcm_data)
        sum_squares = sum(s * s for s in shorts)
        rms = math.sqrt(sum_squares / count)
        # 指数移動平均 (EMA) で平滑化
        self.current_user_rms = (
            BC_RMS_SMOOTHING * self.current_user_rms + (1.0 - BC_RMS_SMOOTHING) * rms
        )

    def calculate_gain(self) -> float:
        """ユーザーのRMSに基づいて再生ゲインを算出"""
        ratio = self.current_user_rms / max(BC_TARGET_USER_RMS, 1.0)
        gain = max(BC_MIN_GAIN, min(BC_MAX_GAIN, ratio))
        return gain

    async def play_backchannel(self, category: str = "reactive", on_audio_chunk=None):
        """相槌の再生処理（WAVがあれば変調再生、未作成時は仮実装としてログとダミー待機）"""
        # カテゴリに応じたフレーズ選定
        phrases = self.category_samples.get(category, self.category_samples["reactive"])
        phrase = random.choice(phrases)
        gain = self.calculate_gain()
        # ピッチ/話速のランダム変動
        pitch_factor = 1.0 + random.uniform(-BC_PITCH_VARIATION_RANGE, BC_PITCH_VARIATION_RANGE)

        # WAVファイルを探す
        category_dir = os.path.join(self.audio_dir, category)
        wav_file = None
        if os.path.exists(category_dir):
            files = [f for f in os.listdir(category_dir) if f.endswith(".wav")]
            if files:
                wav_file = os.path.join(category_dir, random.choice(files))

        if wav_file and os.path.exists(wav_file):
            print(f"[>] [Backchannel] WAV再生: {os.path.basename(wav_file)} (gain={gain:.2f}, pitch_factor={pitch_factor:.3f})")
            await asyncio.to_thread(self._play_wav_with_modulation, wav_file, gain, pitch_factor, on_audio_chunk)
        else:
            # 音声ファイル未作成時の仮実装 (Mock再生)
            print(f"[>] [Backchannel (Mock)] 相槌発声: 『{phrase}』 (cat={category}, gain={gain:.2f}, pitch={pitch_factor:.3f})")
            # 人間の相槌長をシミュレート
            await asyncio.sleep(BC_MOCK_DURATION_SEC)

    def _play_wav_with_modulation(self, wav_path: str, gain: float, pitch_factor: float, on_audio_chunk=None):
        """WAVファイルをゲイン適用およびサンプルレート変調で再生"""
        try:
            with wave.open(wav_path, "rb") as wf:
                sample_rate = int(wf.getframerate() * pitch_factor)
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()

                stream = pya.open(
                    format=pya.get_format_from_width(sampwidth),
                    channels=channels,
                    rate=sample_rate,
                    output=True,
                )
                chunk = CHUNK_SIZE
                data = wf.readframes(chunk)
                while data:
                    if gain != 1.0 and sampwidth == BYTES_PER_SAMPLE_INT16:
                        count = len(data) // BYTES_PER_SAMPLE_INT16
                        shorts = struct.unpack(f"{count}h", data)
                        modulated = [int(max(INT16_MIN, min(INT16_MAX, s * gain))) for s in shorts]
                        data = struct.pack(f"{count}h", *modulated)
                    if on_audio_chunk:
                        on_audio_chunk(data, in_sample_rate=sample_rate)
                    stream.write(data)
                    data = wf.readframes(chunk)

                stream.stop_stream()
                stream.close()
        except Exception as e:
            print(f"[WARN] [Backchannel] WAV再生エラー: {e}")


# ==============================================================================
# MaAI 2ch 音声入力ソース (UserAudioInput & SystemAudioInput)
# ==============================================================================

class UserAudioInput(MaaiInput.Base):
    """
    MaAI 2ch (Channel 1) 用のユーザー音声入力ソース。
    PyAudioのlisten_audioストリームから読み込んだPCMデータ（16kHz, int16）を受け取り、
    16kHz float32に変換してMaAIへ160サンプル（10ms）単位でリアルタイム供給する。
    これにより、PyAudioのマイクオープンを1本に統一し、デバイス競合・音飛びを解消する。
    """
    def __init__(
        self,
        sample_rate: int = MAAI_SAMPLE_RATE,
        frame_size: int = MAAI_FRAME_SIZE,
    ):
        super().__init__()
        self.sampling_rate = sample_rate
        self.frame_size = frame_size
        self._audio_buffer = collections.deque()
        self._buffer_lock = threading.Lock()

    def put_audio_pcm(self, pcm_bytes: bytes):
        if not pcm_bytes:
            return
        count = len(pcm_bytes) // BYTES_PER_SAMPLE_INT16
        if count == 0:
            return
        floats = (np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / INT16_SCALE).tolist()
        with self._buffer_lock:
            self._audio_buffer.extend(floats)
            while len(self._audio_buffer) >= self.frame_size:
                frame = [self._audio_buffer.popleft() for _ in range(self.frame_size)]
                self._put_to_all_queues(frame)

    def start(self):
        self._is_thread_started = True

    def stop(self):
        pass


class SystemAudioInput(MaaiInput.Base):
    """
    MaAI 2ch (Channel 2) 用のシステム音声入力ソース。
    AI発話中 (Gemini回答や相槌) はその音声をリサンプリング (16kHz float32) して供給し、
    非発声時（アイドル時）はゼロ配列（無音データ）をリアルタイムに供給する。
    """
    def __init__(
        self,
        sample_rate: int = MAAI_SAMPLE_RATE,
        frame_size: int = MAAI_FRAME_SIZE,
        max_queue_size: int = SYSTEM_AUDIO_MAX_QUEUE_SIZE,
    ):
        super().__init__()
        self.sampling_rate = sample_rate
        self.frame_size = frame_size
        self.max_queue_size = max_queue_size
        self._audio_buffer = collections.deque()
        self._buffer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread = None

    def put_audio_pcm(self, pcm_bytes: bytes, in_sample_rate: int = RECEIVE_SAMPLE_RATE):
        """システム音声のPCMバイト列を受信し、16kHz float32に変換してバッファに追加"""
        if not pcm_bytes:
            return
        count = len(pcm_bytes) // BYTES_PER_SAMPLE_INT16
        if count == 0:
            return
        shorts = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / INT16_SCALE
        if in_sample_rate != self.sampling_rate and count > 1:
            # 24kHz -> 16kHz リサンプリング
            num_target = int(round(count * self.sampling_rate / in_sample_rate))
            if num_target > 0:
                resampled = np.interp(
                    np.linspace(0, count, num_target, endpoint=False),
                    np.arange(count),
                    shorts,
                ).astype(np.float32)
            else:
                resampled = np.empty(0, dtype=np.float32)
        else:
            resampled = shorts

        with self._buffer_lock:
            self._audio_buffer.extend(resampled.tolist())

    def clear_buffer(self):
        """発話完了時や割り込み時にシステム音声バッファおよびキューをクリア"""
        with self._buffer_lock:
            self._audio_buffer.clear()
        with self._lock:
            for q in self._subscriber_queues:
                try:
                    while not q.empty():
                        q.get_nowait()
                except Exception:
                    pass

    def _process_loop(self):
        while not self._stop_event.is_set():
            try:
                if self._get_queue_size() >= self.max_queue_size:
                    time.sleep(SYSTEM_AUDIO_POLL_INTERVAL_SEC)
                    continue

                with self._buffer_lock:
                    buf_len = len(self._audio_buffer)
                    if buf_len >= self.frame_size:
                        frame = [self._audio_buffer.popleft() for _ in range(self.frame_size)]
                    elif buf_len > 0:
                        frame = [self._audio_buffer.popleft() for _ in range(buf_len)]
                        frame.extend([0.0] * (self.frame_size - buf_len))
                    else:
                        frame = [0.0] * self.frame_size

                self._put_to_all_queues(frame)
            except Exception as e:
                time.sleep(SYSTEM_AUDIO_POLL_INTERVAL_SEC)

    def start(self):
        if not self._is_thread_started:
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._process_loop, daemon=True)
            self._worker_thread.start()
            self._is_thread_started = True

    def stop(self):
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)


# ==============================================================================
# MaAI (VAPモデル) 連携コントローラ
# ==============================================================================

class MaaiController:
    """
    MaAI連携部:
    - 2ch音声（Ch1: ユーザーPCM, Ch2: システムAI音声 / 無音ストリーム）の管理
    - VAPモデルによる P_shift, P_bc のリアルタイム予測
    """
    def __init__(self, state_machine: DialogueStateMachine):
        self.state_machine = state_machine
        self.maai_instance = None
        self.is_running = False
        self.system_audio = SystemAudioInput()
        self.user_audio = UserAudioInput()


        self._init_maai()

    def _init_maai(self):
        try:
            self.maai_instance = MaaiMultiple(
                configs=[
                    {"mode": "vap_mc", "lang": "jp"},
                    {"mode": "bc_2type", "lang": "jp"},
                ],
                audio_ch1=self.user_audio,
                audio_ch2=self.system_audio,
                frame_rate=MAAI_FRAME_RATE,
                context_len_sec=MAAI_CONTEXT_LEN_SEC,
                device="cpu",
                model_type="normal",
            )
            self.maai_instance.start()
            print("[OK] [MaAI] 正常に初期化・起動しました。")
        except ImportError:
            print("[INFO] [MaAI] maaiパッケージ未検出。シミュレーション／待機モードで動作します。")
            self.maai_instance = None
        except Exception as e:
            print(f"[WARN] [MaAI] 初期化スキップ ({e})。シミュレーションモードで継続します。")
            self.maai_instance = None

    def feed_system_audio(self, pcm_data: bytes, in_sample_rate: int = RECEIVE_SAMPLE_RATE):
        """システム発話音声を2ch入力バッファに供給"""
        self.system_audio.put_audio_pcm(pcm_data, in_sample_rate=in_sample_rate)

    def clear_system_audio(self):
        """システム音声バッファをクリア（無音状態へリセット）"""
        self.system_audio.clear_buffer()

    async def poll_loop(self):
        """MaAIの推論結果を監視し、ステートマシンへ通知するループ"""
        self.is_running = True
        try:
            while self.is_running:
                if self.maai_instance:
                    result = await asyncio.to_thread(self.maai_instance.get_result)
                    if result:
                        # p_now は [ユーザー, システム] の確率。
                        p_system_turn = result["vap_mc"]["p_now"][VAP_SYSTEM_SPEAKER_INDEX]

                        # P_bc 取得 (bc_2type の p_bc_react / p_bc_emo)
                        p_react = result["bc_2type"]["p_bc_react"]
                        p_emo = result["bc_2type"]["p_bc_emo"]
                        p_bc = max(p_react, p_emo)
                        category = "emotional" if p_emo > p_react else "reactive"

                        self.state_machine.update_predictions(p_system_turn, p_bc, category)
        except asyncio.CancelledError:
            pass


# ==============================================================================
# メイン対話ループ (AudioVideoLoop)
# ==============================================================================

class AudioVideoLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode

        self.audio_in_queue = asyncio.Queue()
        self.out_queue = asyncio.Queue(maxsize=OUT_QUEUE_MAX_SIZE)  # メモリ使用量増加を防ぐためサイズ制限

        self.session = None
        self.audio_stream = None
        self.session_send_lock = asyncio.Lock()

        # 相槌マネージャ
        self.bc_manager = BackchannelManager()
        # ステートマシン (P_shiftによるGemini投機的リクエスト & 相槌トリガー)
        self.state_machine = DialogueStateMachine(
            on_shift_callback=self.on_speculative_turn_shift,
            on_bc_callback=self.on_backchannel_trigger,
        )
        # MaAI コントローラ
        self.maai_controller = MaaiController(self.state_machine)

    async def on_speculative_turn_shift(self):
        """MaAIがP_shiftを検知した際にGeminiへ投機的トリガーを送信"""
        if self.session:
            try:
                # ユーザー発話終了をGemini Live APIに伝達
                async with self.session_send_lock:
                    await self.session.send_realtime_input(audio_stream_end=True)
                print("[>>] [Gemini Live API] 投機的発話要求を送信しました")
            except Exception as e:
                print(f"[WARN] [Gemini Live API] 投機的送信エラー: {e}")

    async def on_backchannel_trigger(self, category: str):
        """MaAIがP_bcを検知した際の相槌再生（Gemini APIセッションは呼ばない）"""
        await self.bc_manager.play_backchannel(
            category=category,
            on_audio_chunk=self.maai_controller.feed_system_audio,
        )

    # --- Audio Handling ---

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        kwargs = {"exception_on_overflow": False} if __debug__ else {}

        try:
            while True:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)

                # ユーザーの入力音量 (RMS) を更新
                self.bc_manager.update_user_rms(data)

                # MaAI 2ch（ユーザー音声）へ供給
                self.maai_controller.user_audio.put_audio_pcm(data)

                payload = {
                    "data": data,
                    "mime_type": "audio/pcm",
                }
                # 遅延低減のため、キューがいっぱいの場合は最古のデータを破棄
                try:
                    self.out_queue.put_nowait(payload)
                except asyncio.QueueFull:
                    _ = self.out_queue.get_nowait()
                    self.out_queue.put_nowait(payload)

        except asyncio.CancelledError:
            pass
        finally:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        try:
            while True:
                bytestream = await self.audio_in_queue.get()
                await asyncio.to_thread(stream.write, bytestream)
        except asyncio.CancelledError:
            pass
        finally:
            if stream:
                stream.stop_stream()
                stream.close()

    async def receive_audio(self):
        """WebsocketからGeminiの回答PCMを受信し、ステートマシンと同期"""
        try:
            while True:
                turn = self.session.receive()
                interrupted = False
                async for response in turn:
                    server_content = getattr(response, "server_content", None)
                    if server_content and getattr(server_content, "interrupted", False):
                        interrupted = True
                    if data := response.data:
                        self.audio_in_queue.put_nowait(data)
                        # MaAI 2ch（システム音声）へ供給
                        self.maai_controller.feed_system_audio(data, in_sample_rate=RECEIVE_SAMPLE_RATE)
                        continue
                    if text := response.text:
                        print(text, end="", flush=True)

                # Geminiの回答完了（ターン終了）
                self.state_machine.on_gemini_completed()
                self.maai_controller.clear_system_audio()

                # 正常終了した応答音声は再生キューに残す。
                # 割り込み時だけ、既に生成済みの古い音声を破棄する。
                if interrupted:
                    while not self.audio_in_queue.empty():
                        self.audio_in_queue.get_nowait()
        except asyncio.CancelledError:
            pass

    # --- Video Handling ---

    def _capture_frame(self, cap):
        """カメラからフレームをキャプチャしJPEG変換"""
        ret, frame = cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail(IMAGE_MAX_SIZE)

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def capture_frames(self):
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        try:
            while True:
                frame = await asyncio.to_thread(self._capture_frame, cap)
                if frame is None:
                    break
                await asyncio.sleep(VIDEO_FRAME_INTERVAL_SEC)
                await self.out_queue.put(frame)
        except asyncio.CancelledError:
            pass
        finally:
            cap.release()

    def _capture_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]
        i = sct.grab(monitor)
        img = PIL.Image.frombytes("RGB", i.size, i.rgb)
        img.thumbnail(IMAGE_MAX_SIZE)

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def capture_screen(self):
        try:
            while True:
                frame = await asyncio.to_thread(self._capture_screen)
                if frame is None:
                    break
                await asyncio.sleep(VIDEO_FRAME_INTERVAL_SEC)
                await self.out_queue.put(frame)
        except asyncio.CancelledError:
            pass

    # --- Text & Main Loop ---

    async def send_text(self):
        try:
            while True:
                text = await asyncio.to_thread(input, "message > ")
                if text.lower() == "q":
                    print("[BYE] 終了リクエストを受信しました。")
                    break
                await self.session.send_client_content(
                    turns=types.Content(parts=[types.Part.from_text(text=text or "")]),
                    turn_complete=True,
                )
        except asyncio.CancelledError:
            pass

    async def send_realtime(self):
        try:
            while True:
                msg = await self.out_queue.get()
                async with self.session_send_lock:
                    if msg["mime_type"].startswith("audio/"):
                        await self.session.send_realtime_input(audio=msg)
                    else:
                        await self.session.send_realtime_input(media=msg)
        except asyncio.CancelledError:
            pass

    async def run(self):
        """全非同期タスクの統括実行"""
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
            http_options={"api_version": "v1alpha"},
        )
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=OUT_QUEUE_MAX_SIZE)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())

                if self.video_mode == "camera":
                    tg.create_task(self.capture_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.capture_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                # MaAIの判定ループを並行起動
                tg.create_task(self.maai_controller.poll_loop())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            if self.audio_stream:
                self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ai - VAP + Gemini Live API Hybrid Agent")
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="映像入力モード (camera, screen, none)",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    main = AudioVideoLoop(video_mode=args.mode)
    asyncio.run(main.run())
