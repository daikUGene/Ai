# オウム返しするプログラム

import os
from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss
from eff_word_net import samples_loc

import speech_recognition as sr

from gtts import gTTS 
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

import threading

def play_sound_effect(sound):
    play(sound)

base_model = Resnet50_Arc_loss()

ai_hw = HotwordDetector(
    hotword="ai",
    model = base_model,
    reference_file="./ai_ref.json",
    threshold=0.7,
    relaxation_time=2
)

mic_stream = SimpleMicStream(
    window_length_secs=1.5,
    sliding_window_secs=0.75,
)

mic_stream.start_stream()

# 音声認識器の初期化
r = sr.Recognizer()
r.energy_threshold = 500  # 音量のしきい値
r.dynamic_energy_threshold = False

# 高速化のため、ウェイクワード検出時の効果音を事前に読み込み
sound_path = "./hotword_detected.mp3"
sound = AudioSegment.from_file(sound_path, format="mp3")

volume = 0  # 正の値で音量アップ、負の値で音量ダウン
sound = sound + volume  # 音量変更

print("Say Ai")
while True:
    frame = mic_stream.getFrame()
    result = ai_hw.scoreFrame(frame)

    if result==None:
        # ウェイクワードが検出されない場合
        continue

    if(result["match"]):
        # ウェイクワードが検出された場合
        print(result)
        print("ウェイクワードが検出されました。信頼度：", result["confidence"])

        # 別スレッドで効果音を再生
        sound_effect_thread = threading.Thread(target=play_sound_effect, args=(sound,))
        sound_effect_thread.start()

        while True:
            # マイクから音声を取得
            with sr.Microphone() as source:
                # # 音声入力の準備が整ったら、ウェイクワード検出時の効果音を再生
                # play(sound)

                print("Say something!")
                audio = r.listen(source)

            # Google Speech Recognitionを使って音声認識
            try:
                recognition_result = r.recognize_google(audio, language="ja-JP")
            
            except sr.UnknownValueError:
                # 雑音などで認識が上手く行かなかった場合
                print("Google Speech Recognition could not understand audio")

                # 再度音声の取得を開始
                continue

            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))

            else:
                # 音声認識に成功した場合
                print("Google Speech Recognition thinks you said " + recognition_result)

                # オウム返し
                # ファイルライクオブジェクトに書き込み
                fp = BytesIO()
                tts = gTTS(text=recognition_result, lang='ja')
                tts.write_to_fp(fp)
                fp.seek(0)

                # AudioSegmentオブジェクトを作成
                audio = AudioSegment.from_file(fp, format="mp3")

                # 再生速度を変更
                audio = audio.speedup(playback_speed=1.2)

                # 再生
                play(audio) 

                break
