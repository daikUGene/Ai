# マイクの音量を50%くらいにすると雑音が少ないため、スムーズに音声入力が可能
# マイクの雑音に合わせて音声認識器のしきい値を調整

import speech_recognition as sr

# 音声認識器の初期化
r = sr.Recognizer()
r.energy_threshold = 500  # 音量のしきい値
r.dynamic_energy_threshold = False

# マイクから音声を取得
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)
    print(r.energy_threshold)

# Google Speech Recognitionを使って音声認識
try:
    print("Google Speech Recognition thinks you said " + r.recognize_google(audio, language="ja-JP"))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
