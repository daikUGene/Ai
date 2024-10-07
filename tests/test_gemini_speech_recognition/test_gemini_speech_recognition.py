import google.generativeai as genai
import os
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# APIキーの設定
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')  # 環境変数から読み込み
genai.configure(api_key=GOOGLE_API_KEY)

# モデルの設定
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# パラメータの設定
generation_config = {"temperature": 0.7}

# 安全フィルタの設定
# フィルタを一切かけない設定
safety_settings={
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}
# 最も強いフィルタ設定
# safety_settings={
#     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
#     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
#     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
#     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
# }

#--- 音声認識 ----#
# マイクの音量を50%くらいにすると雑音が少ないため、スムーズに音声入力が可能
# マイクの雑音に合わせて音声認識器のしきい値を調整

# import speech_recognition as sr

# # 音声認識器の初期化
# r = sr.Recognizer()
# r.energy_threshold = 500  # 音量のしきい値
# r.dynamic_energy_threshold = False

# # マイクから音声を取得
# with sr.Microphone() as source:
#     print("Say something!")
#     audio = r.listen(source)
#     print("Good!")
# #--- 音声認識 ----#

# prompt = "Generate transcription from the audio, only extract speech and ignore background audio."

# response = model.generate_content([prompt, audio])

# try:
#     print(response.text)
# except ValueError:
#     print(response.prompt_feedback)

# 音声ファイルでの音声認識テスト
# from pydub import AudioSegment

print("upload")
audio_file = genai.upload_file("audio.mp3")

print("start")
# プロンプトの準備
response = model.generate_content(
    [
        "Generate transcription from the audio, only extract speech and ignore background audio.",
        audio_file
    ]
)
print(response.text)
