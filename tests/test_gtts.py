from gtts import gTTS 
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

text = "こんにちは。私はAIアシスタントAiです。よろしくお願いします。"

# ファイルライクオブジェクトに書き込み
fp = BytesIO()
tts = gTTS(text=text, lang='ja')
tts.write_to_fp(fp)
fp.seek(0)

# AudioSegmentオブジェクトを作成
audio = AudioSegment.from_file(fp, format="mp3")

# 再生速度を変更
# 1.3倍速以上だと違和感
audio = audio.speedup(playback_speed=1.2)

# 再生
play(audio) 
