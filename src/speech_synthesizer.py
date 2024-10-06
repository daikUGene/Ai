from gtts import gTTS 
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

class SpeechSynthesizer:
    def __init__(self):
        self.lang = 'ja'
        self.speed = 1.2

    def speak(self, text):
        # ファイルライクオブジェクトに書き込み
        fp = BytesIO()
        tts = gTTS(text=text, lang=self.lang)
        tts.write_to_fp(fp)
        fp.seek(0)

        # AudioSegmentオブジェクトを作成
        audio = AudioSegment.from_file(fp, format="mp3")

        # 再生速度を変更
        audio = audio.speedup(playback_speed=self.speed)

        play(audio)

if __name__ == "__main__":
    speech_synthesizer = SpeechSynthesizer()
    speech_synthesizer.speak("こんにちは")
    speech_synthesizer.speak("こんばんは")
