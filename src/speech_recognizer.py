import speech_recognition as sr

class SpeechRecognizer:
    def __init__(self, energy_threshold=500, is_dynamic_energy_threshold_enabled=False):
        self.r = sr.Recognizer()
        self.r.energy_threshold = energy_threshold  # 音量のしきい値
        self.r.dynamic_energy_threshold = is_dynamic_energy_threshold_enabled

    def recognize(self):
        # マイクから音声を取得
        with sr.Microphone() as source:
            print("Say something!")
            audio = self.r.listen(source)

        # Google Speech Recognitionを使用
        try:
            recognition_result = self.r.recognize_google(audio, language="ja-JP")
            print("Google Speech Recognition thinks you said " + recognition_result)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

        return recognition_result

if __name__ == "__main__":
    speech_recognizer = SpeechRecognizer()
    speech_recognizer.recognize()
