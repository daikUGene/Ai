from llm import LLM
from speech_recognizer import SpeechRecognizer
from speech_synthesizer import SpeechSynthesizer
import re

class Ai:
    def __init__(self):
        self.llm = LLM()
        self.speech_recognizer = SpeechRecognizer()
        self.speech_synthesizer = SpeechSynthesizer()

    def talk(self):
        self.speech_synthesizer.speak("なにか話してくださいよー")

        while True:
            recognition_result = self.speech_recognizer.recognize()

            buffer = ""  # バッファ
            for text in self.llm.generate(recognition_result):
                text = buffer + text  # バッファとチャンクを結合
                buffer = ""  # バッファをクリア

                sentences = re.findall("[^。！？!?]+[。！？!?]?", text)  # 区切り文字で分割

                for sentence in sentences:
                    sentence = sentence.strip()

                    if len(sentence) > 0 and sentence[-1] in "。！？!?":
                        self.speech_synthesizer.speak(sentence)
                        print("success: ", sentence)
                    else:
                        # 区切り文字で終わらない文章は、次のチャンクの先頭部分と結合するためバッファに格納
                        buffer = sentence

if __name__ == "__main__":
    ai = Ai()
    ai.talk()
