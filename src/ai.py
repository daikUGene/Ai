from llm import LLM
from speech_recognizer import SpeechRecognizer
from speech_synthesizer import SpeechSynthesizer

class Ai:
    def __init__(self):
        self.llm = LLM()
        self.speech_recognizer = SpeechRecognizer()
        self.speech_synthesizer = SpeechSynthesizer()

    def talk(self):
        self.speech_synthesizer.speak("なにか話してくださいよー")

        while True:
            recognition_result = self.speech_recognizer.recognize()
            llm_result = self.llm.generate(recognition_result)
            self.speech_synthesizer.speak(llm_result)

if __name__ == "__main__":
    ai = Ai()
    ai.talk()
