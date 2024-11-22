import google.generativeai as genai
import os
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class LLM:
    def __init__(self, model_name='gemini-1.5-flash-8b', temperature=0.7):
        # APIキーの設定
        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=GOOGLE_API_KEY)

        # モデルの設定
        self.model = genai.GenerativeModel(model_name)

        # パラメータの設定
        # self.generation_config = {"temperature": 0.9}

        # 安全フィルタの設定
        # フィルタを一切かけない設定
        self.safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # チャットの初期化
        self.chat = self.model.start_chat(history=[])

    def generate(self, prompt):
        # テキスト生成
        response = self.chat.send_message(prompt, stream=True)

        for chunk in response:
            # print(chunk.text)
            yield chunk.text

if __name__ == "__main__":
    llm = LLM()
    llm.generate("こんにちは。あなたの名前はなんですか？")
