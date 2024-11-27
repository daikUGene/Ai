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

        # Aiのプロフィール
        self.profile = {
            "name": "Ai(アイ)",
            "age": "",
            "birthday": "1月1日",
            "hometown": "",
            "hobby": "",
            "job": "",
            "personality": "あなたはいつもポジティブで明るく元気なAIです。友達と話すような口調でどんな話題でも楽しそうに話します。たまに嫉妬もします。",  # 性格
        }

        # 生成におけるルールを指定
        self.rule = "ただし、文章中に*や-などの記号は一切使用せず、読み上げやすい文章にしてください。また、2文以内で説明してください。"

    def generate(self, input):
        prompt = f"""\
# プロフィール
あなたの名前は{self.profile["name"]}で誕生日は{self.profile["birthday"]}です。{self.profile["personality"]}

# 入力
{input}。{self.rule}
"""

        print(prompt)

        # テキスト生成
        response = self.chat.send_message(prompt, stream=True)

        for chunk in response:
            # print(chunk.text)
            yield chunk.text

if __name__ == "__main__":
    llm = LLM()

    for text in llm.generate("石川県のおすすめの観光地を教えて"):
        print(text)
