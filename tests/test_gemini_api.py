import google.generativeai as genai
import os

# APIキーの設定
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# モデルの設定
model = genai.GenerativeModel('gemini-pro')

# テキスト生成
# response = model.generate_content("What is the meaning of life?")
response = model.generate_content("人生の意味って何でしょうか？")
print(response.text)
