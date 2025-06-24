# 無料プランでは使用不可
# プランをアップグレードするよう促すエラーが返される

from elevenlabs import Voice, VoiceSettings, play
from elevenlabs.client import ElevenLabs
import os

ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)

voice = client.clone(
    name="Ai",
    # description="An old American male voice with a slight hoarseness in his throat. Perfect for news", # Optional
    description="A charming, youthful voice with a childlike quality. ", # Optional
    files=["sample0.wav", "sample1.wav", "sample2.wav"],  # VOICEVOX:ずんだもん
)

audio = client.generate(
    text="こんにちは。わたしはアイです。",
    voice=voice,
    model="eleven_multilingual_v2",
)

play(audio)
