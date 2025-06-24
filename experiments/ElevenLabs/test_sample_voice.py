from elevenlabs import Voice, VoiceSettings, play
from elevenlabs.client import ElevenLabs
import os

ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)

audio = client.generate(
    text="こんにちは。わたしはアイです。",
    voice=Voice(
        voice_id='AZnzlk1XvdvUeBnXmlld',  # Domi
        settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
    ),
    model="eleven_multilingual_v2",
)

play(audio)
