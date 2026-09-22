import asyncio
import base64
import os
import sys
import pyaudio
from openai import AsyncOpenAI

# 音声フォーマット設定 (GPT-Live-1の標準仕様: PCM16 / 24kHz / モノラル)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK_SIZE = 960  # 20ms分 (24000Hz * 2bytes * 0.02s)


async def main() -> None:
    # 環境変数からAPIキーを取得・確認
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("エラー: 環境変数 'OPENAI_API_KEY' が設定されていません。", file=sys.stderr)
        sys.exit(1)

    p = pyaudio.PyAudio()
    loop = asyncio.get_running_loop()

    input_queue: asyncio.Queue[bytes] = asyncio.Queue()
    output_queue: asyncio.Queue[bytes] = asyncio.Queue()

    # 1. マイク入力用コールバック（データを非同期キューへ転送）
    def mic_callback(in_data, frame_count, time_info, status):
        loop.call_soon_threadsafe(input_queue.put_nowait, in_data)
        return (None, pyaudio.paContinue)

    # マイク入力ストリームの初期化
    mic_stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE // 2,
        stream_callback=mic_callback,
    )

    # スピーカー出力ストリームの初期化
    speaker_stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
    )

    # セッションの初期設定
    session_config = {
        "model": "gpt-live-1",
        "instructions": (
            "あなたは親しみやすい相棒AIロボットです。"
            "人間らしい自然な日本語で、短く相槌を打ちながら回答してください。"
        ),
        "audio": {
            "format": {"type": "audio/pcm", "rate": RATE},
            "output": {"voice": "marin"},
        },
    }

    # APIクライアント接続
    async with AsyncOpenAI(api_key=api_key) as client:
        async with client.live.connect() as connection:
            await connection.session.start(session=session_config)

            # タスクA: マイク音声をAPIへ常時ストリーミング送信
            async def send_mic_audio():
                while True:
                    data = await input_queue.get()
                    encoded = base64.b64encode(data).decode("ascii")
                    await connection.session.input_audio.append(audio=encoded)

            # タスクB: APIからの音声応答を受信してスピーカーキューへ追加
            async def receive_api_events():
                async for event in connection:
                    if event.type == "session.started":
                        print("\n>>> GPT-Live-1 に接続しました。マイクに向かって話しかけてください。")
                    elif event.type == "session.output_audio.delta":
                        raw_pcm = base64.b64decode(event.delta)
                        await output_queue.put(raw_pcm)
                    elif event.type == "error":
                        print(f"\nエラーが発生しました: {event}", file=sys.stderr)

            # タスクC: スピーカーキューから音声を取り出して再生
            async def play_speaker_audio():
                while True:
                    raw_pcm = await output_queue.get()
                    await loop.run_in_executor(None, speaker_stream.write, raw_pcm)
                    output_queue.task_done()

            # 入出力ストリームの開始
            mic_stream.start_stream()
            speaker_stream.start_stream()

            send_task = asyncio.create_task(send_mic_audio())
            receive_task = asyncio.create_task(receive_api_events())
            play_task = asyncio.create_task(play_speaker_audio())

            try:
                await asyncio.gather(send_task, receive_task, play_task)
            except KeyboardInterrupt:
                print("\n対話を終了します。")
            finally:
                send_task.cancel()
                receive_task.cancel()
                play_task.cancel()

                mic_stream.stop_stream()
                mic_stream.close()
                speaker_stream.stop_stream()
                speaker_stream.close()
                p.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
