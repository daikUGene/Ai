from maai import Maai, MaaiInput

mic = MaaiInput.Mic(mic_device_index=0)
zero = MaaiInput.Zero()

maai = Maai(mode="bc_2type", lang="jp", frame_rate=10, context_len_sec=5, audio_ch1=mic, audio_ch2=zero, device="cpu")
maai.start()

while True:
    result = maai.get_result()

    print(f"result['p_bc_react'] = {result['p_bc_react']}")  # 応答系相槌の確率
    print(f"result['p_bc_emo'] = {result['p_bc_emo']}")      # 感情表出系相槌の確率