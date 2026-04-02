import torch
import matplotlib.pyplot as plt
from vap.utils.audio import load_waveform

AUDIO_A = "/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0038/V00_S2020_I00000686_P1275A.wav"
AUDIO_B = "/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0032/V00_S2020_I00000686_P1276A.wav"

START = 0.0
END = 10.0
SAMPLE_RATE = 16000

wa, _ = load_waveform(AUDIO_A, start_time=START, end_time=END, sample_rate=SAMPLE_RATE, mono=True)
wb, _ = load_waveform(AUDIO_B, start_time=START, end_time=END, sample_rate=SAMPLE_RATE, mono=True)

t = torch.arange(wa.shape[-1]) / SAMPLE_RATE

fig_a, ax_a = plt.subplots(figsize=(12, 2))
ax_a.plot(t, wa[0].numpy(), color="royalblue", lw=0.7)
ax_a.axis("off")
plt.tight_layout()

fig_b, ax_b = plt.subplots(figsize=(12, 2))
ax_b.plot(t, wb[0].numpy(), color="royalblue", lw=0.7)
ax_b.axis("off")
plt.tight_layout()

plt.show()
