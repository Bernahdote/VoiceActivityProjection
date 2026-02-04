import matplotlib.pyplot as plt
import torch
import vap.data.datamodule as dm
from vap.objective import VAPObjective

#csv = "/mnt/sda/willem/datasets/seamless_interaction/sliding_window_dset.csv" # On linux
csv = "/tmp/sliding_window_single.csv" # On mac


window_sec = 2.0
frame_hz = 50

dset = dm.VAPDataset(csv)

row_indices = list(range(len(dset)))

def plot_sample(sample, time_sec):
    frame_idx = int(time_sec * frame_hz)
    vad = sample["vad"]  # (T,2)
    t_vad = torch.arange(vad.shape[0]) / frame_hz

    objective = VAPObjective(frame_hz=frame_hz)
    labels = objective.get_labels(vad.unsqueeze(0)).squeeze(0)
    decoded = objective.codebook.decode(labels)  # (T, 2, 4)
    grid = (decoded[frame_idx].cpu().numpy() > 0.5).astype(int)

    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], wspace=0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)

    bin_edges = [0.0, 0.2, 0.6, 1.2, 2.0]

    def overlay_bins(ax, row_idx, color):
        for edge in bin_edges[1:-1]:
            ax.axvline(
                time_sec + edge,
                color="black",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
            )
        for j in range(4):
            if grid[row_idx, j] == 1:
                ax.axvspan(
                    time_sec + bin_edges[j],
                    time_sec + bin_edges[j + 1],
                    color=color,
                    alpha=0.2,
                    lw=0,
                )

    ax0.plot(t_vad.numpy(), vad[:, 0].numpy(), color="tab:blue", label="Speaker A")
    overlay_bins(ax0, 0, "tab:blue")
    ax0.set_title("VAD")
    ax0.set_ylabel("activity")
    ax0.set_xlim(time_sec, time_sec + window_sec)
    ax0.set_ylim(-0.1, 1.1)
    ax0.legend()

    ax1.plot(t_vad.numpy(), vad[:, 1].numpy(), color="tab:orange", label="Speaker B")
    overlay_bins(ax1, 1, "tab:orange")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("activity")
    ax1.set_xlim(time_sec, time_sec + window_sec)
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend()

    ax2 = fig.add_subplot(gs[:, 1])
    ax2.imshow(grid, cmap="Greys", vmin=0, vmax=1, origin="upper")
    ax2.set_title(f"Ground truth VAP label ({time_sec:.0f}s)")
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(2))
    ax2.set_xticklabels(["0.2 s", "0.6 s", "1.2 s", "2.0 s"])
    ax2.set_yticklabels(["Speaker A", "Speaker B"])
    ax2.set_xticks([x - 0.5 for x in range(1, 4)], minor=True)
    ax2.set_yticks([y - 0.5 for y in range(1, 2)], minor=True)
    ax2.grid(which="minor", color="black", linewidth=1)

    for i in range(2):
        for j in range(4):
            ax2.text(j, i, str(grid[i, j]), ha="center", va="center", color="red")

    plt.tight_layout()
    plt.show()


for idx in row_indices:
    sample = dset[idx]
    n_frames = sample["vad"].shape[0]
    max_time = n_frames / frame_hz
    time_sec = 0.0
    while time_sec + window_sec <= max_time:
        plot_sample(sample, time_sec)
        time_sec += window_sec
