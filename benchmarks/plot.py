import matplotlib.pyplot as plt
import numpy as np


def plot(data, title, title2, legend, size=(12, 7)):
    labels = []

    all_times_data = {
        "before": [],
        "after": []
    }

    for d in data:
        labels.append(d["label"])
        for key in all_times_data:
            if key not in d:
                raise ValueError(f"Data dictionary missing key '{key}'. Ensure all required fields are present.")
            all_times_data[key].append(d[key])

    x = np.arange(len(labels))

    rotation = 0

    num_bars = 2
    width = 0.35
    offsets = [(i - (num_bars - 1) / 2) * width for i in range(num_bars)]

    fig, ax = plt.subplots(figsize=size)

    rects_list = []

    colors = ['#ff7f0e', '#1f77b4']

    rects_list.append(ax.bar(x + offsets[0], all_times_data["before"], width, label=legend[0], color=colors[0]))
    rects_list.append(ax.bar(x + offsets[1], all_times_data["after"], width, label=legend[1], color=colors[1]))

    ax.set_title(title, fontsize=14)
    ax.set_ylabel(title2, fontsize=12)
    ax.set_xticks(x)

    ax.set_xticklabels(labels, rotation=rotation, ha="center", fontsize=10)

    expected_legend_len = num_bars
    if len(legend) != expected_legend_len:
        raise ValueError(f"Legend must contain {expected_legend_len} entries (for {num_bars} bars plotted), but received {len(legend)} entries.")

    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel_absolute_value(rects_list_to_label):
        for rect in rects_list_to_label:
            height = rect.get_height()
            time_str = f'{height:.2f}'
            ax.annotate(time_str,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    def autolabel_with_improvement(rects_list_to_label, before_data_list):
        for i, rect in enumerate(rects_list_to_label):
            height = rect.get_height()
            before_h = before_data_list[i]
            time_str = f'{height:.2f}'

            if before_h > 0:
                improvement_percent = ((before_h - height) / before_h) * 100
                improvement_str = f'{improvement_percent:.1f}%'
                text_to_annotate = f'{time_str}\n{improvement_str}'
            else:
                text_to_annotate = time_str

            ax.annotate(text_to_annotate,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel_absolute_value(rects_list[0])
    autolabel_with_improvement(rects_list[1], all_times_data["before"])

    plt.tight_layout()
    plt.show()
    plt.close(fig)
