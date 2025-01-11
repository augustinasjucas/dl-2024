import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from reproduce import run_experiment


def main():
    layers_order = ["conv1", "conv2", "conv3", "fc1"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    global_similarities = []
    layerwise_similarities = []
    for _ in range(1):
        model, metric_results = run_experiment(wandb_logging=False)
        global_similarities.append(metric_results[0]["global_similarities"])
        layerwise_similarities.append(metric_results[0]["layerwise_similarities"])

    global_similarities = np.array(global_similarities)
    layerwise_similarities = np.array(layerwise_similarities)
    average_global_similarities = np.mean(global_similarities, axis=0)
    average_layerwise_similarities = np.mean(layerwise_similarities, axis=0)

    num_tasks = average_global_similarities.shape[0]
    min_similarity = min(np.min(average_global_similarities), np.min(average_layerwise_similarities))
    max_similarity = max(np.max(average_global_similarities), np.max(average_layerwise_similarities))

    fig_global, ax_global = plt.subplots(figsize=(8, 6), dpi=300)
    cax_global = ax_global.matshow(
        average_global_similarities, cmap="viridis", interpolation="none", vmin=min_similarity, vmax=max_similarity
    )
    fig_global.colorbar(cax_global)
    ax_global.set_title("Average cosine similarity (global) between tasks")
    ax_global.set_xlabel("Task index")
    ax_global.set_ylabel("Task index")
    ax_global.set_xticks(range(num_tasks))
    ax_global.set_yticks(range(num_tasks))
    plt.savefig(f"{output_dir}/average_global_similarities.png", dpi=300)

    fig_layers, axes = plt.subplots(1, len(layers_order), figsize=(5 * len(layers_order), 5), dpi=300)
    if len(layers_order) == 1:
        axes = [axes]  # Make it iterable

    for idx, layer_name in enumerate(layers_order):
        ax = axes[idx]
        cax_layer = ax.matshow(
            average_layerwise_similarities[idx],
            cmap="viridis",
            interpolation="none",
            vmin=min_similarity,
            vmax=max_similarity,
        )
        ax.set_title(f"{layer_name}")
        ax.set_xlabel("Task index")
        ax.set_ylabel("Task index")
        ax.set_xticks(range(num_tasks))
        ax.set_yticks(range(num_tasks))
        fig_layers.colorbar(cax_layer, ax=ax)
    plt.savefig(f"{output_dir}/average_layerwise_similarities.png", dpi=300)

    np.save(os.path.join(output_dir, "global_similarities.npy"), global_similarities)
    np.save(os.path.join(output_dir, "layerwise_similarities.npy"), layerwise_similarities)


if __name__ == "__main__":
    main()
