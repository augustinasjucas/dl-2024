from reproduce import run_experiment


def main():
    task_sizes = [10, 5]
    iterations = 1
    for _ in range(iterations):
        for task_size in task_sizes:
            run_experiment(project_name=f"gradients-{task_size}-excluding", task_size=task_size, wandb_logging=True)


if __name__ == "__main__":
    main()
