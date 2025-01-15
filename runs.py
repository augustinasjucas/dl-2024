from reproduce import run_experiment


def main():
    task_sizes = [5, 10, 20, 50]
    iterations = 5
    for _ in range(iterations):
        for task_size in task_sizes:
            run_experiment(
                project_name=f"gradients-{task_size}-images-dropout", task_size=task_size, wandb_logging=True
            )


if __name__ == "__main__":
    main()
