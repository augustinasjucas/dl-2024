# Continual Learning: a Layerwise Perspective on Forgetting

# Installation
To reproduce our environment perfectly, please do the following:
1. Install CUDA 12.1: https://developer.nvidia.com/cuda-12-1-0-download-archive.
2. Install pytorch for CUDA 12.1: `pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`.
3. Install remaining requirements: `pip install -r requirements.txt`

Also, when running an experiment for the first time, you might want to include a flag `--wandb_api_key <api_key>`, which will connect to your wandb account for logging. This login is cached, therefore adding this flag is neccessary for only a single run.

# Reproducing the Experiments
For the sake of brevity, here we provide a short description on how to reproduce the exact results we use in the paper.

Every run of *main.py* correponds to a single seed run of an experiment. In order to run multiple seeds of the same experiment, we ran *main.py* with the same parameters a few times (therefore, we have script files for running seeds). This results in multiple instances of the same experiment being logged to the same wandb project, which allowed us to then export the data and analyze it. We did not configure sweeping here, because every run of *main.py* actually creates multiple runs (different runs for logging, for metrics and for the core continual learning), which is not allowed while sweeping in wandb.

Therefore, we have simple manually written scripts to run every experiment. Every script runs the experiments for some number of random seeds. The number of seeds varies from experiment to experiment purely due to strong computational constraints.

**Running all experiments**. You can use `bash run-all-experiments.sh` to run all of the following experiments at once.

**Task size experiment**. `bash experiment-task-size.sh`. Effectively calls main.py with `--dataset_size <5/10/20/50> --cl_epochs 15 --probing_epochs 35` flags. Check the script for all parameters.

**Full replay experiment**. `bash experiment-full-replay.sh`. Effectively calls main.py with `--dataset_size <5/10/20/50> --cl_epochs 15 --probing_epochs 35 --use_replay --replay_buffer_size 30 --replay_batch_size 16` flags. 

**Limited replay experiment**. `bash experiment-limited-replay.sh`. Effectively calls main.py with `--dataset_size <5/10/20/50> --cl_epochs 15 --probing_epochs 35 --use_replay --replay_buffer_size 30 --replay_batch_size 16 --limited_replay --task_limit <1/2/3/4>` flags.

**Gradients experiment**. `bash experiment-gradients.sh`. Effectively calls main.py with `--dataset_size <5/10/20/50> --cl_epochs 50 --use_gradient_alignment` flags.


# Understanding the Codebase

## General Flow of a Single Experiment Instance 
The main flow of every experiment instance (1 seed) is as follows:
1. First, a raw dataset (in all cases - CIFAR100) is taken. The dataset is split into tasks, each of which includes the same number of classes, specified by a parameter passed to the script. 
2. Then, on these split tasks (suppose there are *n* tasks), sequential continual learning is performed (managed by CLTraining class). In it purest form, this class simply trains the same model on the first task, then continues training on the second task, etc. until it finishes training on the last task.
3. Throughout the training (and possibly just after it), multiple metrics are calculated. Metrics will be discussed later, but they can include anything: from probing accuracies, to gradient alignments to simply testing accuracies on datasets.
4. After CL is finished, the experiment is finished.

Therefore, all of the core logic we care about is done inside of the metrics. The `main.py` (main class) is mostly responsible for splitting the dataset into tasks, initializing the metrics (based on provided parameters) and calling `run()` on the continual learning task. All of the core logic we mostly care about is implemented in Metric classes.

## Metrics
During the sequential continual learning, after every major event such as starting a new task, ending of an epoch, getting results for a batch, etc., a callback method is induced in a Metric object. This allows the metrics to calculate whatever values they require. We further describe the main metrics we use. All metric classes are stored in common/metrics/ folder. 

**FlorianProbing**. This metric calculates the accuracies after freezing certain layers and retraining. It logs the results to wandb.

**GradientAlignment**. Computes pairwise cosine similarities between gradients of tasks as they are trained on. Logs the gradient similarities, gradient magnitudes and and task similarity matrices into wandb. The working of this metric is described in detail in the paper.

**LayerL2** and **LayerCosineDistance**. These metrics log how the layers change in between training of tasks and log the results to wandb.

**BeautifulLogging**. A metric responsible for calculating testing accuracies in between epochs. Logs the results to wandb for easy monitoring of how sequential training is occuring.
   
## Replay
One can also pass a Replay object to CLTraining, which will enable a Replay buffer while training continually. 

**SimpleReplay** is the most simple Replay buffer, which, after training on a task, for every class of that task picks a specified number of samples and stores them in the buffer. Then during furhter training, *sample_batch()* can be called on the Replay object and a batch of specified number of samples is sampled from the buffer, which is then merged with the current task data in CLTraining class. 

**LimitedReplay** works the same way as SimpleReplay. However, when a new task is added, if the buffer stores too many tasks, it simply removes all classes of the oldest taksk from the buffer.