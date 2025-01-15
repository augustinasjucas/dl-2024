wandb_entity="continual-learning-2024"
project_name="limited_replay"
for seed_id in $(seq 1 3);
do

    # Replay 1 or 2 tasks for all dataset sizes
    for task_limit in $(seq 1 2);
    do
        python3 -m main --wandb_entity $wandb_entity --wandb_project $project_name-5_split-replaying_$task_limit  --id $seed_id  --dataset_size 5   --cl_epochs 15 --probing_epochs 35 --use_replay --replay_buffer_size 30 --replay_batch_size 16 --limited_replay --task_limit $task_limit
        python3 -m main --wandb_entity $wandb_entity --wandb_project $project_name-10_split-replaying_$task_limit --id $seed_id --dataset_size 10   --cl_epochs 15 --probing_epochs 35 --use_replay --replay_buffer_size 30 --replay_batch_size 16 --limited_replay --task_limit $task_limit
        python3 -m main --wandb_entity $wandb_entity --wandb_project $project_name-20_split-replaying_$task_limit --id $seed_id --dataset_size 20   --cl_epochs 15 --probing_epochs 35 --use_replay --replay_buffer_size 30 --replay_batch_size 16 --limited_replay --task_limit $task_limit
        python3 -m main --wandb_entity $wandb_entity --wandb_project $project_name-50_split-replaying_$task_limit --id $seed_id --dataset_size 50   --cl_epochs 15 --probing_epochs 35 --use_replay --replay_buffer_size 30 --replay_batch_size 16 --limited_replay --task_limit $task_limit
    done

    # Also replay 3 or 4 tasks for dataset size 5
    for task_limit in $(seq 3 4);
    do
        python3 -m main --wandb_entity $wandb_entity --wandb_project $project_name-5_split-replaying_$task_limit  --id $seed_id  --dataset_size 5   --cl_epochs 15 --probing_epochs 35 --use_replay --replay_buffer_size 30 --replay_batch_size 16 --limited_replay --task_limit $task_limit
    done
done
