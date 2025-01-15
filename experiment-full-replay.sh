wandb_entity="continual-learning-2024"
project_name="full_replay"
for seed_id in $(seq 1 10);
do
    python3 -m main --wandb_entity $wandb_entity --wandb_project $project_name-1  --id $seed_id  --dataset_size 1   --cl_epochs 15 --probing_epochs 35 --use_replay --replay_buffer_size 30 --replay_batch_size 16
    python3 -m main --wandb_entity $wandb_entity --wandb_project $project_name-5  --id $seed_id  --dataset_size 5   --cl_epochs 15 --probing_epochs 35 --use_replay --replay_buffer_size 30 --replay_batch_size 16
    python3 -m main --wandb_entity $wandb_entity --wandb_project $project_name-10 --id $seed_id --dataset_size 10   --cl_epochs 15 --probing_epochs 35 --use_replay --replay_buffer_size 30 --replay_batch_size 16
    python3 -m main --wandb_entity $wandb_entity --wandb_project $project_name-20 --id $seed_id --dataset_size 20   --cl_epochs 15 --probing_epochs 35 --use_replay --replay_buffer_size 30 --replay_batch_size 16
    python3 -m main --wandb_entity $wandb_entity --wandb_project $project_name-50 --id $seed_id --dataset_size 50   --cl_epochs 15 --probing_epochs 35 --use_replay --replay_buffer_size 30 --replay_batch_size 16
done