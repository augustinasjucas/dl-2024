wandb_entity="continual-learning-2024"
project_name="new_extra_data"
for seed_id in $(seq 1 3);
do

    # task size=5
    python3 -m main --wandb_entity $wandb_entity --wandb_project $project_name-5-0.125  --id $seed_id  --dataset_size 5   --cl_epochs 15 --probing_epochs 35 --extra_data --percent_new 0.125
    python3 -m main --wandb_entity $wandb_entity --wandb_project $project_name-5-0.25   --id $seed_id  --dataset_size 5   --cl_epochs 15 --probing_epochs 35 --extra_data --percent_new 0.25
    python3 -m main --wandb_entity $wandb_entity --wandb_project $project_name-5-0.5    --id $seed_id  --dataset_size 5   --cl_epochs 15 --probing_epochs 35 --extra_data --percent_new 0.5
    python3 -m main --wandb_entity $wandb_entity --wandb_project $project_name-5-1.0    --id $seed_id  --dataset_size 5   --cl_epochs 15 --probing_epochs 35 --extra_data --percent_new 1

done
