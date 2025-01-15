wandb_entity="continual-learning-2024"
project_name="trash-gradients"
for seed_id in $(seq 1 10);
do
    python3 -m main --wandb_entity $wandb_entity --wandb_project $project_name-5  --id $seed_id  --dataset_size 5   --cl_epochs 50 --probing_epochs 60 --batch_size 256 --lr 0.001 --use_gradient_alignment
    python3 -m main --wandb_entity $wandb_entity --wandb_project $project_name-10 --id $seed_id  --dataset_size 10  --cl_epochs 50 --probing_epochs 60 --batch_size 256 --lr 0.001 --use_gradient_alignment
    python3 -m main --wandb_entity $wandb_entity --wandb_project $project_name-20 --id $seed_id  --dataset_size 20  --cl_epochs 50 --probing_epochs 60 --batch_size 256 --lr 0.001 --use_gradient_alignment
    python3 -m main --wandb_entity $wandb_entity --wandb_project $project_name-50 --id $seed_id  --dataset_size 50  --cl_epochs 50 --probing_epochs 60 --batch_size 256 --lr 0.001 --use_gradient_alignment
done