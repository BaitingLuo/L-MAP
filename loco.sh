export PYTHONPATH=.:$PYTHONPATH

name=T-1
#datasets=(hopper-medium-replay-v0 hopper-medium-v0 hopper-medium-expert-v0 walker2d-medium-expert-v0 walker2d-medium-v0 walker2d-medium-replay-v0)
datasets=(hopper-medium-replay-v2)

for data in ${datasets[@]}; do
  for round in 1; do
    ##CUDA_VISIBLE_DEVICES=0 python scripts/train.py --dataset $data --exp_name $name-$round --tag development --seed $round
    #CUDA_VISIBLE_DEVICES=0 python scripts/trainprior.py --dataset $data --exp_name $name-$round
    for i in 1;
    do
       CUDA_VISIBLE_DEVICES=0 python scripts/plan.py --test_planner MCTS_P --dataset $data --exp_name $name-$round --suffix $i --initial_width 16 --b_percent 0.5 --n_expand 4 --n_actions 4 --action_percent 0.5 --pw_alpha 0.1 --mcts_itr 100 --depth 5
    done
  done
done


for data in ${datasets[@]}; do
  for round in {1..3}; do
    python plotting/read_results.py --exp_name $name-$round --dataset $data
  done
  python plotting/read_results.py --exp_name $name --dataset $data
done


