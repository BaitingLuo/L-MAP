export PYTHONPATH=.:$PYTHONPATH

name=T-1
#datasets=(pen-human-v0 pen-cloned-v0 pen-expert-v0)
#(hammer-cloned-v0 hammer-human-v0 door-human-v0 door-cloned-v0 door-expert-v0 relocate-human-v0 relocate-cloned-v0 relocate-expert-v0)

for round in {1..3}; do
  for data in ${datasets[@]}; do
    python scripts/train.py --dataset $data --exp_name $name-$round --tag development --seed $round
    python scripts/trainprior.py --dataset $data --exp_name $name-$round
    for i in {1..20};
    do
       CUDA_VISIBLE_DEVICES=0 python scripts/plan.py --test_planner MCTS_P --dataset $data --exp_name $name-$round --suffix $i --initial_width 16 --b_percent 0.5 --n_expand 4 --n_actions 4 --action_percent 0.5 --pw_alpha 0.1 --mcts_itr 100 --depth 3
    done
  done
done

for data in ${datasets[@]}; do
  for round in {1..3}; do
    python plotting/read_results.py --exp_name $name-$round --dataset $data
  done
  python plotting/read_results.py --exp_name $name --dataset $data
done

