export PYTHONPATH=.:$PYTHONPATH

name=T-1
antdatasets=(antmaze-large-diverse-v0)
for data in ${antdatasets[@]}; do
  for round in {1..3}; do
    python scripts/train.py --dataset $data --exp_name $name-$round --tag development --seed $round
    python scripts/trainprior.py --dataset $data --exp_name $name-$round
    for i in {1..20};
    do
       CUDA_VISIBLE_DEVICES=0 python scripts/plan.py --test_planner beam_prior --dataset $data --exp_name $name-$round --suffix $i --beam_width 16 --b_percent 0.5 --n_expand 2 --n_actions 4 --action_percent 0.5 --pw_alpha 0 --mcts_itr 200
    done
  done
done

for data in ${antdatasets[@]}; do
  for round in {1..3}; do
    python plotting/read_results.py --exp_name $name-$round --dataset $data
  done
  python plotting/read_results.py --exp_name $name --dataset $data
done


