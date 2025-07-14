```
conda env create -f environment.yml
conda activate tap
pip install -e .
```

To break it down:
1. train the encoder and decoder:
```
python scripts/train.py --dataset $data --exp_name $name-$round --tag development --seed $round
```
2. train the prior:
```
python scripts/trainprior.py --dataset $data --exp_name $name-$round
```
3. evaluate the trained models:
```
for i in {1..20};
do
    python scripts/plan.py --test_planner mcts_prior --dataset $data --exp_name $name-$round --suffix $i --beam_width 16 --b_percent 0.5 --n_expand 4 --n_actions 4 --action_percent 0.5 --pw_alpha 0 --mcts_itr 100
done 
```
4. report the results:
```
python plotting/read_results.py --exp_name $name --dataset $data
```

