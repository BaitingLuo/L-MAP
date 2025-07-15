# Scalable Decision-Making in Stochastic Environments through Learned Temporal Abstraction (L-MAP)

This is the official repository of L-MAP (Scalable Decision-Making in Stochastic Environments through Learned Temporal Abstraction).

**Read paper on [OpenReview](https://openreview.net/forum?id=pQsllTesiE) |
[Arxiv](https://arxiv.org/abs/2502.21186)**

Additional details are incoming.
If you have any questions about the paper/code, please feel free to reach out to baiting.luo@vanderbilt.edu

## Installation

### Preferred Option
```
# make the env
conda create -n lmap python=3.10 -y

conda activate lmap

# follow instructions on the website to setup Mujoco
https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco

sudo apt-get install build-essential libglew-dev libglfw3-dev mesa-common-dev libgl1-mesa-dev
pip install -r requirements.txt

pip install -e .
```

### Another Option
```
conda env create -f environment.yml
conda activate lmap
pip install -e .
```

## Train and Test the Model


### Loading a custom offline dataset (D4RL format)
*(Skip this section if you only run the official D4RL tasks)*
1. **Prepare your dataset**  
   Any D4RL‑style file works.  
   The stochastic MuJoCo datasets used in our paper can be downloaded here:  
   <https://github.com/marc-rigter/1R2R>

2. **Locate your `d4rl` installation**

   ```bash
   python -m pip show d4rl
   
3. **Manually Load Datasets**

   Locate the file offline_env.py within your d4rl installation path, and modify the method def get_dataset(self, h5path=None). An example script of offline_env.py is provided; simply replace the dataset path accordingly.


### Run training and testing pipeline:
```
./loco.sh
```

### To break it down:
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

### Arguments for Planner
```
python scripts/plan.py --test_planner MCTS_P --dataset $data --exp_name $name-$round --suffix $i --initial_width 16 --b_percent 0.5 --n_expand 4 --n_actions 4 --action_percent 0.5 --pw_alpha 0.1 --mcts_itr 100 --depth 5

--test_planner: Choose between:
    - MCTS_P: Uses a small chunk of tokens to approximate dynamics (`subsampled_sequence_length=2*macro_step`).
    - MCTS_F: Uses the full sequence of tokens to approximate dynamics (`subsampled_sequence_length=n*macro_step`).

--initial_width: Number of macro-actions/action chunks expanded at the root state.
--b_percent: Keep macro-actions at the root with the top `(b_percent × 100)%` returns.
--n_expand: Number of sampled tokens at each chance node.
--n_actions: Number of macro-actions expanded at each leaf node.
--action_percent: Keep macro-actions at leaves with the top `(action_percent × 100)%` returns.
--pw_alpha: Progressive widening alpha (higher values lead to more expansions).
--mcts_itr: Number of MCTS iterations.
--depth: Maximum tree search depth.
```
## Citation

If this code or paper is helpful in your research, please use the following citation:

```
@inproceedings{
luo2025scalable,
title={Scalable Decision-Making in Stochastic Environments through Learned Temporal Abstraction},
author={Baiting Luo and Ava Pettet and Aron Laszka and Abhishek Dubey and Ayan Mukhopadhyay},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=pQsllTesiE}
}
```

## Acknowledgments

Official Trajectory Autoencoding Planner (TAP) Implementation: [https://github.com/ZhengyaoJiang/latentplan](https://github.com/ZhengyaoJiang/latentplan)

Official Trajectory Transformer (TT) Implementation: [https://github.com/jannerm/trajectory-transformer](https://github.com/jannerm/trajectory-transformer)

Official One Risk to Rule Them All (1R2R) Implementation: [https://github.com/marc-rigter/1R2R](https://github.com/marc-rigter/1R2R)