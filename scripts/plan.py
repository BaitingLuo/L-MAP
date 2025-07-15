import json
from os.path import join
import os
import numpy as np
import time
import latentplan.utils as utils
import latentplan.datasets as datasets
from latentplan.search import (
    enumerate_all,
    make_prefix,
    extract_actions,
    update_context,
)
from mujoco_stochastic import hopper_high_noise
from mujoco_stochastic import walker2d_high_noise
from mujoco_stochastic import hopper_mod_noise
from mujoco_stochastic import walker2d_mod_noise
from currency_exchange import currency_exchange
from hiv_treatment import hiv_treatment
import matplotlib.pyplot as plt

from latentplan.search import MCTS_P, MCTS_F


class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
    config: str = 'config.vqvae'

#######################
######## setup ########
#######################

args = Parser().parse_args('plan')
args.nb_samples = int(args.nb_samples)
args.n_expand = int(args.n_expand)
args.initial_width = int(args.initial_width)
args.n_actions = int(args.n_actions)
args.b_percent = float(args.b_percent)
args.action_percent = float(args.action_percent)
args.pw_alpha = float(args.pw_alpha)
args.mcts_itr = int(args.mcts_itr)
args.depth = int(args.depth)
args.horizon = int(args.horizon)
args.rounds = int(args.rounds)
args.logbase = os.path.expanduser(args.logbase)
args.savepath = os.path.expanduser(args.savepath)
args.uniform = bool(args.uniform)


try:
    args.prob_weight = float(args.prob_weight)
except:
    args.prob_weight = 5e2

print("check save path", args.savepath)
#######################
####### models ########
#######################
#enforce the certain model
#args.exp_name = 'T-1-1'
#env = datasets.load_environment(args.dataset)
#env.seed(1*int(args.suffix)*100)

d4rl_env = datasets.load_environment(args.dataset)
if 'hopper' in args.dataset:
    env = hopper_high_noise.HopperHighNoise()
elif 'walker2d' in args.dataset:
    env = walker2d_high_noise.Walker2DHighNoise()
else:
    env = d4rl_env
#env = d4rl_env
#env = currency_exchange.CurrencyExchange()
#env = hiv_treatment.HIVTreatment()

dataset = utils.load_from_config(args.logbase, args.dataset, args.exp_name,
        'data_config.pkl')


gpt, gpt_epoch = utils.load_model(args.logbase, args.dataset, args.exp_name,
        epoch=args.gpt_epoch, device=args.device)


if args.test_planner in ["sample_prior", "sample_prior_tree", "beam_prior", "beam_mimic", "beam_uniform", "MCTS_P"]:
    prior, _ = utils.load_prior_model(args.logbase, args.dataset, args.exp_name,
                                      epoch=args.gpt_epoch, device=args.device)

gpt.set_padding_vector(dataset.normalize_joined_single(np.zeros(gpt.transition_dim-1)))
#######################
####### dataset #######
#######################

#if args.task_type == "locomotion":
#    renderer = utils.make_renderer(args)
timer = utils.timer.Timer()

discount = dataset.discount
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

preprocess_fn = datasets.get_preprocess_fn(env.name)
#######################
###### main loop ######
#######################
REWARD_DIM = VALUE_DIM = 1
transition_dim = observation_dim + action_dim + REWARD_DIM + VALUE_DIM
observation = env.reset()
total_reward = 0
discount_return = 0

if "antmaze" in env.name:
    if dataset.disable_goal:
        observation = np.concatenate([observation, np.zeros([2], dtype=np.float32)])
        rollout = [np.concatenate([env.state_vector().copy(), np.zeros([2], dtype=np.float32)])]
    else:
        observation = np.concatenate([observation, env.target_goal])
        rollout = [np.concatenate([env.state_vector().copy(), env.target_goal])]

context = []
mses = []

T = env.max_episode_steps
print(T)
frames = []
gpt.eval()
total_high_loss_count = 0
for t in range(T):
    observation = preprocess_fn(observation)
    if dataset.normalized_raw:
        observation = dataset.normalize_states(observation)

    if "antmaze" in env.name:
        if dataset.disable_goal:
            state = np.concatenate([state, np.zeros([2], dtype=np.float32)])
        else:
            state = np.concatenate([state, env.target_goal])

    if t % args.plan_freq == 0:
        ## concatenate previous transitions and current observations to input to model
        prefix = make_prefix(observation, transition_dim, device=args.device)[-1, -1, None, None]
        ## sample sequence from model beginning with `prefix`
        if args.test_planner == 'MCTS_P':
            prior.eval()
            start_time = time.time()
            sequence = MCTS_P(prior, gpt, prefix,
                              initial_width=args.initial_width,
                              n_expand=args.n_expand,
                              n_action=args.n_actions,
                              b_percent= args.b_percent,
                              action_percent = args.action_percent,
                              pw_alpha = args.pw_alpha,
                              mcts_itr = args.mcts_itr,
                              macro_step=args.macro_step,
                              depth=args.depth,
                              )

            end_time = time.time()  # End timer
            # Calculate the elapsed time in seconds
            elapsed_time = end_time - start_time
            print("decision time:", elapsed_time)
        elif args.test_planner == 'MCTS_F':
            prior.eval()
            start_time = time.time()
            sequence = MCTS_F(prior, gpt, prefix,
                              initial_width=args.initial_width,
                              n_expand=args.n_expand,
                              n_action=args.n_actions,
                              b_percent= args.b_percent,
                              action_percent = args.action_percent,
                              pw_alpha = args.pw_alpha,
                              mcts_itr = args.mcts_itr,
                              macro_step=args.macro_step,
                              depth=args.depth,
                              )
            end_time = time.time()  # End timer
            # Calculate the elapsed time in seconds
            elapsed_time = end_time - start_time
            print("decision time:", elapsed_time)

    else:
        sequence = sequence[1:]

    if t == 0:
        first_value = float(dataset.denormalize_values(sequence[0,-2]))
        first_search_value = float(dataset.denormalize_values(sequence[-1, -2]))

    ## [ horizon x transition_dim ] convert sampled tokens to continuous latentplan
    sequence_recon = sequence

    ## [ action_dim ] index into sampled latentplan to grab first action

    feature_dim = dataset.observation_dim
    action = extract_actions(sequence_recon, feature_dim, action_dim, t=0)
    if dataset.normalized_raw:
        action = dataset.denormalize_actions(action)
        sequence_recon = dataset.denormalize_joined(sequence_recon)

    ## execute action in environment
    next_observation, reward, terminal, _ = env.step(action)
    if "antmaze" in env.name:
        if dataset.disable_goal:
            next_observation = np.concatenate([next_observation, np.zeros([2], dtype=np.float32)])
        else:
            next_observation = np.concatenate([next_observation, env.target_goal])


    ## update return
    total_reward += reward
    discount_return += reward* discount**(t)
    score = d4rl_env.get_normalized_score(total_reward)
    context = update_context(observation, action, reward, device=args.device)

    print(
        f'[ plan ] t: {t} / {T} | r: {reward:.2f} | R: {total_reward:.2f} | score: {score:.4f} | '
        f'time: {timer():.4f} | {args.dataset} | {args.exp_name} | {args.suffix}\n'
    )
    ########render here
    #renderer.render_matplotlib(state.copy())
    #frames.append(frame)
    if t % args.vis_freq == 0 or terminal or t == T-1:
        if not os.path.exists(args.savepath):
            os.makedirs(args.savepath)
        # ffmpeg will report a error in some setup
        # if "antmaze" in env.name or "medium" in env.name:
        #     _, mse = renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'),
        #                      sequence_recon, state)
        # else:
        #     #print(args.savepath, t, sequence_recon, state)
        #     _, mse = renderer.render_real(join(args.savepath, f'{t}_plan.mp4'),
        #                                   sequence_recon, state)


        ## save rollout thus far
        #renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=30)
        # if not terminal:
        #     mses.append(mse)

    if terminal: break

    observation = next_observation
## save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal, 'gpt_epoch': gpt_epoch,
             'first_value': first_value, 'first_search_value': first_search_value, 'discount_return': discount_return,
             'prediction_error': np.mean(mses)}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
