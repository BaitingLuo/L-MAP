__credits__ = ["Rushiv Arora"]

import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class HopperModNoise(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    ### Description

    This environment is based on the work done by Erez, Tassa, and Todorov in
    ["Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks"](http://www.roboticsproceedings.org/rss07/p10.pdf). The environment aims to
    increase the number of independent state and control variables as compared to
    the classic control environments. The hopper is a two-dimensional
    one-legged figure that consist of four main body parts - the torso at the
    top, the thigh in the middle, the leg in the bottom, and a single foot on
    which the entire body rests. The goal is to make hops that move in the
    forward (right) direction by applying torques on the three hinges
    connecting the four body parts.

    ### Action Space
    The action space is a `Box(-1, 1, (3,), float32)`. An action represents the torques applied between *links*

    | Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    |-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
    | 0   | Torque applied on the thigh rotor  | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
    | 1   | Torque applied on the leg rotor    | -1          | 1           | leg_joint                        | hinge | torque (N m) |
    | 3   | Torque applied on the foot rotor   | -1          | 1           | foot_joint                       | hinge | torque (N m) |

    ### Observation Space

    Observations consist of positional values of different body parts of the
    hopper, followed by the velocities of those individual parts
    (their derivatives) with all the positions ordered before all the velocities.

    By default, observations do not include the x-coordinate of the hopper. It may
    be included by passing `exclude_current_positions_from_observation=False` during construction.
    In that case, the observation space will have 12 dimensions where the first dimension
    represents the x-coordinate of the hopper.
    Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the x-coordinate
    will be returned in `info` with key `"x_position"`.

    However, by default, the observation is a `ndarray` with shape `(11,)` where the elements
    correspond to the following:

    | Num | Observation           | Min                  | Max                | Name (in corresponding XML file) | Joint| Unit |
    |-----|-----------------------|----------------------|--------------------|----------------------|--------------------|--------------------|
    | 0   | z-coordinate of the top (height of hopper)       | -Inf                 | Inf                | rootz | slide | position (m) |
    | 1   | angle of the top                                 | -Inf                 | Inf                | rooty | hinge | angle (rad) |
    | 2   | angle of the thigh joint                         | -Inf                 | Inf                | thigh_joint | hinge | angle (rad) |
    | 3   | angle of the leg joint                           | -Inf                 | Inf                | leg_joint | hinge | angle (rad) |
    | 4   | angle of the foot joint                          | -Inf                 | Inf                | foot_joint | hinge | angle (rad) |
    | 5   | velocity of the x-coordinate of the top          | -Inf                 | Inf                | rootx | slide | velocity (m/s) |
    | 6   | velocity of the z-coordinate (height) of the top | -Inf                 | Inf                | rootz | slide | velocity (m/s)  |
    | 7   | angular velocity of the angle of the top         | -Inf                 | Inf                | rooty | hinge | angular velocity (rad/s) |
    | 8   | angular velocity of the thigh hinge              | -Inf                 | Inf                | thigh_joint | hinge | angular velocity (rad/s) |
    | 9   | angular velocity of the leg hinge                | -Inf                 | Inf                | leg_joint | hinge | angular velocity (rad/s) |
    | 10  | angular velocity of the foot hinge               | -Inf                 | Inf                | foot_joint | hinge | angular velocity (rad/s) |

    ### Rewards
    The reward consists of three parts:
    - *healthy_reward*: Every timestep that the hopper is healthy (see definition in section "Episode Termination"), it gets a reward of fixed value `healthy_reward`.
    - *forward_reward*: A reward of hopping forward which is measured
    as *`forward_reward_weight` * (x-coordinate before action - x-coordinate after action)/dt*. *dt* is
    the time between actions and is dependent on the frame_skip parameter
    (fixed to 4), where the frametime is 0.002 - making the
    default *dt = 4 * 0.002 = 0.008*. This reward would be positive if the hopper
    hops forward (positive x direction).
    - *ctrl_cost*: A cost for penalising the hopper if it takes
    actions that are too large. It is measured as *`ctrl_cost_weight` *
    sum(action<sup>2</sup>)* where *`ctrl_cost_weight`* is a parameter set for the
    control and has a default value of 0.001

    The total reward returned is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost* and `info` will also contain the individual reward terms

    ### Starting State
    All observations start in state
    (0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) with a uniform noise
     in the range of [-`reset_noise_scale`, `reset_noise_scale`] added to the values for stochasticity.

    ### Episode Termination
    The hopper is said to be unhealthy if any of the following happens:

    1. An element of `observation[1:]` (if  `exclude_current_positions_from_observation=True`, else `observation[2:]`) is no longer contained in the closed interval specified by the argument `healthy_state_range`
    2. The height of the hopper (`observation[0]` if  `exclude_current_positions_from_observation=True`, else `observation[1]`) is no longer contained in the closed interval specified by the argument `healthy_z_range` (usually meaning that it has fallen)
    3. The angle (`observation[1]` if  `exclude_current_positions_from_observation=True`, else `observation[2]`) is no longer contained in the closed interval specified by the argument `healthy_angle_range`

    If `terminate_when_unhealthy=True` is passed during construction (which is the default),
    the episode terminates when any of the following happens:

    1. The episode duration reaches a 1000 timesteps
    2. The hopper is unhealthy

    If `terminate_when_unhealthy=False` is passed, the episode is terminated only when 1000 timesteps are exceeded.

    ### Arguments

    No additional arguments are currently supported in v2 and lower.

    ```
    env = gym.make('Hopper-v2')
    ```

    v3 and beyond take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.

    ```
    env = gym.make('Hopper-v3', ctrl_cost_weight=0.1, ....)
    ```

    | Parameter               | Type       | Default        |Description                    |
    |-------------------------|------------|----------------|-------------------------------|
    | `xml_file`              | **str**    | `"hopper.xml"` | Path to a MuJoCo model |
    | `forward_reward_weight` | **float**  | `1.0`          | Weight for *forward_reward* term (see section on reward) |
    | `ctrl_cost_weight`      | **float**  | `0.001`        | Weight for *ctrl_cost* reward (see section on reward) |
    | `healthy_reward`        | **float**  | `1`            | Constant reward given if the ant is "healthy" after timestep |
    | `terminate_when_unhealthy` | **bool**| `True`         | If true, issue a done signal if the hopper is no longer healthy |
    | `healthy_state_range`   | **tuple**  | `(-100, 100)`  | The elements of `observation[1:]` (if  `exclude_current_positions_from_observation=True`, else `observation[2:]`) must be in this range for the hopper to be considered healthy |
    | `healthy_z_range`       | **tuple**  | `(0.7, float("inf"))`    | The z-coordinate must be in this range for the hopper to be considered healthy |
    | `healthy_angle_range`   | **tuple**  | `(-0.2, 0.2)`   | The angle given by `observation[1]` (if  `exclude_current_positions_from_observation=True`, else `observation[2]`) must be in this range for the hopper to be considered healthy |
    | `reset_noise_scale`     | **float**  | `5e-3`         | Scale of random perturbations of initial position and velocity (see section on Starting State) |
    | `exclude_current_positions_from_observation`| **bool** | `True`| Whether or not to omit the x-coordinate from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies |


    ### Version History

    * v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
    * v2: All continuous control environments now use mujoco_py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release (1.0.0)
    """

    def __init__(
        self,
        xml_file="hopper.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_state_range=(-100.0, 100.0),
        healthy_z_range=(0.7, float("inf")),
        healthy_angle_range=(-0.2, 0.2),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
        noise_scale=2.5,
        noise_step=0.1,
    ):
        self.noise_scale = noise_scale
        self.noise_step = noise_step
        self.force_body_index = 1
        self.n_bodies = 5
        self.current_force = np.zeros((self.n_bodies, 6))
        self.name = 'hopper-medium-expert-v2'
        self.max_episode_steps = 1000

        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        #self.seed = seed
        #self.np_random = np.random.RandomState(self.seed)
        mujoco_env.x.__init__(self, xml_file, 4)

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.sim.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = np.clip(self.sim.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        step_scale = self.noise_scale * self.noise_step
        new_force = self.current_force[self.force_body_index, 0] + np.random.uniform(-1, 1) * step_scale
        #new_force = self.current_force[self.force_body_index, 0] + self.np_random.uniform(-1, 1) * step_scale
        new_force = np.clip(new_force, -self.noise_scale, self.noise_scale)
        self.current_force[:, 0] = new_force
        self.data.xfrc_applied[:] = self.current_force

        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
        }

        #print("Masses")
        #print(self.sim.model.body_mass)

        return observation, reward, done, info

    def reset_model(self):
        self.current_force = np.zeros((self.n_bodies, 6))
        self.data.xfrc_applied[:] = self.current_force

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
