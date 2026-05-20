""" Wrapper for struct_env respecting the interface of PyMARL. """

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    print("")
    print("")
    print("ModuleNotFoundError")
    print("You need to install torch to use the wrapper as in the repository.")
    print("")
    print("")
    raise ModuleNotFoundError

from imp_marl.environments.struct_zayas import StructZayas
# from imp_marl.imp_wrappers.pymarl_wrapper.MultiAgentEnv import MultiAgentEnv
from .multiagentenv import MultiAgentEnv


class StructZayasWrapper(MultiAgentEnv):
    """
    Wrapper for StructZayas respecting the interface of EPyMARL.

    It manipulates an imp_env to create all inputs for EPyMARL agents.
    """

    def __init__(
        self,
        n_comp: int = 22,
        discount_reward: float = 1.0,
        state_obs: bool = True,
        state_d_rate: bool = False,
        obs_d_rate: bool = False,
        obs_multiple: bool = False,
        obs_all_d_rate: bool = False,
        campaign_cost: bool = False,
        map_name: str = None,
        seed=None,
        common_reward: bool = True,
        **kwargs,
    ):
        """
        Initialise based on the full configuration.

        Args:
            n_comp: (int) Number of structure
            custom_param: (dict)
                struct: Number of structure required
                        {"k_comp": int} for k_comp out of n_comp
                        Default is None, meaning k_comp=n_comp-1
                 owf: Number of levels per wind turbine
                        {"lev": int}
                        Default is 3
            discount_reward: (float) Discount factor [0,1[
            state_obs: (bool) State contains the concatenation of obs
            state_d_rate: (bool) State contains the concatenation of drate
            state_alphas: (bool) State contains the concatenation of alpha
            obs_d_rate: (bool) Obs contains the drate of the agent
            obs_multiple: (bool) Obs contains the concatenation of all obs
            obs_all_d_rate: (bool) Obs contains the concatenation of all drate
            obs_alphas: (bool) Obs contains the alphas
            env_correlation: (bool) env_correlation: True=correlated, False=uncorrelated
            campaign_cost: (bool) campaign_cost = True=campaign cost taken into account
            seed: (int) seed for the random number generator
        """
        # Check struct type and default values

        assert (
            isinstance(state_obs, bool)
            and isinstance(state_d_rate, bool)
            and isinstance(obs_d_rate, bool)
            and isinstance(obs_multiple, bool)
            and isinstance(obs_all_d_rate, bool)
            and isinstance(campaign_cost, bool)
        ), "Error in env parameters"
        assert 0 <= discount_reward <= 1, "Error in discount_reward"
        assert not (obs_d_rate and obs_all_d_rate), "Error in env parameters"
        assert state_obs or state_d_rate, "Error in env parameters"

        self.n_comp = n_comp
        self.discount_reward = discount_reward
        self.state_obs = state_obs
        self.state_d_rate = state_d_rate
        self.obs_d_rate = obs_d_rate
        self.obs_multiple = obs_multiple
        self.obs_all_drate = obs_all_d_rate
        self.common_reward = common_reward
        self.campaign_cost = campaign_cost
        self.map_name = map_name
        self._seed = seed

        
        self.config = {
            "n_comp": n_comp,
            "discount_reward": discount_reward,
            "campaign_cost": campaign_cost,
        }
        
        self.struct_env = StructZayas(self.config)
        self.n_agents = self.struct_env.n_comp

        self.episode_limit = self.struct_env.ep_length
        self.agent_list = self.struct_env.agent_list
        self.n_actions = self.struct_env.actions_per_agent

        self.action_histogram = {"action_" + str(k): 0 for k in range(self.n_actions)}

        self.unit_dim = self.get_unit_dim()  # Qplex requirement

    def update_action_histogram(self, actions):
        """
        Update the action histogram for logging.

        Args:
            actions: list of actions
        """
        for k, action in zip(self.struct_env.agent_list, actions):
            if type(action) is torch.Tensor:
                action_str = str(action.cpu().numpy())
            else:
                action_str = str(action)
            self.action_histogram["action_" + action_str] += 1

    def step(self, actions):
        """
        Ask to run a step in the environment.

        Args:
            actions: list of actions

        Returns:
            rewards: list of rewards
            done: True if the episode is finished
            info: dict of info for logging
        """
        truncated = False

        self.update_action_histogram(actions)

        action_dict = {
            k: action for k, action in zip(self.struct_env.agent_list, actions)
        }

        ## sample instead of computing the VoI in closed form (~ 25 samples)
        ## inspector networks ... (same critic: b'...)
        _, rewards, done, _ = self.struct_env.step(action_dict)

        if self.common_reward:
            reward = rewards[self.struct_env.agent_list[0]]
        else:
            reward = [rewards[a] for a in self.struct_env.agent_list]

        info = {}
        if done:
            for k in self.action_histogram:
                self.action_histogram[k] /= self.episode_limit * self.n_agents
            info = self.action_histogram

        return self.get_obs(), reward, done, truncated, info

    def get_obs(self):
        """Returns all agent observations in a list."""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_unit_dim(self):
        """Returns the dimension of the unit observation used by QPLEX."""
        return len(self.all_obs_from_struct_env()) // self.n_agents

    def get_obs_agent(self, agent_id: int):
        """
        Returns observation for agent_id

        Args:
            agent_id: id of the agent (int in range(self.n_agents)).
        """
        agent_name = self.struct_env.agent_list[agent_id]

        if self.obs_multiple:
            obs = self.all_obs_from_struct_env()
        else:
            obs = self.struct_env.observations[agent_name]

        if self.obs_d_rate:
            obs = np.append(obs, self.get_normalized_drate()[agent_id])

        if self.obs_all_drate:
            obs = np.append(obs, self.get_normalized_drate())

        return obs

    def get_obs_size(self):
        """Returns the size of the observation."""
        return len(self.get_obs_agent(0))

    def get_normalized_drate(self):
        """Returns the normalized d_rate."""
        return self.struct_env.d_rate / self.struct_env.ep_length

    def all_obs_from_struct_env(self):
        """Returns all observations concatenated in a single vector."""
        # Concatenate all obs with a single time.
        idx = 0
        obs = None
        for k, v in self.struct_env.observations.items():
            if idx == 0:
                obs = v
                idx = 1
            else:
                obs = np.append(obs, v)
        return obs

    def get_state(self):
        """Returns the state of the environment."""
        state = []
        if self.state_obs:
            state = np.append(state, self.all_obs_from_struct_env())
        if self.state_d_rate:
            state = np.append(state, self.get_normalized_drate())
        return state

    def get_state_size(self):
        """Returns the shape of the state"""
        return len(self.get_state())

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """
        Returns the available actions for agent_id.

        Args:
            agent_id: id of the agent (int in range(self.n_agents)).
        """
        return [1] * self.n_actions

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.struct_env.actions_per_agent

    def reset(self, seed=None, options=None):
        """Returns initial observations and states."""

        if seed is not None:
            self._seed = seed

        self.action_histogram = {"action_" + str(k): 0 for k in range(self.n_actions)}

        self.struct_env.reset()

        info = {}

        return self.get_obs(), info

    def render(self):
        """See base class."""
        pass

    def close(self):
        """See base class."""
        pass

    def seed(self):
        """Returns the random seed"""
        return self._seed

    def save_replay(self):
        """See base class."""
        pass

    def get_stats(self):
        """See base class."""
        return {}
