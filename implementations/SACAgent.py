"""
Implements a SAC agent using the new base class and networks from SACNetworks.py.
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Any

from implementations.SACBase import SACBase
from implementations.SACNetworks import (
    SACPolicyNetwork,
    SACCriticNetwork
)
from implementations.observations_to_inputs import observations_to_inputs_simplier
from implementations.SimplierNetwork import SimplierNetwork
# ...existing code for Jar, Observations, etc. if needed...


class SACAgent(SACBase):
    """
    Soft Actor-Critic agent that handles discrete or continuous actions,
    using the provided observation encoding logic.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        alpha: float = 0.2,
        gamma: float = 0.99,
        polyak: float = 0.995,
        lr: float = 3e-4
    ):
        """
        Args:
            state_dim: Dimension of the flattened/processed observation.
            action_dim: Number of action dimensions.
            latent_dim: Dimensionality of the encoded latent vector.
            alpha: Entropy coefficient for SAC.
            gamma: Discount factor.
            polyak: Polyak averaging factor for target networks.
            lr: Learning rate.
        """
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = alpha
        self.polyak = polyak

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.action_types = SimplierNetwork.action_types()

        # Networks
        self.policy = SACPolicyNetwork(latent_dim, action_dim).to(self.device)

        # Two Q networks (and their targets)
        self.q1 = SACCriticNetwork(latent_dim, action_dim).to(self.device)
        self.q2 = SACCriticNetwork(latent_dim, action_dim).to(self.device)
        self.q1_target = SACCriticNetwork(latent_dim, action_dim).to(self.device)
        self.q2_target = SACCriticNetwork(latent_dim, action_dim).to(self.device)

        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=lr)

    def select_actions(self, states: dict[int, any]) -> dict[int, any]:
        """
        Encodes states, samples actions from policy distribution.
        """
        self.policy.eval()
        actions_out = {}
        for agent_id, obs in states.items():
            # Convert obs to tile_data, self_data, plus masks for Move/AttackStyle
            tile_data, self_data, move_mask, style_mask = observations_to_inputs_simplier(obs, self.device)
            with torch.no_grad():
                mean, log_std = self.policy(tile_data, self_data)
                std = log_std.exp()
                z = torch.randn_like(mean)
                raw_action = mean + std * z
                # Keep impossible action masking for Move & AttackStyle as done in PPO
                # (Exact implementation depends on how your environment uses them.)
                masked_move = move_mask * raw_action[..., :5]
                masked_style = style_mask * raw_action[..., 5:8]
                # ...existing or simplified logic for AttackTargetPos, AttackOrNot...
                # For demonstration, final action might just be raw_action or partial
                final_action = torch.tanh(raw_action)
            actions_out[agent_id] = {
                "action_tensor": final_action,
                # For an environment that needs discrete or dimension-slicing, adapt here
                "raw_policy_outputs": (mean, log_std)
            }
        return actions_out

    def train_step(
        self,
        states: dict[int, list[any]],
        actions: dict[int, list[any]],
        rewards: dict[int, list[float]],
        dones: dict[int, list[bool]]
    ) -> None:
        """
        Runs one SAC update using gathered transitions (simplified approach).
        """
        # This example does not implement a replay buffer: using on-policy data for demonstration.
        # In practice, use an experience replay for SAC.
        all_obs = []
        all_acts = []
        all_rews = []
        all_next_obs = []
        all_dones = []

        # Gather transitions in a single list for simplicity
        for agent_id in states:
            st_list = states[agent_id]
            ac_list = actions[agent_id]
            rw_list = rewards[agent_id]
            dn_list = dones[agent_id]
            for i in range(len(st_list) - 1):
                all_obs.append(st_list[i])
                all_acts.append(ac_list[i]["action_tensor"])
                all_rews.append(rw_list[i])
                all_dones.append(dn_list[i])
                all_next_obs.append(st_list[i+1])

        if not all_obs:
            return

        # Convert to tensors
        obs_t = []
        next_obs_t = []
        for o, no in zip(all_obs, all_next_obs):
            obs_t.append(observations_to_inputs_simplier(o, self.device)[0])
            next_obs_t.append(observations_to_inputs_simplier(no, self.device)[0])

        obs_t = torch.cat(obs_t, dim=0)
        next_obs_t = torch.cat(next_obs_t, dim=0)
        act_t = torch.cat(all_acts, dim=0).to(self.device)
        rew_t = torch.tensor(all_rews, device=self.device).unsqueeze(-1)
        done_t = torch.tensor(all_dones, device=self.device).unsqueeze(-1)

        # 1) Encode states
        latent = self.encoder(obs_t)
        next_latent = self.encoder(next_obs_t)

        # 2) Compute next action and Q target
        with torch.no_grad():
            next_mean, next_log_std = self.policy(next_latent)
            next_std = next_log_std.exp()
            eps = torch.randn_like(next_mean)
            next_raw = next_mean + next_std * eps
            next_action = torch.tanh(next_raw)

            # Compute next Q
            next_q1 = self.q1_target(next_latent, next_action)
            next_q2 = self.q2_target(next_latent, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * self._log_prob_tanh(next_raw, next_std)
            target = rew_t + self.gamma * (1 - done_t) * next_q

        # 3) Update Q networks
        q1_val = self.q1(latent, act_t)
        q2_val = self.q2(latent, act_t)
        q1_loss = F.mse_loss(q1_val, target)
        q2_loss = F.mse_loss(q2_val, target)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        # 4) Update policy
        current_mean, current_log_std = self.policy(latent)
        current_std = current_log_std.exp()
        z = torch.randn_like(current_mean)
        pi_raw = current_mean + current_std * z
        pi_action = torch.tanh(pi_raw)
        q1_pi = self.q1(latent, pi_action)
        q2_pi = self.q2(latent, pi_action)
        q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * self._log_prob_tanh(pi_raw, current_std) - q_pi).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        # 5) Update encoder if needed (here we do a shared representation approach)
        # (Optional advanced approach: if separate, do that here)

        # 6) Polyak update targets
        with torch.no_grad():
            for p, p_targ in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def save(self, name: str | None = None) -> None:
        # ...existing code for saving with Jar or similar...
        pass

    @classmethod
    def load(cls, name: str | None = None) -> 'SACAgent':
        # ...existing code for loading with Jar or similar...
        pass

    def _log_prob_tanh(self, raw: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Approximate log probability of a Tanh-squashed Gaussian sample.
        """
        # This is a simplified version ignoring some constant terms.
        log_prob_gaussian = -0.5 * ((raw / (std + 1e-7))**2 + 2 * torch.log(std + 1e-7))
        # correction for tanh
        log_prob_tanh = torch.log(1 - torch.tanh(raw).pow(2) + 1e-7)
        return (log_prob_gaussian.sum(dim=-1, keepdim=True) - log_prob_tanh.sum(dim=-1, keepdim=True))