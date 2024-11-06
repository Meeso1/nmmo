import numpy as np
import torch
import torch.nn as nn

import pufferlib
import pufferlib.models
import pufferlib.emulation


class Recurrent(pufferlib.models.RecurrentWrapper):
    def __init__(self, env, policy, input_size=256, hidden_size=256, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)


class Policy(pufferlib.models.Policy):
    def __init__(self, env, input_size=256, hidden_size=256, task_size=2048):
        '''Pure PyTorch base policy
    
        This spec allows PufferLib to repackage your policy
        for compatibility with RL frameworks

        encode_observations -> decode_actions is PufferLib's equivalent of PyTorch forward
        This structure provides additional flexibility for us to include an LSTM
        between the encoder and decoder.

        To port a policy to PufferLib, simply put everything from forward() before the
        recurrent cell (or, if no recurrent cell, everything before the action head)
        into encode_observations and put everything after into decode_actions.

        You can delete the recurrent cell from forward(). PufferLib handles this for you
        with its framework-specific wrappers. Since each frameworks treats temporal data a bit
        differently, this approach lets you write a single PyTorch network for multiple frameworks.

        Specify the value function in critic(). This is a single value for each batch element.
        It is called on the output of the recurrent cell (or, if no recurrent cell,
        the output of encode_observations)
        '''
        super().__init__(env)
        
        self.encoder = nn.Linear(np.prod(self.observation_space.shape), hidden_size)

        if self.is_multidiscrete:
            self.decoders = nn.ModuleList([nn.Linear(hidden_size, n)
                for n in self.action_space.nvec])
        else:
            self.decoder = nn.Linear(hidden_size, self.action_space.n)

        self.value_head = nn.Linear(hidden_size, 1)

    def encode_observations(self, flat_observations):
        '''
        Encodes a batch of observations into hidden states

        Call pufferlib.emulation.unpack_batched_obs at the start of this
        function to unflatten observations to their original structured form:

        observations = pufferlib.emulation.unpack_batched_obs(
            env_outputs, self.unflatten_context)
 
        Args:
            flat_observations: A tensor of shape (batch, ..., obs_size)

        Returns:
            hidden: Tensor of (batch, ..., hidden_size)
            lookup: Tensor of (batch, ...) that can be used to return additional embeddings
        '''
        hidden = flat_observations.reshape(flat_observations.shape[0], -1).float()
        hidden = torch.relu(self.encoder(hidden))
        #return torch.from_numpy(np.random.uniform(0, 1, size=hidden.shape)), None
        return hidden, None


    def decode_actions(self, flat_hidden, lookup):
        '''
        Decodes a batch of hidden states into multidiscrete actions

        Args:
            flat_hidden: Tensor of (batch, ..., hidden_size)
            lookup: Tensor of (batch, ...), if returned by encode_observations

        Returns:
            actions: Tensor of (batch, ..., action_size)
            value: Tensor of (batch, ...)

        actions is a concatenated tensor of logits for each action space dimension.
        It should be of shape (batch, ..., sum(action_space.nvec))
        '''
        value = self.value_head(flat_hidden)
        #value = torch.rand((1,)).cuda() #torch.from_numpy(np.random.uniform(0, 1, size=(1,)))

        if self.is_multidiscrete:
            actions = [decoder(flat_hidden) for decoder in self.decoders]
            #actions = [torch.rand((*flat_hidden.shape[:-1], n)).cuda() for n in self.action_space.nvec]
            return actions, value

        action = self.decoder(flat_hidden)
        #action = torch.rand((self.action_space.n,)).cuda()
        
        return action, value
