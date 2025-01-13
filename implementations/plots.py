import os
from matplotlib import pyplot as plt
import numpy as np
from implementations.jar import Jar


def get_entropies_from_save(save_name: str) -> dict[str, list[float]]:     
	history = Jar("saves").get(save_name)
	entropies_per_episode = [[entropies 
                           		for step in ep[4] 
                           		for entropies in step.values()]
						    for ep in history]
 
	means_per_episode_per_type = [{type: np.mean([entropies[type] for entropies in ep]) 
                                  	for type in ep[0].keys()} 
                                 for ep in entropies_per_episode]
 
	means_per_type_per_episode = {type: [ep[type] for ep in means_per_episode_per_type]
                                  for type in means_per_episode_per_type[0].keys()}
	return means_per_type_per_episode


def _show_or_save(save_as: str | None = None) -> None:
    if save_as:
        os.makedirs("plots/plots", exist_ok=True)
        plt.savefig(f"plots/plots/{save_as}.png")
        plt.close()
    else:
        plt.show()

def plot_losses(
    losses: list[tuple[list[float], list[float]]],
    window: int = 500,
    save_as: str | None = None
) -> None:
    actor_losses = [l for loss in losses for l in loss[0]]
    critic_losses = [l for loss in losses for l in loss[1]]
        
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # Set y-axis limit based on first 100 values
    y_max = max(actor_losses[:100]) * 3
    peaks = [(i, val) for i, val in enumerate(actor_losses) if val > y_max]
    
    # Plot regular actor losses with cap
    capped_losses = [min(l, y_max) for l in actor_losses]
    ax1.plot(capped_losses, label="Actor Loss", color='blue', alpha=0.4)
    
    # Plot smoothed line
    actor_losses_smooth = np.convolve(actor_losses, np.ones(window)/window, mode='valid')
    actor_losses_std = np.array([np.std(actor_losses[max(0, i-window):i+1]) 
                                for i in range(window-1, len(actor_losses))])
    
    ax1.plot(range(window-1, len(capped_losses)), actor_losses_smooth, 
             label=f"Running Mean (window={window})", color='red')
    ax1.fill_between(range(window-1, len(capped_losses)), 
                     actor_losses_smooth - actor_losses_std,
                     actor_losses_smooth + actor_losses_std,
                     alpha=0.2, color='red', label='Standard Deviation')
    
    # Mark peaks that exceed the cap
    if peaks:
        peak_x, _ = zip(*peaks)
        ax1.scatter(peak_x, [y_max] * len(peaks), marker='^', color='red', s=100, 
                   label=f'Peaks > {y_max:.2f}')
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Actor Loss Over Time")
    ax1.legend()
    
    ax2.semilogy(critic_losses, label="Critic Loss", color='blue', alpha=0.4)
    critic_losses_smooth = np.convolve(critic_losses, np.ones(window)/window, mode='valid')
    ax2.semilogy(range(window-1, len(critic_losses)), critic_losses_smooth, 
                 label=f"Running Mean (window={window})", color='red')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (log scale)")
    ax2.set_title("Critic Loss Over Time")
    ax2.legend()
    
    plt.tight_layout()
    _show_or_save(save_as)
  
    
def plot_losses_from_save(agent_name: str, window: int = 500, save_as: str | None = None) -> None:
    history = Jar("saves").get(agent_name)
    losses = [episode[2] for episode in history]
    plot_losses(losses, window, save_as)


def plot_rewards(
    rewards: list[dict[int, float]],
    random_agent_reward: float | None = None,
    window: int = 50,
    save_as: str | None = None
) -> None:
    num_agents = len(rewards[0])
    avg_rewards = [np.mean([r for r in reward.values()]) for reward in rewards]
    max_rewards = [np.max([r for r in reward.values()]) for reward in rewards]
    min_rewards = [np.min([r for r in reward.values()]) for reward in rewards]
    ninetieth_percentile_rewards = [np.percentile([r for r in reward.values()], 90) for reward in rewards] if num_agents > 30 else None

    _, ax = plt.subplots(figsize=(10, 6))
    
    if ninetieth_percentile_rewards is not None:
        ax.plot(ninetieth_percentile_rewards, label="90th Percentile Reward", color='purple', alpha=0.4)
        
    ax.plot(avg_rewards, label="Average Reward", color='red', alpha=0.4)
    ax.plot(max_rewards, label="Max Reward", color='pink', alpha=0.8)
    ax.plot(min_rewards, label="Min Reward", color='green', alpha=0.4)
    
    if random_agent_reward is not None:
        ax.axhline(y=random_agent_reward, label="Random Agent Reward", color='black', linestyle='--')
    
    avg_rewards_smooth = np.convolve(avg_rewards, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(avg_rewards)), avg_rewards_smooth, label=f"Running Mean (window={window})", color='blue')

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Rewards Over Time")
    ax.legend()
    _show_or_save(save_as)


def plot_rewards_from_save(
    agent_name: str, 
    window: int = 50, 
    random_agent_reward: float | None = None,
    save_as: str | None = None
) -> None:
    history = Jar("saves").get(agent_name)
    rewards = [episode[1] for episode in history]
    plot_rewards(rewards, random_agent_reward, window, save_as)


def compare_rewards(
    first_config_data: tuple[str, list[dict[int, float]]],
    second_config_data: tuple[str, list[dict[int, float]]],
    window: int = 50,
    save_as: str | None = None
) -> None:
    _, ax = plt.subplots(figsize=(10, 6))
    first_config, first_rewards = first_config_data
    second_config, second_rewards = second_config_data
    
    # Calculate percentiles and means for first config
    avg_first_rewards = [np.mean([r for r in reward.values()]) for reward in first_rewards]
    top_10_first = [np.percentile([r for r in reward.values()], 90) for reward in first_rewards]
    bottom_10_first = [np.percentile([r for r in reward.values()], 10) for reward in first_rewards]
    
    # Calculate percentiles and means for second config
    avg_second_rewards = [np.mean([r for r in reward.values()]) for reward in second_rewards]
    top_10_second = [np.percentile([r for r in reward.values()], 90) for reward in second_rewards]
    bottom_10_second = [np.percentile([r for r in reward.values()], 10) for reward in second_rewards]
    
    # Calculate smoothed values
    first_rewards_smooth = np.convolve(avg_first_rewards, np.ones(window)/window, mode='valid')
    first_top_smooth = np.convolve(top_10_first, np.ones(window)/window, mode='valid')
    first_bottom_smooth = np.convolve(bottom_10_first, np.ones(window)/window, mode='valid')
    
    second_rewards_smooth = np.convolve(avg_second_rewards, np.ones(window)/window, mode='valid')
    second_top_smooth = np.convolve(top_10_second, np.ones(window)/window, mode='valid')
    second_bottom_smooth = np.convolve(bottom_10_second, np.ones(window)/window, mode='valid')
    
    # Plot main lines
    ax.plot(range(window-1, len(first_rewards)), first_rewards_smooth,
            label=f"{first_config} Running Mean (window={window})", color='blue')
    ax.plot(range(window-1, len(second_rewards)), second_rewards_smooth,
            label=f"{second_config} Running Mean (window={window})", color='red')
    
    # Plot percentile lines with low alpha
    ax.plot(range(window-1, len(first_rewards)), first_top_smooth, color='blue', alpha=0.1)
    ax.plot(range(window-1, len(first_rewards)), first_bottom_smooth, color='blue', alpha=0.1)
    ax.plot(range(window-1, len(second_rewards)), second_top_smooth, color='red', alpha=0.1)
    ax.plot(range(window-1, len(second_rewards)), second_bottom_smooth, color='red', alpha=0.1)
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Rewards Over Time")
    ax.legend()
    _show_or_save(save_as)


def plot_lifetimes(
    lifetimes: list[dict[int, float]],
    random_agent_lifetime: float | None = None,
    window: int = 50,
    save_as: str | None = None
) -> None:
    num_agents = len(lifetimes[0])
    avg_lifetimes = [np.mean([r for r in lifetime.values()]) for lifetime in lifetimes]
    max_lifetimes = [np.max([r for r in lifetime.values()]) for lifetime in lifetimes]
    min_lifetimes = [np.min([r for r in lifetime.values()]) for lifetime in lifetimes]
    ninetieth_percentile = [np.percentile([r for r in lifetime.values()], 90) for lifetime in lifetimes] if num_agents > 30 else None

    _, ax = plt.subplots(figsize=(10, 6))
    
    if ninetieth_percentile is not None:
        ax.plot(ninetieth_percentile, label="90th Percentile", color='purple', alpha=0.4)
        
    ax.plot(avg_lifetimes, label="Average Lifetime", color='red', alpha=0.4)
    ax.plot(max_lifetimes, label="Max Lifetime", color='pink', alpha=0.8)
    ax.plot(min_lifetimes, label="Min Lifetime", color='green', alpha=0.4)
    
    if random_agent_lifetime is not None:
        ax.axhline(y=random_agent_lifetime, label="Random Agent Lifetime", color='black', linestyle='--')

    avg_rewards_smooth = np.convolve(avg_lifetimes, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(avg_lifetimes)), avg_rewards_smooth, label=f"Running Mean (window={window})", color='blue')

    ax.set_xlabel("Episode")
    ax.set_ylabel("Lifetime")
    ax.set_title("Agent Lifetime Over Time")
    ax.legend()
    _show_or_save(save_as)   
    

def plot_lifetimes_from_save(
    agent_name: str, 
    random_agent_lifetime: float | None = None, 
    window: int = 50,
    save_as: str | None = None
) -> None:
    history = Jar("saves").get(agent_name)
    lifetimes = [episode[3] for episode in history]
    plot_lifetimes(lifetimes, random_agent_lifetime, window, save_as)


def plot_entropies(
    entropies: list[dict[str, dict[str, float]]],
    window: int = 50,
    save_as: str | None = None
) -> None:
    entropies_per_episode = [[entropies 
                             for step in ep 
                             for entropies in step.values()]
                            for ep in entropies]
    means_per_episode_per_type = [{type: np.mean([entropies[type] for entropies in ep]) 
                                 for type in ep[0].keys()} 
                                for ep in entropies_per_episode]
    means_per_type_per_episode = {type: [ep[type] for ep in means_per_episode_per_type]
                                 for type in means_per_episode_per_type[0].keys()}

    _, axs = plt.subplots(2, 2, figsize=(20, 12))
    axs = axs.flatten()
    
    for idx, (type, entropies) in enumerate(means_per_type_per_episode.items()):
        ax = axs[idx]
        ax.plot(entropies, label=f"{type} Entropy", alpha=0.4)
        
        entropies_smooth = np.convolve(entropies, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(entropies)), entropies_smooth, 
               label=f"Running Mean (window={window})")
        
        ax.set_xlabel("Episode")
        ax.set_ylabel("Entropy")
        ax.set_title(f"{type} Entropy Over Time")
        ax.set_ylim(bottom=0)  # Set y-axis minimum to 0
        ax.legend()
    
    plt.tight_layout()
    _show_or_save(save_as)


def plot_entropies_from_save(
    agent_name: str, 
    window: int = 50,
    save_as: str | None = None
) -> None:
    history = Jar("saves").get(agent_name)
    entropies = [episode[4] for episode in history]
    plot_entropies(entropies, window, save_as)
