import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import nmmo

from implementations.ActionData import ActionData
from implementations.Observations import Observations
from implementations.EvaluationCallback import EvaluationCallback


class AnimationCallback(EvaluationCallback):
    def __init__(self, agent_id: int, output_name: str, quiet: bool = True):
        self.agent_id = agent_id
        self.output_name = output_name
        self.quiet = quiet
        self._plots_dir = "plots"
        self._image_dir = f"{self._plots_dir}/frames"
        self._current_episode_steps = 0
        
        self.tile_color_map = {
            'void': '#000000',      # Black
            'water': '#4169E1',     # Royal blue
            'grass': '#7CBA3B',     # Yellow-green
            'stone': '#808080',     # Gray
            
            'ore': '#8B4513',       # Saddle brown
            'slag': '#A0522D',      # Sienna
            
            'herb': '#00FF7F',      # Spring green
            'weeds': '#556B2F',     # Dark olive green
            
            'foilage': '#90EE90',   # Light green
            'scrub': '#3CB371',     # Medium sea green
            
            'crystal': '#B8860B',   # Dark goldenrod
            'fragment': '#DEB887',  # Burlywood
            
            'tree': '#228B22',      # Forest green
            'stump': '#8B4513',     # Saddle brown
            
            'fish': '#87CEEB',      # Sky blue
            'ocean': '#000080',     # Navy blue
        }

    def create_animation(self, output_file: str, fps: float = 2) -> None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))

        def update(step):
            filename = os.path.join(self._image_dir, f"step_{step}.png")
            if os.path.exists(filename):
                img = plt.imread(filename)
                ax.clear()
                ax.imshow(img)
                ax.axis('off')
            else:
                if not self.quiet:
                    print(f"File {filename} does not exist")

        ani = animation.FuncAnimation(fig, update, frames=self._current_episode_steps, repeat=False)
        ani.save(output_file, writer='pillow', fps=fps)
        plt.close(fig)
        if not self.quiet:
            print(f"Saved animation to {output_file}")

    def _get_color_grid(
        self, 
        tile_rows: np.ndarray, 
        tile_cols: np.ndarray, 
        tile_values: np.ndarray, 
        min_row: int, 
        min_col: int, 
        grid_rows: int, 
        grid_cols: int
    ) -> np.ndarray:        
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return [int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4)]

        index_to_material = {m.index: m.tex for m in nmmo.material.All.materials}
        color_grid = np.zeros((grid_rows, grid_cols, 3))
        for r, c, v in zip(tile_rows, tile_cols, tile_values):
            color_grid[r - min_row, c - min_col] = hex_to_rgb(self.tile_color_map[index_to_material[v]])
        return color_grid

    def _get_action_text(self, action_data: ActionData, entity_ids: np.ndarray) -> str:
        if not action_data:
            return "No action"
        
        action_parts = []
        if 'Move' in action_data.action_dict and 'Direction' in action_data.action_dict['Move']:
            probability = action_data.distributions["Move"].probs[0, action_data.action_dict['Move']['Direction']]
            direction_name = {
                0: 'Up', 1: 'Down', 2: 'Right', 3: 'Left', 4: 'Stay'
            }[action_data.action_dict['Move']['Direction']]
            action_parts.append(f"Move: {direction_name} ({(probability * 100):.2f}%)")
        if 'Attack' in action_data.action_dict:
            attack = action_data.action_dict['Attack']
            if 'Style' in attack and 'Target' in attack:
                style_prob = action_data.distributions["AttackStyle"].probs[0, attack['Style']]
                attack_or_not_prob = action_data.distributions["AttackOrNot"].probs[0, 1 if attack['Target'] != 100 else 0]
                sees_valid_target = np.sum(entity_ids < 0) > 0
                target_id = entity_ids[attack["Target"]] if attack['Target'] != 100 \
                            else "None" if sees_valid_target \
                            else "----"
                action_parts.append(f"Attack: Style {attack['Style']} ({(style_prob * 100):.2f}%), Target {target_id} ({(attack_or_not_prob * 100):.2f}%)")
        if 'Use' in action_data.action_dict and 'InventoryItem' in action_data.action_dict['Use']:
            action_parts.append(f"Use: Item {action_data.action_dict['Use']['InventoryItem']}")
        if 'Destroy' in action_data.action_dict and 'InventoryItem' in action_data.action_dict['Destroy']:
            action_parts.append(f"Destroy: Item {action_data.action_dict['Destroy']['InventoryItem']}")
            
        return '\n'.join(action_parts) if action_parts else "No action"

    def _plot_entities(self, ax, agent_observations: Observations, min_row: int, min_col: int):
        def get_health_color(health):
            if health > 75: return 'green'
            elif health > 50: return 'yellow'
            elif health > 25: return 'orange'
            return 'red'

        for idx, entity_id in enumerate(agent_observations.entities.id):
            if entity_id == self.agent_id or entity_id == 0:
                continue

            local_x = agent_observations.entities.row[idx] - min_row
            local_y = agent_observations.entities.col[idx] - min_col
            health = agent_observations.entities.health[idx]
            ax.scatter(local_y, local_x, c=get_health_color(health), s=50, 
                      alpha=0.7, label=f'Entity {entity_id}', edgecolors='black')
            
    def _plot_tile_legend(self, ax):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(self.tile_color_map))
        ax.invert_yaxis()
        ax.axis('off')
        items = list(self.tile_color_map.items())
        for idx, (mat, color) in enumerate(items):
            rect = plt.Rectangle((0, idx), 0.5, 0.5, facecolor=color, edgecolor='black', alpha=0.8)
            ax.add_patch(rect)
            text_color = 'white' if mat in ['ocean', 'void'] else 'black'
            ax.text(0.25, idx + 0.25, mat, va='center', ha='center', fontsize=12, family='monospace', color=text_color)

    def plot_agent_view(
        self, 
        obs: dict[int, Observations], 
        env_actions: dict[int, ActionData], 
        agent_id: int, 
        step: int
    ) -> None:
        if not os.path.exists(self._image_dir):
            os.makedirs(self._image_dir)

        agent_observations = obs.get(agent_id)
        if agent_observations is None or not isinstance(agent_observations, Observations):
            if not self.quiet:
                print(f"Agent {agent_id} not found in observations at step {step}")
            return

        # Extract tile information
        tiles = agent_observations.tiles
        tile_rows, tile_cols, tile_values = tiles[:, 0], tiles[:, 1], tiles[:, 2]
        min_row, max_row = tile_rows.min(), tile_rows.max()
        min_col, max_col = tile_cols.min(), tile_cols.max()
        grid_rows, grid_cols = max_row - min_row + 1, max_col - min_col + 1

        # Create plot
        _, (ax, ax_legend) = plt.subplots(1, 2, figsize=(10, 8), gridspec_kw={'width_ratios': [10, 2]})
        self._plot_tile_legend(ax_legend)
        
        color_grid = self._get_color_grid(tile_rows, tile_cols, tile_values, min_row, min_col, grid_rows, grid_cols)
        ax.imshow(color_grid, interpolation='nearest', alpha=0.8)

        agent_idx = np.where(agent_observations.entities.id == agent_id)[0][0]

        # Plot agent
        center_x = agent_observations.entities.row[agent_idx] - min_row
        center_y = agent_observations.entities.col[agent_idx] - min_col
        agent_health = agent_observations.entities.health[agent_idx]
        agent_food = agent_observations.entities.food[agent_idx]
        agent_water = agent_observations.entities.water[agent_idx]
        agent_health_color = 'lightgreen' if agent_health > 75 else 'yellowgreen' if agent_health > 50 else 'darkorange' if agent_health > 25 else 'darkred'
        ax.scatter(center_y, center_x, c=agent_health_color, s=100, label=f'Agent {agent_id}', edgecolors='black')

        # Plot other entities
        self._plot_entities(ax, agent_observations, min_row, min_col)

        # Add text information
        agent_stats = f"Health: {agent_health:>3}\n" + \
                      f"Food:   {agent_food:>3}\n" + \
                      f"Water:  {agent_water:>3}"
        action_text = self._get_action_text(env_actions.get(agent_id), agent_observations.entities.id)
        
        ax.text(0.05, 0.05, f"Action:\n{action_text}", transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', horizontalalignment='left', color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'), family='monospace')
        ax.text(0.95, 0.95, f"Step:  {step:>4}", transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right', color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'), family='monospace')
        ax.text(0.95, 0.90, agent_stats, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right', color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'), family='monospace')

        # Set up axes
        ax.set_xticks(np.arange(grid_cols))
        ax.set_yticks(np.arange(grid_rows))
        ax.set_xticklabels(np.arange(min_col, max_col + 1))
        ax.set_yticklabels(np.arange(min_row, max_row + 1))
        legend = ax.legend(loc='upper left')
        for text in legend.get_texts():
            text.set_family('monospace')
        
        plt.tight_layout()
        output_file = f"{self._image_dir}/step_{step}.png"
        plt.savefig(output_file)
        plt.close()
        
        if not self.quiet:
            print(f"Saved agent view to {output_file}")

    def step(
        self,
        observations_per_agent: dict[int, Observations], 
        actions_per_agent: dict[int, ActionData], 
        episode: int, 
        step: int) -> None:
        self.plot_agent_view(
            observations_per_agent, 
            actions_per_agent, 
            self.agent_id, 
            step)
        self._current_episode_steps += 1

    def episode_start(self, episode: int) -> None:
        self._current_episode_steps = 0

    def episode_end(
        self, 
        episode: int, 
        rewards_per_agent: dict[int, float], 
        losses: tuple[list[float], list[float], list[float]],
        eval_rewards: list[dict[int, float]] | None
    )-> None:
        self.create_animation(f"{self._plots_dir}/animations/{self.output_name}_{episode}_{int(time.time())}.gif", fps=2)
        if os.path.exists(self._image_dir):
            shutil.rmtree(self._image_dir)
