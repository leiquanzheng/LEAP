
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from copy import deepcopy
from .modeling_llama import LayerAction


@dataclass
class MCTSConfig:
    """MCTS Configuration"""
    # UCB
    exploration_weight: float = 3.0
    layer_penalty_weight: float = 0.3

    # Constraints of Layer Ratio
    min_layer_ratio: float = 0.40
    max_layer_ratio: float = 0.50
    target_layer_ratio: float = 0.45

    # Termination Condition
    max_iterations: int = 40
    convergence_threshold: int = 6

    # Early Exit
    early_stop_reward_threshold: float = 0.85
    min_iterations_before_stop: int = 20

    use_random_baseline: bool = False

@dataclass
class RewardConfig:
    """Reward Function Configuration"""

    alpha: float = 0.50
    beta: float = 0.35
    gamma: float = 0.15

    max_draft_length: int = 8
    max_speedup: float = 1.6
    baseline_time_per_token: float = 0.0


@dataclass
class MCTSNode:
    """MCTS Node"""
    layer_configs: Dict[int, str]
    parent: Optional['MCTSNode'] = None
    children: Dict[str, 'MCTSNode'] = field(default_factory=dict)

    # Statistics
    visits: int = 0
    total_reward: float = 0.0

    next_group_idx: int = 0
    action_taken: Optional[str] = None
    group_id: Optional[int] = None

    _total_groups: int = 0


    @property
    def is_terminal(self) -> bool:

        return self.next_group_idx >= self._total_groups

    @property
    def is_fully_expanded(self) -> bool:

        return len(self.children) >= len(self.valid_actions) if hasattr(self, 'valid_actions') else False

    @property
    def q_value(self) -> float:

        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def __repr__(self) -> str:
        skips = [k for k, v in self.layer_configs.items() if v == LayerAction.SKIP]
        repeats = [k for k, v in self.layer_configs.items() if v == LayerAction.REPEAT]
        return f"MCTSNode(visits={self.visits}, Q={self.q_value:.3f}, skip={skips}, repeat={repeats})"



class MCTS:

    def __init__(
            self,
            groups: List[Dict],
            num_physical_layers: int,
            config: MCTSConfig = None,
            reward_config: RewardConfig = None,
    ):
        self.groups = groups
        self.num_physical_layers = num_physical_layers
        self.config = config or MCTSConfig()
        self.reward_config = reward_config or RewardConfig()

        self.root = MCTSNode(
            layer_configs={},
            next_group_idx=0,
        )
        self.root._total_groups = len(groups)
        self.root.valid_actions = self._get_valid_actions_for_node(self.root)

        self.best_config_history: List[Dict[int, str]] = []
        self.iteration_count = 0

        self.stats = {
            'total_iterations': 0,
            'total_simulations': 0,
            'best_reward': float('-inf'),
            'rewards_history': [],
        }

        self.random_best_config: Optional[Dict[int, str]] = None

        self.global_best_config: Optional[Dict[int, str]] = None
        self.global_best_reward: float = float('-inf')
        self.global_best_ratio: float = 1.0

        self.fallback_best_config: Optional[Dict[int, str]] = None
        self.fallback_best_reward: float = float('-inf')

    def _get_valid_actions_for_node(self, node: MCTSNode) -> List[str]:
        if node.next_group_idx >= len(self.groups):
            return []
        group = self.groups[node.next_group_idx]
        return group.get('actions', [LayerAction.EXECUTE])

    def compute_layer_ratio(self, layer_configs: Dict[int, str]) -> float:
        effective = 0
        for i in range(self.num_physical_layers):
            action = layer_configs.get(i, LayerAction.EXECUTE)
            if action == LayerAction.EXECUTE:
                effective += 1
            elif action == LayerAction.REPEAT:
                effective += 2
        return effective / self.num_physical_layers

    def compute_layer_ratio_for_node(self, node: MCTSNode) -> float:
        configs = node.layer_configs
        if not configs:
            return 1.0
        effective = 0
        count = 0
        for action in configs.values():
            count += 1
            if action == LayerAction.EXECUTE:
                effective += 1
            elif action == LayerAction.REPEAT:
                effective += 2
        return effective / count if count > 0 else 1.0

    def compute_ucb(self, node: MCTSNode, parent_visits: int) -> float:
        if node.visits == 0:
            return float('inf')

        Q = node.q_value

        # Exploration
        if parent_visits > 0:
            exploration = self.config.exploration_weight * math.sqrt(
                math.log(parent_visits) / node.visits
            )
        else:
            exploration = self.config.exploration_weight * 10.0

        # Penalty for Layer Ratio
        layer_ratio = self.compute_layer_ratio_for_node(node)

        if layer_ratio < self.config.min_layer_ratio:
            penalty = (self.config.min_layer_ratio - layer_ratio) * 2
        elif layer_ratio > self.config.max_layer_ratio:
            penalty = (layer_ratio - self.config.max_layer_ratio) * 2
        else:
            penalty = abs(layer_ratio - self.config.target_layer_ratio) * 0.5

        ucb = Q + exploration - self.config.layer_penalty_weight * penalty
        return ucb

    def select(self) -> MCTSNode:
        """
        Selection: Prefer nodes that are not yet fully expanded
        """
        node = self.root

        while not node.is_terminal:
            valid_actions = self._get_valid_actions_for_node(node)

            if len(node.children) < len(valid_actions):
                return node

            if not node.children:
                break

            best_ucb = float('-inf')
            best_children = []

            for child in node.children.values():
                ucb = self.compute_ucb(child, node.visits)
                if ucb > best_ucb + 1e-6:
                    best_ucb = ucb
                    best_children = [child]
                elif abs(ucb - best_ucb) <= 1e-6:
                    best_children.append(child)

            if not best_children:
                break

            node = random.choice(best_children)

        return node

    def expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Expansion
        """
        if node.is_terminal:
            return node

        valid_actions = self._get_valid_actions_for_node(node)
        unexpanded = [a for a in valid_actions if a not in node.children]

        if not unexpanded:
            return None

        created_children = []
        group = self.groups[node.next_group_idx]

        for action in unexpanded:
            new_configs = dict(node.layer_configs)
            layers = group['layers']

            # Action Mapping
            if action == LayerAction.SKIP and len(layers) > 0:
                # Retain the first layer in a group
                if len(layers) > 1:
                    new_configs[layers[0]] = LayerAction.EXECUTE
                    for layer_idx in layers[1:]:
                        new_configs[layer_idx] = LayerAction.SKIP
                else:
                    new_configs[layers[0]] = LayerAction.SKIP
            else:
                for layer_idx in layers:
                    new_configs[layer_idx] = action

            child = MCTSNode(
                layer_configs=new_configs,
                parent=node,
                next_group_idx=node.next_group_idx + 1,
                action_taken=action,
                group_id=node.next_group_idx,
            )
            child._total_groups = len(self.groups)
            child.valid_actions = self._get_valid_actions_for_node(child)

            node.children[action] = child
            created_children.append(child)

        return random.choice(created_children)


    def _heuristic_completion(self, node: MCTSNode) -> Dict[int, str]:
        """
        Heuristic rollout for completing the remaining layer configuration.

        The remaining groups are assigned EXECUTE according to the target layer
        ratio; otherwise, SKIP is selected when allowed. REPEAT is disabled to
        avoid excessive layer retention and negative penalties.
        """

        # 1. Current Configuration
        full_config = dict(node.layer_configs)

        # 2. Traverse Remaining Groups
        current_group_idx = node.next_group_idx

        while current_group_idx < len(self.groups):
            group = self.groups[current_group_idx]
            valid_actions = group.get('actions', [LayerAction.EXECUTE])

            chosen_action = None

            execute_prob = max(0.10, self.config.target_layer_ratio - 0.05)

            # Strategy 1
            if LayerAction.EXECUTE in valid_actions:
                if random.random() < execute_prob:
                    chosen_action = LayerAction.EXECUTE

            # Strategy 2
            if chosen_action is None:
                if LayerAction.SKIP in valid_actions:
                    chosen_action = LayerAction.SKIP

                elif LayerAction.EXECUTE in valid_actions:
                    chosen_action = LayerAction.EXECUTE

                else:
                    chosen_action = valid_actions[0]


            # 3. Apply Action to Group
            layers = group['layers']

            if chosen_action == LayerAction.SKIP and len(layers) > 0:
                if len(layers) > 1:
                    full_config[layers[0]] = LayerAction.EXECUTE
                    for layer_idx in layers[1:]:
                        full_config[layer_idx] = LayerAction.SKIP
                else:
                    full_config[layers[0]] = LayerAction.SKIP
            else:
                for layer_idx in layers:
                    full_config[layer_idx] = chosen_action

            current_group_idx += 1

        # 4. Check
        for i in range(self.num_physical_layers):
            if i not in full_config:
                full_config[i] = LayerAction.EXECUTE

        return full_config

    def _generate_random_config_with_constraints(self) -> Dict[int, str]:
        """
        Randomly generate configuration with constraints.
        """
        full_config = {}

        execute_prob = max(0.10, self.config.target_layer_ratio - 0.05)

        for group in self.groups:

            valid_actions = group.get('actions', [LayerAction.EXECUTE])

            chosen_action = None

            if LayerAction.EXECUTE in valid_actions:
                if random.random() < execute_prob:
                    chosen_action = LayerAction.EXECUTE

            if chosen_action is None:
                other_actions = [a for a in valid_actions if a != LayerAction.EXECUTE]

                if other_actions:
                    chosen_action = random.choice(other_actions)
                else:
                    chosen_action = LayerAction.EXECUTE

            layers = group['layers']
            if chosen_action == LayerAction.SKIP and len(layers) > 0:
                if len(layers) > 1:
                    full_config[layers[0]] = LayerAction.EXECUTE
                    for layer_idx in layers[1:]:
                        full_config[layer_idx] = LayerAction.SKIP
                else:
                    full_config[layers[0]] = LayerAction.SKIP
            else:
                for layer_idx in layers:
                    full_config[layer_idx] = chosen_action

        for i in range(self.num_physical_layers):
            if i not in full_config:
                full_config[i] = LayerAction.EXECUTE

        return full_config


    def run_one_iteration(self) -> Tuple[MCTSNode, Dict[int, str]]:
        """
        One MCTS Iteration
        """

        # ==================== Random Baseline ====================
        if self.config.use_random_baseline:
            config = self._generate_random_config_with_constraints()

            dummy_node = MCTSNode(layer_configs=config)

            self.iteration_count += 1
            self.stats['total_iterations'] = self.iteration_count

            return dummy_node, config
        # ============================================================

        # 1. Selection
        node = self.select()

        # 2. Expansion
        child = self.expand(node)
        if child is None:
            child = node

        # 3. Simulation (Rollout)
        layer_configs_for_eval = self._heuristic_completion(child)

        self.iteration_count += 1
        self.stats['total_iterations'] = self.iteration_count

        # Return Child Node
        return child, layer_configs_for_eval

    def compute_reward(self, accept_length: int, draft_time: float, verify_time: float, layer_configs: Dict[int, str],
                       is_terminal: bool = True) -> float:
        """
        Compute Reward Based on Direct Speedup
        """

        cfg = self.reward_config

        total_time = draft_time + verify_time + 1e-6

        if cfg.baseline_time_per_token > 0:
            baseline_time = accept_length * cfg.baseline_time_per_token
        else:
            baseline_time = accept_length * verify_time

        if total_time > 0:
            speedup = baseline_time / total_time
        else:
            speedup = 0.0

        if speedup >= 1.0:
            if speedup < 1.05:
                final_reward = -0.2
            elif speedup > 1.4:
                final_reward = (speedup - 1.0) * 1.2
            else:
                final_reward = (speedup - 1.0)
        else:
            final_reward = (speedup - 1.0) * 4.0

        layer_ratio = self.compute_layer_ratio(layer_configs)

        min_r = self.config.min_layer_ratio
        max_r = self.config.max_layer_ratio
        target = self.config.target_layer_ratio

        if layer_ratio > 0.80:
            return -5.0

        if min_r <= layer_ratio <= max_r:
            band = max(max_r - min_r, 1e-6)
            closeness = 1.0 - abs(layer_ratio - target) / band
            ratio_bonus = 0.6 * closeness
            final_reward += ratio_bonus
        elif layer_ratio > max_r:
            over = layer_ratio - max_r
            ratio_penalty = 8.0 * over
            final_reward -= ratio_penalty
        else:
            under = min_r - layer_ratio
            ratio_penalty = 3.0 * under
            final_reward -= ratio_penalty

        return max(-5.0, min(5.0, final_reward))


    def backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagation
        """

        self.stats['total_simulations'] += 1
        self.stats['rewards_history'].append(reward)


        if reward > self.stats['best_reward']:
            self.stats['best_reward'] = reward

            if self.config.use_random_baseline:
                self.random_best_config = deepcopy(node.layer_configs)

        if self.config.use_random_baseline:
            return

        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def get_best_config(self) -> Dict[int, str]:
        """
        Get the Best Configuration
        """

        if self.config.use_random_baseline:
            if self.random_best_config is not None:
                return self.random_best_config
            else:
                return {i: LayerAction.EXECUTE for i in range(self.num_physical_layers)}

        if self.global_best_config is not None:
            return self.global_best_config

        if self.fallback_best_config is not None:
            squeezed = self._squeeze_to_target_ratio(self.fallback_best_config)
            return squeezed

        best_node = self._find_best_terminal_node(self.root)
        if best_node:
            cfg = self._complete_config(best_node.layer_configs)
        else:
            partial = self._get_best_partial_config(self.root)
            cfg = self._complete_config(partial)

        return self._squeeze_to_target_ratio(cfg)

    def _squeeze_to_target_ratio(self, config: Dict[int, str]) -> Dict[int, str]:

        cfg = dict(config)

        for i in range(self.num_physical_layers):
            cfg.setdefault(i, LayerAction.EXECUTE)

        skippable_layers = []
        for g in self.groups:
            if LayerAction.SKIP in g.get('actions', []):
                skippable_layers.extend(g['layers'])

        skippable_layers = list(dict.fromkeys(skippable_layers))
        center = self.num_physical_layers / 2.0
        skippable_layers.sort(key=lambda x: abs(x - center))

        max_r = self.config.max_layer_ratio
        min_r = self.config.min_layer_ratio

        for layer_idx in skippable_layers:
            if self.compute_layer_ratio(cfg) <= max_r:
                break
            if cfg.get(layer_idx) != LayerAction.SKIP:
                cfg[layer_idx] = LayerAction.SKIP

        if self.compute_layer_ratio(cfg) < min_r:
            restore_order = list(reversed(skippable_layers))
            for layer_idx in restore_order:
                if self.compute_layer_ratio(cfg) >= min_r:
                    break
                if cfg.get(layer_idx) == LayerAction.SKIP:
                    cfg[layer_idx] = LayerAction.EXECUTE

        return cfg

    def _find_best_terminal_node(self, node: MCTSNode) -> Optional[MCTSNode]:
        if node.is_terminal: return node
        if not node.children: return None
        best_child = max(node.children.values(), key=lambda c: c.visits)
        return self._find_best_terminal_node(best_child)

    def _get_best_partial_config(self, node: MCTSNode) -> Dict[int, str]:
        config = dict(node.layer_configs)
        while node.children:
            best_child = max(node.children.values(), key=lambda c: c.visits)
            config.update(best_child.layer_configs)
            node = best_child
        return config

    def _complete_config(self, partial_config: Dict[int, str]) -> Dict[int, str]:
        complete = {}
        for i in range(self.num_physical_layers):
            complete[i] = partial_config.get(i, LayerAction.EXECUTE)
        return complete


    def check_convergence(self) -> bool:
        if self.iteration_count < 50: return False
        current_best = self.get_best_config()
        self.best_config_history.append(current_best)
        if len(self.best_config_history) > self.config.convergence_threshold * 2:
            self.best_config_history.pop(0)
        if len(self.best_config_history) < self.config.convergence_threshold:
            return False
        recent = self.best_config_history[-self.config.convergence_threshold:]
        first = recent[0]
        for cfg in recent[1:]:
            if cfg != first: return False
        print(f"  >>> Covergence：Consecutive {self.config.convergence_threshold} iterations")
        return True

    def check_early_stop(self) -> bool:
        if self.iteration_count < 50: return False
        if not self.stats['rewards_history']: return False
        recent = self.stats['rewards_history'][-10:]
        avg = sum(recent) / len(recent)
        ratio = self.compute_layer_ratio(self.get_best_config())
        if avg >= self.config.early_stop_reward_threshold and ratio < 0.85:
            print(f"  >>> Early Exit")
            return True
        return False

    def update_with_reward(
        self,
        node: MCTSNode,
        reward: float,
        evaluated_config: Optional[Dict[int, str]] = None,
    ):
        """
        Update of MCTS with Reward
        """

        cfg_for_tracking = evaluated_config if evaluated_config is not None else node.layer_configs

        full_cfg = self._complete_config(cfg_for_tracking)
        ratio = self.compute_layer_ratio(full_cfg)

        if self.config.min_layer_ratio <= ratio <= self.config.max_layer_ratio:
            if reward > self.global_best_reward:
                self.global_best_reward = reward
                self.global_best_ratio = ratio
                self.global_best_config = deepcopy(full_cfg)

        if reward > self.fallback_best_reward:
            self.fallback_best_reward = reward
            self.fallback_best_config = deepcopy(full_cfg)

        self.backpropagate(node, reward)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'iterations': self.iteration_count,
            'simulations': self.stats['total_simulations'],
            'best_reward': self.stats['best_reward'],
            'avg_reward': np.mean(self.stats['rewards_history']) if self.stats['rewards_history'] else 0,
            'best_config': self.get_best_config(),
            'layer_ratio': self.compute_layer_ratio(self.get_best_config()),
        }

    def print_tree(self, node: MCTSNode = None, depth: int = 0, max_depth: int = 3):
        if node is None:
            node = self.root

        if depth > max_depth:
            return

        indent = "  " * depth
        print(f"{indent}{node}")

        for child in node.children.values():
            self.print_tree(child, depth + 1, max_depth)


def create_groups_from_zones(
        zone_structure: Dict,
        num_layers: int,
) -> List[Dict]:

    groups = []

    if 'groups' in zone_structure:
        for group_info in zone_structure['groups']:
            zone = group_info.get('zone', 1)
            layers = group_info.get('layers', [])

            if zone == 1:
                actions = [LayerAction.EXECUTE]
            elif zone == 2:
                actions = [LayerAction.EXECUTE, LayerAction.SKIP]
            elif zone == 3:
                actions = [LayerAction.EXECUTE, LayerAction.REPEAT]
            else:
                actions = [LayerAction.EXECUTE]

            groups.append({
                'zone': zone,
                'layers': layers,
                'actions': actions,
            })

    return groups


