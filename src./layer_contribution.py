"""
Zone Partitiong and Layer Grouping

Metrics：
1. R_ℓ (Relative F-Norm)
2. ΔL (Accepted Length Gain)

Usage：
    python src/layer_contribution.py
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import torch.nn.functional as F
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from dataclasses import dataclass
import gc
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ZoneConfig:
    """Zone Partitiong Configuration"""

    r_threshold: float = 0.3       # R_ℓ threshold
    ear_threshold: float = 0.01    # ΔL threshold
    min_protect_layers: int = 3    # Protected Layers in Zone 1
    max_protect_layers: int = 2    # Protected Layers in Zone 3
    group_similarity_threshold: float = 0.1
    
    # Adaptive Threshold
    use_adaptive: bool = True
    use_percentile: bool = True    # True: Percentile Threshold
    r_percentile: float = 70.0     # Percentile Threshold of R_ℓ
    ear_percentile: float = 70.0   # Percentile Threshold of ΔL
    r_std_multiplier: float = 0.5
    ear_std_multiplier: float = 1.0


@dataclass
class ZoneStructure:
    """Zone Structure"""
    zone_assignments: List[int]
    zone_layers: Dict[int, List[int]]
    zone_groups: Dict[int, List[List[int]]]
    action_spaces: Dict[int, List[str]]


class LayerContributionAnalyzer:

    def __init__(self, model, num_positions: int = 8):
        self.model = model

        # Number of Layers
        if hasattr(model, "config"):
            self.num_layers = model.config.num_hidden_layers
        elif hasattr(model, "get_num_layers"):
            self.num_layers = model.get_num_layers()
        else:
            self.num_layers = len(model.layers)

        self.num_positions = num_positions
        self.hooks = []
        self.enabled = False

        # Inputs and Outputs of Each Layer
        self.layer_outputs: Dict[int, torch.Tensor] = {}
        self.layer_inputs: Dict[int, torch.Tensor] = {}

    def _make_output_hook(self, layer_idx: int):

        def hook(module, input, output):
            if not self.enabled:
                return

            if isinstance(input, tuple):
                inp_tensor = input[0]
            else:
                inp_tensor = input

            if isinstance(output, tuple):
                out_tensor = output[0]
            else:
                out_tensor = output

            seq_len = inp_tensor.shape[1]
            extract_len = min(seq_len, self.num_positions)

            self.layer_inputs[layer_idx] = inp_tensor[:, -extract_len:, :].detach().cpu()
            self.layer_outputs[layer_idx] = out_tensor[:, -extract_len:, :].detach().cpu()

        return hook

    def register_hooks(self):

        if hasattr(self.model, "layers"):
            layers = self.model.layers

        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        else:
            raise AttributeError("Analyzer cannot find layer list")

        for idx, layer in enumerate(layers):
            h = layer.register_forward_hook(self._make_output_hook(idx))
            self.hooks.append(h)

        print(f"[Analyzer] Register {len(self.hooks)} hooks，Coverd {len(layers)} layers")

    def enable(self):
        self.enabled = True
        self.layer_outputs.clear()
        self.layer_inputs.clear()

    def disable(self):
        self.enabled = False

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.layer_outputs.clear()
        self.layer_inputs.clear()

    # Calculate F norms
    def compute_relative_f_norms(self) -> torch.Tensor:
        """
        Calculate Relative F-Norm (R_ℓ) of Every Layer
        R_ℓ = ||output - input||_F / ||input||_F
        """
        device = next(iter(self.layer_outputs.values())).device if self.layer_outputs else 'cpu'
        relative_f_norms = torch.zeros(self.num_layers, device=device)
        
        for layer_idx in range(self.num_layers):
            layer_in = self.layer_inputs.get(layer_idx)
            layer_out = self.layer_outputs.get(layer_idx)
            
            if layer_in is None or layer_out is None:
                print(f"Warning")
                continue
            
            layer_in = layer_in.float()
            layer_out = layer_out.float()
            
            diff = layer_out - layer_in
            diff_f_norm = torch.norm(diff)
            input_f_norm = torch.norm(layer_in)
            relative_f_norms[layer_idx] = diff_f_norm / (input_f_norm + 1e-8)
        
        return relative_f_norms

    def compute_expected_accept_lengths(self) -> torch.Tensor:
        """
        Calculate expected Accepted Length of Every Layer of the last k positions
        """
        device = next(iter(self.layer_outputs.values())).device if self.layer_outputs else 'cpu'
        expected_lengths = torch.zeros(self.num_layers, device=device)

        num_positions = self.num_positions

        final_layer_out = self.layer_outputs.get(self.num_layers - 1)
        if final_layer_out is None:
            print("Warning")
            return expected_lengths

        final_layer_out = final_layer_out[:, -num_positions:, :]
        
        final_normed = self.model.model.norm(final_layer_out.float())
        final_normed = final_normed.to(self.model.lm_head.weight.dtype)

        with torch.no_grad():
            final_logits = self.model.lm_head(final_normed)  # (batch, num_positions, vocab_size)
            final_probs = F.softmax(final_logits, dim=-1)  # (batch, num_positions, vocab_size)

        for layer_idx in range(self.num_layers):
            layer_out = self.layer_outputs.get(layer_idx)
            if layer_out is None:
                continue

            layer_out = layer_out[:, -num_positions:, :]  # (batch, num_positions, hidden_size)

            layer_normed = self.model.model.norm(layer_out.float())
            layer_normed = layer_normed.to(self.model.lm_head.weight.dtype)

            with torch.no_grad():
                layer_logits = self.model.lm_head(layer_normed)  # (batch, num_positions, vocab_size)
                layer_probs = F.softmax(layer_logits, dim=-1)  # (batch, num_positions, vocab_size)

                draft_tokens = layer_logits.argmax(dim=-1)  # (batch, num_positions)


                draft_token_probs = layer_probs.gather(
                    -1,
                    draft_tokens.unsqueeze(-1)
                ).squeeze(-1)  # (batch, num_positions)

            target_token_probs = final_probs.gather(
                -1,
                draft_tokens.unsqueeze(-1)
            ).squeeze(-1)  # (batch, num_positions)


            # α_t = min(1, p_target_t / p_draft_t)
            alphas = torch.clamp(
                target_token_probs / (draft_token_probs + 1e-10),
                max=1.0
            )  # (batch, num_positions)

            cumulative_probs = torch.cumprod(alphas, dim=1)  # (batch, num_positions)

            expected_length = cumulative_probs.sum(dim=1).mean()

            expected_lengths[layer_idx] = expected_length
        #print(f'{expected_lengths=}')
        return expected_lengths

    def compute_delta_ear(self, expected_lengths: torch.Tensor) -> torch.Tensor:
        """
        Calculate Accepted Length Gain (ΔL)

        ΔEAR_l = EAR_l - EAR_{l-1}
        """

        delta_ear = torch.zeros(self.num_layers, device=expected_lengths.device)
        delta_ear[0] = expected_lengths[0]

        for i in range(1, self.num_layers):
            delta_ear[i] = expected_lengths[i] - expected_lengths[i - 1]

        return delta_ear

    def analyze_sample(self, inputs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        self.enable()

        with torch.no_grad():
            _ = self.model(**inputs, output_hidden_states=True)

        self.disable()

        # Calculate R_ℓ
        relative_f_norms = self.compute_relative_f_norms()

        expected_accept_lens = self.compute_expected_accept_lengths()

        # Calculate ΔL
        delta_ear = self.compute_delta_ear(expected_accept_lens)

        return relative_f_norms, expected_accept_lens, delta_ear


def compute_adaptive_thresholds(
    relative_f_norms: np.ndarray,
    delta_ear: np.ndarray,
    config: ZoneConfig
) -> Tuple[float, float]:
    """
    Adaptive Threshold
    """
    if config.use_percentile:
        # Percentile
        r_threshold = np.percentile(relative_f_norms, config.r_percentile)
        ear_threshold = np.percentile(delta_ear, config.ear_percentile)
    else:
        # mean + k * std
        r_mean = np.mean(relative_f_norms)
        r_std = np.std(relative_f_norms)
        r_threshold = r_mean + config.r_std_multiplier * r_std

        ear_mean = np.mean(delta_ear)
        ear_std = np.std(delta_ear)
        ear_threshold = ear_mean + config.ear_std_multiplier * ear_std

    return r_threshold, ear_threshold


def assign_zones(
        relative_f_norms: np.ndarray,
        delta_ear: np.ndarray,
        config: ZoneConfig,
        r_threshold: Optional[float] = None,
        ear_threshold: Optional[float] = None
) -> List[int]:
    """
    Assign Zones
    """
    num_layers = len(relative_f_norms)
    zones = []

    r_thresh = r_threshold if r_threshold is not None else config.r_threshold
    ear_thresh = ear_threshold if ear_threshold is not None else config.ear_threshold

    min_layer_for_zone3 = max(config.min_protect_layers, int(num_layers * 0.25))

    for l in range(num_layers):
        if l < config.min_protect_layers:
            zones.append(1)
            continue

        if l >= num_layers - config.max_protect_layers:
            zones.append(3)
            continue

        r_l = relative_f_norms[l]
        d_l = delta_ear[l]

        is_candidate_z1 = (r_l > r_thresh)

        is_candidate_z3 = (d_l > ear_thresh) and (l >= min_layer_for_zone3)


        if is_candidate_z1 and is_candidate_z3:
            r_norm = r_l / (r_thresh + 1e-10)
            d_norm = d_l / (ear_thresh + 1e-10)

            if r_norm >= d_norm:
                zones.append(1)
            else:
                zones.append(3)

        elif is_candidate_z1:
            zones.append(1)

        elif is_candidate_z3:
            zones.append(3)

        else:
            zones.append(2)

    return zones


def smooth_zone_sequences(zones: List[int], min_segment_length: int = 2, protect_zone1: bool = True) -> List[int]:
    """
    Smooth Zones
    """
    n = len(zones)
    if n <= 1:
        return zones

    smoothed = list(zones)

    segments = []
    if n > 0:
        curr_start = 0
        curr_zone = zones[0]
        for i in range(1, n):
            if zones[i] != curr_zone:
                segments.append((curr_start, i - 1, curr_zone))
                curr_zone = zones[i]
                curr_start = i
        segments.append((curr_start, n - 1, curr_zone))

    for i in range(len(segments)):
        start, end, seg_zone = segments[i]
        seg_len = end - start + 1

        if seg_len >= min_segment_length:
            continue

        if protect_zone1 and seg_zone == 1:
            continue

        prev_zone = segments[i - 1][2] if i > 0 else None
        next_zone = segments[i + 1][2] if i < len(segments) - 1 else None

        prev_is_long = False
        if i > 0:
            prev_len = segments[i - 1][1] - segments[i - 1][0] + 1
            prev_is_long = prev_len >= min_segment_length

        next_is_long = False
        if i < len(segments) - 1:
            next_len = segments[i + 1][1] - segments[i + 1][0] + 1
            next_is_long = next_len >= min_segment_length

        target_zone = None

        if protect_zone1 and (prev_zone == 1 or next_zone == 1):
            target_zone = 1

        elif target_zone is None:
            if prev_is_long and not next_is_long:
                target_zone = prev_zone
            elif next_is_long and not prev_is_long:
                target_zone = next_zone
            elif prev_is_long and next_is_long:
                target_zone = prev_zone
            else:
                target_zone = prev_zone if prev_zone is not None else next_zone


        if target_zone is not None:
            for k in range(start, end + 1):
                smoothed[k] = target_zone

    for _ in range(2):
        for i in range(1, n - 1):
            left = smoothed[i - 1]
            center = smoothed[i]
            right = smoothed[i + 1]

            if center != left and center != right:
                if left == right:
                    smoothed[i] = left

            if protect_zone1:
                if (left == 1 or right == 1) and center != 1:
                    if left == 1 and right == 1:
                        smoothed[i] = 1

    return smoothed


def group_layers_layerwise(zones: List[int]) -> List[Dict]:
    """
    Group Layers in a Layerwise Manner
    """
    groups = []

    for layer_idx, zone in enumerate(zones):
        if zone == 1:
            actions = ['EXECUTE']
        elif zone == 2:
            actions = ['EXECUTE', 'SKIP']
        else:
            actions = ['EXECUTE', 'REPEAT']

        groups.append({
            'layers': [layer_idx],
            'zone': zone,
            'actions': actions
        })

    return groups

def group_layers_smart(
    zones: List[int],
    delta_ear: np.ndarray,
    rl_scores: np.ndarray,
) -> List[Dict]:
    """
    Group Layers Adaptively
    """
    n = len(zones)
    groups = []
    current_group = []

    z2_indices = [i for i, z in enumerate(zones) if z == 2]
    z3_indices = [i for i, z in enumerate(zones) if z == 3]

    # Maximum Group Size of Zone 2 and Zone 3
    ZONE2_MAX_STRIDE = max(3, len(z2_indices) // 6) if len(z2_indices) > 0 else 4
    ZONE3_MAX_STRIDE = max(2, len(z3_indices) // 4) if len(z3_indices) > 0 else 2


    z2_ears = [delta_ear[i] for i in z2_indices]
    z3_ears = [delta_ear[i] for i in z3_indices]


    if z2_ears:
        Z2_SPIKE_THRESHOLD = max(0.08, np.percentile(z2_ears, 85))
    else:
        Z2_SPIKE_THRESHOLD = 0.02


    if z3_ears:
        Z3_ELITE_THRESHOLD = np.percentile(z3_ears, 82.5)
    else:
        Z3_ELITE_THRESHOLD = 2.0

    #print(f"[Group Config] Z2_Stride={ZONE2_MAX_STRIDE}, Z3_Stride={ZONE3_MAX_STRIDE}")
    #print(f"[Group Config] Z2_Threshold={Z2_SPIKE_THRESHOLD:.5f}, Z3_Threshold={Z3_ELITE_THRESHOLD:.5f}")


    is_z2_spike = [False] * n
    for i in z2_indices:
        if delta_ear[i] > Z2_SPIKE_THRESHOLD or (i > 0 and rl_scores[i] > rl_scores[i - 1] * 1.3):
            is_z2_spike[i] = True

    for i in range(n):
        if zones[i] == 1:
            if current_group and zones[current_group[-1]] == 1:
                current_group.append(i)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [i]
            continue

        z = zones[i]
        score = delta_ear[i]
        prev_z = zones[i - 1] if i > 0 else -1

        should_cut = False

        if prev_z != -1 and z != prev_z:
            should_cut = True

        elif z == 2:
            curr_is_spike = is_z2_spike[i]

            if len(current_group) > 0:
                prev_idx = current_group[-1]
                prev_is_spike = is_z2_spike[prev_idx] if zones[prev_idx] == 2 else False

                if prev_is_spike and not curr_is_spike:
                    should_cut = True
                elif not prev_is_spike and curr_is_spike:
                    should_cut = True
                elif not prev_is_spike and not curr_is_spike:
                    if len(current_group) >= ZONE2_MAX_STRIDE:
                        should_cut = True

        elif z == 3:
            is_elite = (score >= Z3_ELITE_THRESHOLD)

            if is_elite:
                should_cut = True
            elif len(current_group) > 0 and score > max(delta_ear[current_group[-1]], 1e-6) * 2.0:
                should_cut = True
            elif len(current_group) >= ZONE3_MAX_STRIDE:
                should_cut = True

        if should_cut and current_group:
            groups.append(current_group)
            current_group = [i]
        else:
            current_group.append(i)

    if current_group:
        groups.append(current_group)

    result_groups = []
    for layer_list in groups:
        if not layer_list:
            continue
        zone = zones[layer_list[0]]

        if zone == 1:
            actions = ['EXECUTE']
        elif zone == 2:
            actions = ['EXECUTE', 'SKIP']
        else:  # zone == 3
            actions = ['EXECUTE', 'REPEAT']

        result_groups.append({
            'layers': layer_list,
            'zone': zone,
            'actions': actions
        })

    return result_groups


def build_zone_structure(
    relative_f_norms: np.ndarray,
    delta_ear: np.ndarray,
    config: ZoneConfig = None,
    use_smart_grouping: bool = True
) -> Tuple[ZoneStructure, Tuple[float, float], List[Dict]]:

    if config is None:
        config = ZoneConfig()

    num_layers = len(relative_f_norms)

    # Step 1: Calculte Thresholds
    if config.use_adaptive:
        r_threshold, ear_threshold = compute_adaptive_thresholds(
            relative_f_norms, delta_ear, config
        )
        if config.use_percentile:
            pass
        else:
            pass
    else:
        r_threshold = config.r_threshold
        ear_threshold = config.ear_threshold

    # Step 2: Zone Partitioning
    raw_zone_assignments = assign_zones(
        relative_f_norms, delta_ear, config,
        r_threshold=r_threshold, ear_threshold=ear_threshold
    )

    # Step 3: Zone Smooth
    zone_assignments = smooth_zone_sequences(
        raw_zone_assignments,
        min_segment_length=2,
        protect_zone1=True
    )

    zone_layers = {1: [], 2: [], 3: []}
    for l, z in enumerate(zone_assignments):
        zone_layers[z].append(l)

    # Step 5: Layer Grouping
    if use_smart_grouping:
        # Layer Grouping adaptively
        groups = group_layers_smart(zone_assignments, delta_ear, relative_f_norms)
    else:
        # Layer Grouping in a Layerwise Manner
        groups = group_layers_layerwise(zone_assignments)

    # Step 6: Construct Zone_groups
    zone_groups = {1: [], 2: [], 3: []}
    for g in groups:
        zone_groups[g['zone']].append(g['layers'])

    # Step 7: Action Space
    action_spaces = {
        1: ['EXECUTE'],              # Zone 1
        2: ['EXECUTE', 'SKIP'],      # Zone 2
        3: ['EXECUTE', 'REPEAT']     # Zone 3
    }

    zone_structure = ZoneStructure(
        zone_assignments=zone_assignments,
        zone_layers=zone_layers,
        zone_groups=zone_groups,
        action_spaces=action_spaces
    )

    return zone_structure, (r_threshold, ear_threshold), groups



def run_contribution_analysis(
    model_path: str,
    dataset_name: str,
    data_path: str = "./data",
    max_samples: int = 120,
    max_length: int = 512,
    num_positions: int = 8,
    zone_config: ZoneConfig = None,
):

    print("=" * 100)
    print(f"Model: {model_path}")
    print(f"Dateset: {dataset_name}")
    print(f"Max Sample Number: {max_samples}")
    print(f"Positions: {num_positions}")
    print("=" * 100)

    if zone_config is None:
        zone_config = ZoneConfig()

    print("\n[1/4] Dataset Loading...")
    samples = get_dataset(dataset_name, data_path)
    samples = samples[:max_samples]
    print(f"{len(samples)} Samples")


    print("\n[2/4] Model loading...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"Model loaded, {num_layers} layers")


    print("\n[3/4] Analyzing...")
    analyzer = LayerContributionAnalyzer(model, num_positions=num_positions)
    analyzer.register_hooks()

    all_relative_f_norms = []
    all_expected_accept_lens = []
    all_delta_ear = []

    with torch.no_grad():
        for idx, text in enumerate(tqdm(samples, desc="Analyzing")):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            ).to(model.device)

            if inputs.input_ids.shape[1] < num_positions:
                print(f"Lentgh of sample {idx} is not engough for {num_positions}，skipping")
                continue

            rel_f_norms, expected_accept_lens, delta_ear = analyzer.analyze_sample(inputs)

            all_relative_f_norms.append(rel_f_norms.cpu().numpy())
            all_expected_accept_lens.append(expected_accept_lens.cpu().numpy())
            all_delta_ear.append(delta_ear.cpu().numpy())

            if (idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

    analyzer.remove_hooks()

    print("\n[4/4] Summary")
    avg_relative_f_norms = np.mean(all_relative_f_norms, axis=0)
    std_relative_f_norms = np.std(all_relative_f_norms, axis=0)
    avg_expected_accept_lens = np.mean(all_expected_accept_lens, axis=0)
    std_expected_accept_lens = np.std(all_expected_accept_lens, axis=0)
    avg_delta_ear = np.mean(all_delta_ear, axis=0)
    std_delta_ear = np.std(all_delta_ear, axis=0)

    zone_structure, (r_threshold_used, ear_threshold_used), groups = build_zone_structure(
        avg_relative_f_norms, avg_delta_ear, zone_config,use_smart_grouping=True
    )

    results = {
        "model": model_path,
        "dataset": dataset_name,
        "num_samples": len(all_relative_f_norms),
        "num_layers": num_layers,
        "num_positions": num_positions,

        "zone_config": {
            "use_adaptive": zone_config.use_adaptive,
            "use_percentile": zone_config.use_percentile,
            "r_percentile": zone_config.r_percentile,
            "ear_percentile": zone_config.ear_percentile,
            "r_std_multiplier": zone_config.r_std_multiplier,
            "ear_std_multiplier": zone_config.ear_std_multiplier,
            "min_protect_layers": zone_config.min_protect_layers,
            "max_protect_layers": zone_config.max_protect_layers,
        },

        "thresholds_used": {
            "r_threshold": r_threshold_used,
            "ear_threshold": ear_threshold_used,
        },

        "avg_relative_f_norms": avg_relative_f_norms.tolist(),
        "std_relative_f_norms": std_relative_f_norms.tolist(),
        "avg_expected_accept_lens": avg_expected_accept_lens.tolist(),
        "std_expected_accept_lens": std_expected_accept_lens.tolist(),
        "avg_delta_ear": avg_delta_ear.tolist(),
        "std_delta_ear": std_delta_ear.tolist(),
        "zone_assignments": zone_structure.zone_assignments,
        "zone_layers": {str(k): v for k, v in zone_structure.zone_layers.items()},
        "zone_groups": {str(k): v for k, v in zone_structure.zone_groups.items()},
    }

    return avg_relative_f_norms, avg_delta_ear, avg_expected_accept_lens, zone_structure, results


if __name__ == "__main__":
    model_name = "Llama3.1-8b-instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    torch.cuda.empty_cache()

    config = ZoneConfig(
        r_threshold=0.5,
        ear_threshold=0.1,

        min_protect_layers=3,
        max_protect_layers=2,

        group_similarity_threshold=0.15,

        use_adaptive=True,
        use_percentile=True,
        r_percentile=75,
        ear_percentile=82.5,

        r_std_multiplier=1.0,
        ear_std_multiplier=1.0,
    )

    datasets = ["gsm8k_prompts"]
    layers = np.arange(1, num_layers+1)

    rel_f_norms = np.zeros( (len(datasets), num_layers))
    delta_ears = np.zeros( ( len(datasets), num_layers))
    zones = np.zeros( (len(datasets), num_layers))

    for i in range(len(datasets)):
        rel_f_norm, delta_ear, expected_accept_lens, zone_structure, results = run_contribution_analysis(
            model_path= model_name,
            dataset_name=datasets[i],
            data_path="./data",
            max_samples=120,
            max_length=256,
            num_positions=8,
            zone_config=config,
        )
        
        rel_f_norms[i, :] = rel_f_norm
        delta_ears[i, :] = delta_ear
        zones[i, :] = np.array( zone_structure.zone_assignments)
        gc.collect()
        torch.cuda.empty_cache()

    avg_rel_f_norm = np.mean(rel_f_norms, axis=0)
    mu_r, sd_r = float(np.mean(avg_rel_f_norm)), float(np.std(avg_rel_f_norm))
    thr_r1 = np.percentile(avg_rel_f_norm, 75)
    thr_r2 = mu_r + 1.0 * sd_r

    avg_delta_ear = np.mean(delta_ears, axis=0)
    mu_g, sd_g = float(np.mean(avg_delta_ear)), float(np.std(avg_delta_ear))
    thr_g1 = np.percentile(avg_delta_ear, 82.5)
    thr_g2 = mu_g + 1.0 * sd_g



