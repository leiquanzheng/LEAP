# LEAP: Zone-Aware MCTS for LLM Self-Speculative Decoding

**Introduction**
-----
SWIFT is an **on-the-fly self-speculative decoding** algorithm that adaptively selects intermediate layers of LLMs to skip during inference. This method **does not require auxiliary models or additional training**, making it a *plug-and-play* and *cost-effective* solution for accelerating LLM inference.

SWIFT divides LLM inference into two distinct phases:

- **Optimization phase:** Identify the optimal skipped layer set given the input data stream.
- **Acceleration phase:** Employ the determined configuration to accelerate LLM inference.

During the optimization stage, SWIFT performs an optimization step prior to each LLM decoding step to adjust the skipped layer set, which involves:

**a) Efficient layer set optimization.**  
SWIFT integrates random search with interval Bayesian optimization to propose layer set candidates efficiently.

**b) Parallel candidate evaluation.**  
SWIFT uses LLM-generated tokens as ground truth, enabling simultaneous validation of the proposed candidates. The best-performing layer set is selected to accelerate the current decoding step.

<p align="center">
  <img src="assets/swift_overview.png" width="95%">
</p>
