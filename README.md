# LEAP: Zone-Aware MCTS for LLM Self-Speculative Decoding

**Introduction**
----
LEAP is a **plug-and-play self-speculative decoding** framework that accelerates LLM inference by adaptively constructing a draft model from the target model itself. It requires **no auxiliary models** or **additional training**, while preserving the original output distribution through lossless verification.

LEAP divides inference acceleration into two stages:

- **Initialization phase:** Estimate layer redundancy during prefilling and organize layers into zone-aware groups.
- **Acceleration phase:** Use MCTS to search effective layer-group actions and construct an efficient draft model.

During decoding, LEAP uses real-time speedup feedback to update the MCTS search direction. The best-performing configuration is then fixed to accelerate the remaining generation process.

<p align="center">
  <img src="assets/leap_overview.png" width="95%">
</p>
