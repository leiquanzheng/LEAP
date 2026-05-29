# LEAP: Zone-Aware MCTS for LLM Self-Speculative Decoding

**Introduction**
----
LEAP is a **zone-aware MCTS-based self-speculative decoding** framework that accelerates LLM inference by adaptively constructing a lightweight draft model from the target model itself. This method **does not require auxiliary draft models or additional training**, making it a *plug-and-play* and *lossless* solution for accelerating LLM inference.

LEAP divides LLM inference acceleration into two distinct stages:

- **Prefilling-based initialization:** Estimate layer redundancy from the prefilling stage and organize layers into redundancy-aware zones and groups.
- **MCTS-guided acceleration:** Use Monte Carlo Tree Search to explore layer-group actions and construct an efficient draft model for speculative decoding.

During the initialization stage, LEAP computes redundancy signals from intermediate features and output distributions. Based on these signals, layers are partitioned into **early**, **middle**, and **final** zones, where different zones play different roles in feature transformation and acceptance improvement. Consecutive layers within each zone are further merged into groups to reduce the search space.

During the acceleration stage, LEAP formulates draft model construction as a sequential decision-making problem. MCTS adaptively selects group-level actions, including **execute**, **skip**, and **repeat**, and evaluates each candidate configuration through real-time speculative decoding feedback. The best-performing configuration is then fixed to accelerate the remaining decoding process.

<p align="center">
  <img src="assets/leap_overview.png" width="95%">
</p>
