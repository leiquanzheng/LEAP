# LEAP: Zone-Aware MCTS for LLM Self-Speculative Decoding

**Introduction**
-----
LEAP is a **zone-aware MCTS-based self-speculative decoding** framework that accelerates LLM inference by adaptively constructing a lightweight draft model from the target model itself. Instead of relying on auxiliary draft models or additional training, LEAP provides a **plug-and-play**, **training-free**, and **lossless** acceleration solution for large language model inference.

LEAP divides LLM inference acceleration into two distinct stages:

- **Prefilling-based initialization:** Estimate layer redundancy during the prefilling stage and partition LLM layers into redundancy-aware zones and groups.
- **MCTS-guided acceleration:** Employ Monte Carlo Tree Search to adaptively explore layer-group actions and construct an effective draft model for self-speculative decoding.

During the initialization stage, LEAP derives redundancy information from a single prefilling forward pass. Based on two complementary signals, i.e., relative feature change and acceptance gain, LEAP partitions layers into **early**, **middle**, and **final** zones. The early zone mainly preserves feature transformation, the middle zone contains more redundant layers, and the final zone contributes more to acceptance improvement.

During speculative decoding, LEAP formulates draft model construction as a sequential decision-making problem. MCTS incrementally explores layer-group actions, including **execute**, **skip**, and **repeat**, and directly uses real-time inference speedup as feedback. The best-performing configuration is then fixed to accelerate the remaining decoding process.

<p align="center">
  <img src="assets/leap_overview.png" width="95%">
</p>
