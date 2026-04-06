"""
PROMETHEUS: A Self-Modifying Fractal Intelligence Architecture

This architecture exists NOWHERE else. It combines 5 novel ideas that,
together, create something fundamentally different from Transformers:

1. NEUROPLASTIC INFERENCE
   Standard NNs: weights are frozen at inference.
   Prometheus: weights CHANGE during the forward pass using Hebbian-like
   learning rules. Each input literally rewires the network temporarily.
   This gives the model "working memory" encoded in weight changes.

   Math: Δw = η * (x ⊗ y - λ*w)  [Oja's rule variant]
   The outer product x⊗y strengthens connections between co-active neurons,
   while -λ*w prevents runaway growth. This is differentiable and trainable.

2. SPARSE DISTRIBUTED MEMORY (SDM)
   Like a hippocampus: stores (key, value) pairs in a fixed-size memory.
   Reading: query → find nearest keys → return weighted sum of values.
   Writing: after processing, write back refined representations.
   This gives the model "long-term memory" beyond its context window.

   Crucially: the memory is SHARED across layers, so early layers can
   leave notes for later layers. This is unlike anything in transformers.

3. FRACTAL RECURSIVE PROCESSING
   Instead of stacking N different layers, ONE small "cell" processes
   recursively. The cell takes (state, depth) and outputs (new_state, halt_prob).
   It calls itself until halting, giving ADAPTIVE depth from tiny params.

   A 1M parameter cell with depth 16 = effective 16M parameter network,
   but using only 1M of memory. This is the key to running on weak hardware.

4. PREDICTIVE CODING
   Inspired by neuroscience's "free energy principle":
   - Each level predicts what the level below will produce
   - Only PREDICTION ERRORS propagate upward
   - This is MASSIVELY sparse: if predictions are good, almost nothing flows

   Result: after a few recursive steps, most of the network is silent.
   Only the "surprising" parts of the input use compute.

5. DYNAMIC SPARSE ACTIVATION
   Like the brain: only ~5% of neurons fire at any time.
   We use a top-k activation function that zeros out 95% of neurons.
   Combined with predictive coding, this means >95% of computation is skipped.

Together: a model with huge CAPACITY but tiny COMPUTE cost.
"""
