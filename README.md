# Amine Chalghoum

Machine learning engineer in Aachen.

Most of my work circles the same question: what does a model get to condition on, and when. Smoothing lets a world model see the future during training to improve the lower bound of its ELBO. Neighborhood attention decides how far a token can look before you pay quadratic cost. Retrieval decides what an LLM is allowed to know. Different domains, one design axis — and it's usually where the interesting constraints live.

---

## [daydream-playground](https://github.com/achalghoum/daydream-playground)
*Smoothed discrete latent dynamics for uncertainty quantification in model-based RL*

A latent world model trained with causal inference only sees the past when it infers `z_t`, so it has to explain away observation noise it cannot yet distinguish from genuine dynamics. It compensates by inflating aleatoric uncertainty. The Variational Recurrent Kalman Network fixes this with exact Kalman smoothing — condition the training posterior on future observations, get a calibrated filter — but exactness costs you linear-Gaussian latents, which is the wrong prior for discrete-action, visually complex domains like Atari. DreamerV2's categorical RSSM fits those domains, and reintroduces the problem.

The thesis asks whether smoothing survives the move from exact to approximate. I take three smoothing structures from the dynamical VAE literature — SRNN, DKF and RVAE, which differ precisely in what the backward network conditions on (latent feedback, observation stream, or the recurrent state that already summarizes both) — and retrofit each onto a categorical RSSM, with a Gaussian variant for continuous control. DSAE, VRNN and STORN are implemented as ablations.

Training runs two passes: a causal forward pass identical to DreamerV2, then a non-causal backward network that re-estimates the posterior with access to future observations. The ELBO is computed against the smoothed posterior, and a three-way KL decomposition — smoother against prior, smoother against causal posterior, causal posterior against prior — pulls the deployment-time filter toward the smoothed one. The backward network is discarded at rollout, so control stays causal. You buy calibration with training compute, not with test-time information you won't have.

Evaluated on DeepMind Control and Atari under clean and occluded observations, on return, calibration and cost. Smoothing helps, but the effect is domain-dependent and vanishes when policy collapse narrows the replay distribution — the smoother has no future left to exploit once the agent stops exploring.

## [eyetoy](https://github.com/achalghoum/eyetoy)
*Multi-scale neighborhood attention (MSNAT)*

Neighborhood attention buys linear cost by restricting each token to a local window, then recovers global context by stacking layers until receptive fields overlap. That's a slow way to see far, and it forces every head in a layer to agree on one notion of "near."

MSNAT splits each layer's heads into parallel groups with different token-merging factors and window sizes — 1×/2×/4×/8× merging against windows of 17/13/11/9 in early layers — so a single block reads fine detail and coarse structure at once, and the receptive field grows exponentially with depth while cost stays linear. Global information travels through learnable register tokens on a separate O(N) message-passing path, under a `flex_attention` block mask that lets spatial tokens reach registers but not each other. The whole stack is dimension-generic: 1D, 2D and 3D share the same attention code.

Trained from scratch with no pretraining. 3M parameters reaches 89% top-1 on CIFAR-10 and 70% on CIFAR-100; a 35M model reaches 42% top-1 on ImageNet-1k after nine epochs. Written up as a paper proposal.

## [spool](https://github.com/achalghoum/spool)
*Bidirectional temporal attention for video*

Causal video models are the norm because prediction is the usual objective, but recognition isn't prediction — deciding what happened at frame *t* is easier if you have seen frame *t+5*. Spool takes that seriously and splits attention heads into lookback and lookahead groups over a local temporal window, with the split configurable, so the model's temporal asymmetry becomes a hyperparameter rather than an assumption.

Global temporal context arrives through learnable context frames appended at the sequence boundaries: every frame attends to them, they attend to everything, and the cost stays linear in sequence length instead of quadratic. Frame-level features come from a frozen DINOv2 backbone, which keeps the learned part purely temporal.

## [dewey](https://github.com/achalghoum/dewey)
*Diversity-aware multi-stage retrieval*

Top-*k* dense retrieval optimizes relevance and gets redundancy for free: the *k* nearest neighbours of a query are often near each other too, so the LLM receives one perspective restated *k* times. Dewey treats context selection as a coverage problem instead of a ranking one.

Candidates from a dense FAISS pass are embedded spectrally (Nyström-approximated, so the eigendecomposition scales) and clustered into thematic groups, which makes the query's distinct facets explicit rather than implicit in the score ordering. Within each cluster a PageRank pass over the affinity graph finds documents that are central to their own theme rather than merely close to the query, and selection takes a few from each cluster — diversity enforced structurally, not through a penalty term. A sparse BM25 pass then locates the relevant chunks inside the selected documents, and a learned re-ranker scores them.

Graph construction, spectral embedding, clustering and PageRank are written from scratch in PyTorch so the whole pipeline stays on the GPU and differentiable end to end.

---

## Elsewhere

The day job is production Python — four years of it, currently on a knowledge-graph platform built around an eight-stage LLM extraction pipeline. Things I've picked up along the way that don't fit above:

- **Extraction pipelines that admit they're wrong.** Append-only claim ledgers with provenance and retraction, human-in-the-loop review for low-confidence merges, readable views as replayable projections.
- **Entity resolution** at production scale, and MCP servers for exposing structured data to agents.
- **Backends.** FastAPI, PostgreSQL, SQLAlchemy, multi-tenant isolation, AWS serverless, Docker.
- **Messy real-world input.** OCR with fuzzy matching, German free-text parsing, rules-based validation.
- **Languages.** Python, TypeScript, Java, C#. Arabic, English, German, French.

amine.chalghoum@gmail.com
