# Novelty boundary and prior art

This repository does not claim that FFT-based circulant neural layers or Langevin EBMs are new inventions.

## Established components

- [An exploration of parameter redundancy in deep networks with circulant projections](https://openaccess.thecvf.com/content_iccv_2015/html/Cheng_An_Exploration_of_ICCV_2015_paper.html) describes circulant projections for fully connected networks and the same asymptotic storage/arithmetic reductions.
- [CirCNN](https://arxiv.org/abs/1708.08917) describes block-circulant neural weights and FFT-based multiplication.
- [Improved Contrastive Divergence Training of Energy Based Models](https://proceedings.mlr.press/v139/du21b/du21b.pdf) discusses EBM training stability and Langevin-based negative sampling.
- [Block Circulant Adapter for Large Language Models](https://www.ijcai.org/proceedings/2025/560) and [Compact Circulant Layers with Spectral Priors](https://arxiv.org/abs/2602.21965) show that related spectral/block-circulant neural architectures remain active research topics.

## Current conclusion

The broad combination “replace dense EBM layers with circulant FFT layers” is not currently treated as a novel claim. A specific paper contribution could still be possible, but it must be narrower and evidenced by a new theorem, a non-obvious sampler/objective, or a technically specific proof-state interface. A search result that does not contain the exact phrase “circulant EBM” is not sufficient to establish novelty.

Patent novelty, inventive step, scientific novelty, and practical usefulness are separate questions. If patent protection is intended, obtain professional advice before publicly releasing enabling details; public disclosure can affect rights differently across jurisdictions.
