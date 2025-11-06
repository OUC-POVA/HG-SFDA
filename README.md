# HG-SFDA: HyperGraph Learning Meets Source-free Unsupervised Domain Adaptation

Code for our paper **"HG-SFDA: HyperGraph Learning Meets Source-free Unsupervised Domain Adaptation"**

## Contributions

- We formulate the source-free unsupervised domain adaptation (SFDA) as a hypergraph learning problem and explore the high-order neighborhood relations among target samples to excavate the underlying structural information.
- With the constructed hypergraph, we design a novel self-loop strategy to elegantly involve the domain shift into optimization.
- We describe an adaptive learning scheme to enhance the mainstream objectives by considering different attention levels.

## Code Environment

- **PyTorch:** 1.13.1 with CUDA 11.6
- **Scikit-learn:** 0.24.2
- **Other dependencies:**
  - numpy: 1.26.3
  - cvxpy
  - tqdm

## Instructions

1. Download Office-Home datasets and change the path in the code to it.
2. To train the model on the source domain, directly run `train_src_on.sh`.
3. For source-free domain adaptation, directly run `train_tar_on.sh`.
