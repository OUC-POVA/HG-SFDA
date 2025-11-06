# 📘 Project Title: [HG-SFDA: HyperGraph Learning Meets Source-free Unsupervised Domain Adaptation]

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](link_to_paper.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-red)](https://arxiv.org/abs/xxxx.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/your_username/your_repo.svg?style=social)](https://github.com/your_username/your_repo)

---

## 🌟 Overview
This repository contains the official implementation of the paper:

> **[HG-SFDA: HyperGraph Learning Meets Source-free Unsupervised Domain Adaptation]**  
> *Jinkun Jiang, Qingxuan Lv, Yuezun Li, IEEE Member, Yong Du, IEEE Senior Member, Junyu Dong, IEEE
Member, Sheng Chen, IEEE Life Fellow, Hui Yu, IEEE Senior Member*  
> Published in *[IEEE Transactions on Image Processing, 2025]*

🎯 **Goal:** [Source-Free unsupervised Domain Adaptation (SFDA) aims to classify target samples by only accessing a pre-trained source model and unlabelled target samples. Since no source data is available, transferring the knowledge from the source domain to the target domain is challenging. Existing methods normally exploit the pair-wise relation among target samples and attempt to discover their correlations by clustering these samples based on semantic features. The drawbacks of these methods include: 1)~the pair-wise relation is limited to exposing the underlying correlations of two more samples, hindering the exploration of the structural information embedded in the target domain; and 2)~the clustering process only relies on the semantic feature, while overlooking the critical effect of domain shift, \ie, the distribution differences between the source and target domains."]

🧠 **Core Idea:** [we propose a new SFDA method that exploits the high-order neighborhood relation and explicitly takes the domain shift effect into account. Specifically, we formulate the SFDA as a hypergraph learning problem and construct hyperedges to explore the {deep structural} and context information among multiple samples. Moreover, we integrate a self-loop strategy into the constructed hypergraph to elegantly introduce the domain uncertainty of each sample. By clustering these samples based on hyperedges, both the semantic feature and domain shift effects are considered. We then describe an adaptive relation-based objective to tune the model with soft attention levels for all samples."]

---

## 🏗️ Method Overview

<p align="center">
  <img src="doc/Overview.png" width="80%" alt="Architecture Diagram">
</p>

**Figure 1:** Overview of the proposed architecture. (Replace this figure with your own diagram.)

### 🔍 Key Contributions
- **High-order Relation Modeling:** Capture cross-scale interactions using hypergraph edges.  
- **Adaptive Fusion:** Weighted strategy to balance shallow and deep feature representations.  
- **Lightweight Design:** Maintain YOLO efficiency while enhancing semantic structure understanding.  

Mathematical formulation example:

$$
H = \\sigma( D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X )
$$

where $H$ is the incidence matrix, $W$ is the hyperedge weight, and $X$ represents node features.

---

## 📁 Repository Structure
