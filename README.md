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

🎯 **Goal:** [Existing methods normally exploit the pair-wise relation among target samples and attempt to discover their correlations by clustering these samples based on semantic features. The drawbacks of these methods include: 1)~the pair-wise relation is limited to exposing the underlying correlations of two more samples, hindering the exploration of the structural information embedded in the target domain; and 2)~the clustering process only relies on the semantic feature, while overlooking the critical effect of domain shift, \ie, the distribution differences between the source and target domains."]

🧠 **Core Idea:** [We propose a new SFDA method that exploits the high-order neighborhood relation and explicitly takes the domain shift effect into account. Specifically, we formulate the SFDA as a hypergraph learning problem and construct hyperedges to explore the {deep structural} and context information among multiple samples. Moreover, we integrate a self-loop strategy into the constructed hypergraph to elegantly introduce the domain uncertainty of each sample. By clustering these samples based on hyperedges, both the semantic feature and domain shift effects are considered. We then describe an adaptive relation-based objective to tune the model with soft attention levels for all samples."]

---

## 🏗️ Method Overview

<p align="center">
  <img src="doc/Overview.png" width="90%" alt="Architecture Diagram">
</p>

**Figure 1:** Overview of the proposed architecture.

### 🔍 Key Contributions  
- **High-order Relation Modeling:**  Different from the existing pair-wise relation-based methods, \eg, NRC, AaD, SF(DA)$^{2}$, we formulate SFDA as a hypergraph learning problem and explore the high-order neighborhood relations to excavate the underlying structural information.
- **Self-Loop:** With the constructed hypergraph, we design a novel self-loop strategy to elegantly involve the domain shift into optimization.
- **Adaptive Scheme:** We describe an adaptive learning scheme to enhance the mainstream objectives by considering different attention levels.

where $H$ is the incidence matrix, $W$ is the hyperedge weight, and $X$ represents node features.

---

## 📁 Repository Structure

.
├── README.md
├── LICENSE
├── docs/
│ ├── architecture.png
│ ├── teaser.png
│ └── results_comparison.png
├── configs/
│ └── yolov8_amsh.yaml
├── data/
│ └── (dataset layout)
├── models/
│ ├── backbone.py
│ ├── neck_amsh.py # 超图构建与超图卷积模块
│ └── head.py
├── scripts/
│ ├── train.py
│ ├── val.py
│ └── detect.py
├── tools/
│ └── convert_coco_to_yolo.py
├── requirements.txt
└── results/
├── logs/
└── plots/

---

# 🚀 快速开始（Quick start）

## 环境（推荐）
```bash
# 创建 conda 环境（示例）
conda create -n hyperyolo python=3.10 -y
conda activate hyperyolo
pip install -r requirements.txt

data/
  images/
    train/
    val/
  labels/
    train/
    val/
