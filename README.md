# Optimizing Traffic Signal Control Using LLM-Driven Reward Weight Adjustment in Reinforcement Learning

| ![D3QN-LLM](./tsc_real/asset/D3QN-LLM%20framework.png) | ![Flow](./tsc_real/asset/Flow%20of%20the%20Proposed%20Methodology.png) |
|:--:|:--:|
| **Fig.1.D3QN-LLM Framework** | **Fig.2.Flow of the Proposed Methodology** |

## 📝 Research Overview

- This study utilizes a **Large Language Model (LLM)** to overcome the challenge of setting weights in multi-objective reward functions in reinforcement learning. By dynamically adjusting the reward function weights in real-time, the LLM enables the reinforcement learning agent to learn an optimal policy that maximizes rewards.  
- The experimental environment consists of a **single intersection**, with the goal of **minimizing Average Travel Time** through traffic signal control.  
- **Dueling Double DQN (D3QN)** is used as the reinforcement learning algorithm, while **GPT-4o-mini** is employed as the LLM for adjusting reward function weights.  
- The proposed model is referred to as **D3QN-LLM**, and its framework is illustrated in **Fig. 1**. The detailed process flow of the proposed method is shown in **Fig. 2**.  


## ❗️ Contributions of the Research

- By utilizing an **LLM** to dynamically address the weight-setting challenge in reward function design, a key aspect of reinforcement learning, this study proposes a new research direction.

## 🔧 Technology Stack and Environment

### 🚀 Key Technologies
- **Python**, **PyTorch (GPU Acceleration)** - Implementation of deep learning models and reinforcement learning  
- **SUMO (Simulation of Urban Mobility)** - Traffic simulation and vehicle data generation  
- **GPT-4o-mini (LLM API)** - Lightweight language model for traffic situation analysis  
- **LangChain Framework** - LLM prompt engineering and response control  

### 🛠 Development Environment
- **OS**: Ubuntu 22.04.5 / Windows (WSL supported)  
- **CUDA Support**: PyTorch acceleration in an NVIDIA GPU environment  
- **SUMO Version**: 1.20.0 or later recommended  
- **Python Environment**: Python 3.8 or later (venv virtual environment recommended)  

## 📂 Project Structure
```
📦D3QN-LLM
 ┣ 📂asset
 ┣ 📂d3qn_imgs
 ┣ 📂d3qn_models
 ┣ 📂eval_d3qn_imgs
 ┣ 📂eval_d3qn_txts
 ┣ 📂llm
 ┣ 📂logs
 ┣ 📂sumo
 ┃ ┣ 📂add
 ┃ ┣ 📂detectors
 ┃ ┣ 📂net
 ┃ ┣ 📂rou
 ┃ ┣ 📂trip
 ┣ 📜d3qn_agent.py
 ┣ 📜d3qn_tsc_main.py
 ┗ 📜tsc_env.py
```

## 📄 Citation

The following research papers and presentations are related to this project.  
If this research has been helpful, please cite the papers below.

### 📖 Scopus Journal
- **[Accepted]** **Sujeong Choi** and Yujin Lim, “Optimizing Traffic Signal Control Using LLM-Driven Reward Weight Adjustment in Reinforcement Learning,” *Journal of Information Processing Systems* (Editing/Proofreading in progress, Expected publication: February 2025).

### 🎤 Domestic Conference
- **[Oral Presentation]**  
  **최수정** and 임유진, "LLM을 이용한 강화학습기반 교차로 신호 제어," *한국정보처리학회 학술대회논문집*, vol. 31, no. 2, pp. 672-675, 2024.

## 📚 References

1. B. P. Gokulan and D. Srinivasan, "Distributed geometric fuzzy multiagent urban traffic signal control," *IEEE Transactions on Intelligent Transportation Systems*, vol. 11, no. 3, pp. 714-727, Sep. 2010. [DOI](https://doi.org/10.1109/TITS.2010.2050688)
2. H. Ceylan and M. G. H. Bell, "Traffic signal timing optimisation based on genetic algorithm approach including drivers’ routing," *Transportation Research Part B: Methodological*, vol. 38, no. 4, pp. 329-342, 2004. [DOI](https://doi.org/10.1016/S0191-2615(03)00015-8)
3. Sujeong Choi and Yujin Lim, "Reinforcement Learning-Based Traffic Signal Control Using Large Language Models," *Annual Conference of KIPS*, vol. 31, no. 2, pp. 672-675, 2024.
4. G. Zheng, X. Zang, N. Xu, H. Wei, Z. Yu, V. Gayah, et al., "Diagnosing reinforcement learning for traffic signal control," *arXiv preprint*, 2019. [DOI](https://doi.org/10.48550/arXiv.1905.04716)
5. H. Lee, Y. Han, Y. Kim, and Y. H. Kim, "Effects analysis of reward functions on reinforcement learning for traffic signal control," *PLoS ONE*, vol. 17, no. 11, 2022. [DOI](https://doi.org/10.1371/journal.pone.0277813)
6. S. Lai, Z. Xu, W. Zhang, H. Liu, and H. Xiong, "Large language models as traffic signal control agents: Capacity and opportunity," *arXiv preprint*, 2023. [DOI](https://doi.org/10.48550/arXiv.2312.16044)
7. A. Pang, M. Wang, M. O. Pun, C. S. Chen, and X. Xiong, "iLLM-TSC: Integration reinforcement learning and large language model for traffic signal control policy improvement," *arXiv preprint*, 2024. [DOI](https://doi.org/10.48550/arXiv.2407.06025)
8. P. A. Lopez, M. Behrisch, L. B. Walz, J. Erdmann, Y. P. Flotterod, R. Hilbrich, L. Lucken, J. Rummel, P. Wagner, and E. Wiessner, "Microscopic traffic simulation using SUMO," *Proceedings of the 2018 21st International Conference on Intelligent Transportation Systems (ITSC)*, pp. 2575-2582, 2018. [DOI](https://doi.org/10.1109/ITSC.2018.8569938)
9. L. Da, M. Gao, H. Mei, and H. Wei, "Prompt to transfer: Sim-to-real transfer for traffic signal control with prompt learning," *AAAI Conference on Artificial Intelligence*, vol. 38, pp. 82–90, 2024. [DOI](https://doi.org/10.1609/aaai.v38i1.27758)

## 📜 Usage and Copyright Notice

This project code was developed for personal research and experimental purposes and is stored in a public repository.  
This code **is not open-source** and may not be used for commercial purposes, copied, or distributed without permission.  


**Copyright (c) 2025 suzzang**