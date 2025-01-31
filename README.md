<div align="center">
<img src="https://github.com/user-attachments/assets/50da1db8-cbed-4c89-9010-6392c60b3574" width="200" />

<h1> IpomoeaAquatica <br> Reinforcement Learning for Space Invaders </h1>

 [Luigina Costante](https://github.com/Luigina2001), [Angelo Nazzaro](https://github.com/angelonazzaro)
</div>

**IpomoeaAquatica** is a reinforcement learning project designed to train an intelligent agent to play **Space Invaders** using **Q-Learning, Deep Q-Network (DQN), and Asynchronous Advantage Actor-Critic (A3C)**. The project explores different RL strategies to evaluate their efficiency in a dynamic gaming environment.  

## 📌 Features  
- **Environment**: OpenAI Gym's `SpaceInvaders-v5`  
- **Algorithms Implemented**:  
  - 🟢 **Q-Learning** – A tabular baseline for comparison  
  - 🟦 **DQN** – Deep learning-based Q-learning with CNNs  
  - 🟥 **A3C** – A parallelized actor-critic approach for on-policy learning  
- **Custom Wrappers** for observation, action processing, and adaptive reward shaping  
- **Early Stopping** for optimized training  

## 🏷️ Setup  
1. Clone the repository:  
   ```bash
   git clone https://github.com/Luigina2001/IpomoeaAquatica.git
   cd IpomoeaAquatica
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Train an agent:  
   ```bash
   python train.py --agent DQN  # Options: Q-Learning, DQN, A3C
   ```
4. Evaluate the trained model:  
   ```bash
   python test_agent.py --checkpoint_path saved_model.pth --model DQN
   ```

## 📊 Comparative Analysis of Results  

| Algorithm | Learning Stability | Score Improvement | Sample Efficiency | Convergence Time |
|-----------|-------------------|------------------|------------------|----------------|
| **Q-Learning** | 🚨 Unstable | 📉 Low | ❌ Inefficient | ⏳ Very slow |
| **DQN** | ⚖️ Moderate | 📈 Higher than Q-Learning | ✅ Improved with replay buffer | 🕒 Faster than Q-Learning |
| **A3C** | 🟢 Stable | 🚀 Best | 🔥 Highly efficient | ⚡ Fastest |

- **Q-Learning** serves as a baseline but struggles with the high-dimensional state space.  
- **DQN** improves learning stability and performance through experience replay and CNN-based feature extraction.  
- **A3C** achieves **the best results** by leveraging parallel actor-learners, enabling faster convergence and **higher final scores**.  

## 🌟 Key Insights  
- **A3C demonstrated superior performance**, achieving higher scores with more stable learning dynamics.  
- **DQN improved upon Q-Learning**, but suffered from occasional instability due to experience replay sampling.  
- **Early stopping helped optimize resource usage**, preventing unnecessary training when convergence was detected.  

## 🌝 Citation  
If you use this project, please cite:  
```
@article{costantenazzaro2025:ipomoea_aquatica,
  author    = {Luigina Costante, Angelo Nazzaro},
  title     = {IpomoeaAquatica: An Intelligent Agent for Space Invaders},
  year      = {2025},
  institution = {University of Salerno}
}
```

