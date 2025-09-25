# Goal-Conditioned Value Learning for Token-Level Rewarding (VERL)

This repository contains code and experiments for **goal-conditioned value functions** with **token-level reward shaping** for large language models (LLMs).  
The work extends **Value-Enhanced Reinforcement Learning (VERL)** and aims to provide better **credit assignment** and **controlled decoding** for sequence generation.

---

##  Overview

Traditional RLHF setups reward whole sequences, but this often fails to give **fine-grained credit** at the **token level**.  
This repo explores:

- **Goal-Conditioned Value Functions (GCVF)** – learning a value function conditioned on explicit user goals or prompts.
- **Token-Level Rewarding** – computing token-wise advantages and rewards to guide training and decoding.
- **Controlled Decoding** – integrating learned token values into inference to bias generation toward desirable outcomes.

The repository supports **PPO**, **TRPO**, and **custom VERL-style algorithms** for experiments.

---

## ✨ Features

- ✅ **Goal-conditioned value estimation** for LLMs  
- ✅ **Token-level reward shaping** with advantage functions  
- ✅ **Controlled decoding** using value functions
- ✅ Plug-and-play 
- ✅ Modular code for **experiments & reproducibility**  
 

---

## 📂 Project Structure
