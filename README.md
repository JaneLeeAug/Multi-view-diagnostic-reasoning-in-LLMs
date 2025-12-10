Clinical Diagnosis Needs More Than One Mind: Multi-View Diagnostic Reasoning in Large Language Models
---

Clinical diagnosis requires integrating multiple complementary reasoning strategies. This project explores two approaches to enable LLMs to “reason with more than one mind”:

- **Collaborative Multi-Agent Diagnostic Reasoning**
- **Internalized Multi-View Reasoning**

## 🔍 Overview

## 📁 Repository Structure

## 🚀 Usage

### Collaborative Multi-Agent Diagnostic Reasoning

To run **Collaborative Multi-Agent Reasoning (CMAR)** and **Independent Multi-Agent Reasoning (IMAR)**:

1. Execute `multi-agent_round1.ipynb` to run IMAR.
2. Execute `multi-agent_round2.ipynb` to run CMAR, which builds upon IMAR results.

This two-step process ensures that agents first reason independently and then collaborate to refine their diagnosis.

### Internalized Multi-View Reasoning

1. The teacher model generates training data through **`multi-agent_round1.ipynb`** and **`gpt-oss-120b_API.ipynb`**, stored respectively in:

- `gpt-4o_training_data.xlsx`
- `gpt-oss-120b_training_data.xlsx`

You may choose either dataset and convert it into a CSV file for training.

2. The file used to evaluate student model performance is **`testing_data.csv`**.

3. Run the following scripts or notebooks depending on the student model:

- `python gpt-oss-20b.py`
- `python gpt-oss-120b.py`
- **LLaMA 3.1 8B (Colab):** https://colab.research.google.com/drive/1G2wBf3C9V4Ita5O1TZLKn9JxyVFkWzPd?usp=sharing
- **Mistral 7B Instruct v0.3 (Colab):** https://colab.research.google.com/drive/1Uz6vhClCYjFxn5h-aj9cua7wJ96Sc32L?usp=sharing
- **Phi-4 Mini Instruct (Colab):** https://colab.research.google.com/drive/1AhWKg44x_1Ssmpn655V3SU5cKLYteRBI?usp=sharing

## 🔗 Citation
