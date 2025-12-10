Clinical Diagnosis Needs More Than One Mind: Multi-View Diagnostic Reasoning in Large Language Models
---

Clinical diagnosis requires integrating multiple complementary reasoning strategies. This project explores two approaches to enable LLMs to “reason with more than one mind”:

- **Collaborative Multi-Agent Diagnostic Reasoning**
- **Internalized Multi-View Reasoning**

## 🔍 Overview

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/JaneLeeAug/Multi-view-diagnostic-reasoning-in-LLMs.git
cd Multi-view-diagnostic-reasoning-in-LLMs

# Install dependencies
pip install -r requirements.txt
```
## 📁 Repository Structure

Multi-view-diagnostic-reasoning-in-LLMs/
├── Collaborative Multi-Agent Reasoning/
│   ├── GPT-4_multi-agent.xlsx
│   ├── GPT-4o_multi-agent.xlsx
│   ├── multi-agent_round1.ipynb
│   └── multi-agent_round2.ipynb
├── Internalized Multi-View Reasoning/
│   ├── data/
│   │   ├── gpt-4o_training_data.xlsx
│   │   ├── gpt-oss-120b_training_data.xlsx
│   │   └── results.xlsx
│   └── script/
│       ├── gpt-oss-20b.py
│       ├── gpt-oss-120b.py
│       ├── llama-3.1_8b.ipynb
│       ├── mistral-7b-instruct-v0.3.ipynb
│       ├── phi-4-mini-instruct.ipynb
│       └── gpt-oss-120b_API.ipynb
├── README.md
└── requirements.txt

## 🚀 Usage

### Collaborative Multi-Agent Diagnostic Reasoning

To run **Collaborative Multi-Agent Reasoning (CMAR)** and **Independent Multi-Agent Reasoning (IMAR)**:

1. Execute `multi-agent_round1.ipynb` to run IMAR.
2. Execute `multi-agent_round2.ipynb` to run CMAR, which builds upon IMAR results.

This two-step process ensures that agents first reason independently and then collaborate to refine their diagnosis.

### Internalized Multi-View Reasoning

1. The teacher model generates training data using **`multi-agent_round1.ipynb`** and **`gpt-oss-120b_API.ipynb`**, saved as:

   - `gpt-4o_training_data.xlsx`
   - `gpt-oss-120b_training_data.xlsx`

   Either dataset can be converted into a CSV file for training.

2. The dataset used to evaluate the student model is **`testing_data.csv`**.

3. Run the corresponding script or Colab notebook depending on the student model:

   | Student Model                   | Script / Notebook |
   |---------------------------------|-------------------|
   | **gpt-oss-20b**                 | `gpt-oss-20b.py` |
   | **gpt-oss-120b**                | `gpt-oss-120b.py` |
   | **LLaMA 3.1 8B**                | [Open Notebook](https://colab.research.google.com/drive/1G2wBf3C9V4Ita5O1TZLKn9JxyVFkWzPd?usp=sharing) |
   | **Mistral 7B Instruct v0.3**    | [Open Notebook](https://colab.research.google.com/drive/1Uz6vhClCYjFxn5h-aj9cua7wJ96Sc32L?usp=sharing) |
   | **Phi-4 Mini Instruct**         | [Open Notebook](https://colab.research.google.com/drive/1AhWKg44x_1Ssmpn655V3SU5cKLYteRBI?usp=sharing) |

4. The responses of both raw and fine-tuned student models are summarized in **`results.xlsx`**.
