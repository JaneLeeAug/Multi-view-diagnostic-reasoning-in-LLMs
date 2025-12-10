#
# Copyright [2024-] [Unsloth AI, Daniel Han-Chen & Michael Han-Chen]
# Copyright [2025] [National Taiwan University Natural Language Processing Lab] (modifications)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# **MODIFICATION NOTICE:**
# This file has been modified by [National Taiwan University Natural Language Processing Lab]. 
# **Key modifications include:**
#
# 1. **Data Loading:** Adjusted the method for reading and processing the dataset to fit new formats/requirements.
# 2. **Execution Method:** Changes made to the program's execution logic and structure (e.g., adapted for a different computing environment or API usage).
#
# All other terms and conditions of the Apache License, Version 2.0 apply.

"""
!uv pip install "torch>=2.8.0" "triton>=3.4.0" numpy torchvision bitsandbytes "transformers>=4.55.3"
!uv pip install "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo"
!uv pip install "unsloth[base] @ git+https://github.com/unslothai/unsloth"
!uv pip install git+https://github.com/triton-lang/triton.git@05b2c186c1b6c9a08375389d5efe9cb4c401c075#subdirectory=python/triton_kernels
!uv pip install --upgrade --no-deps transformers==4.56.2 tokenizers
!uv pip install --no-deps trl==0.22.2
"""

import os
import sys
import torch
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

# ---------- Set HF cache ----------
hf_cache_dir = ""
os.environ["HF_HOME"] = hf_cache_dir
os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
os.environ["HF_DATASETS_CACHE"] = hf_cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
os.makedirs(hf_cache_dir, exist_ok=True)
print(f"HF_HOME set to: {hf_cache_dir}")

os.environ["LD_LIBRARY_PATH"] = "/usr/lib64-nvidia:" + os.environ.get("LD_LIBRARY_PATH", "")

# ---------- Data paths ----------
TRAINING_DATA_PATH = ""
INFERENCE_DATA_PATH = ""

# Check if files exist
if not os.path.exists(TRAINING_DATA_PATH):
    sys.exit(f"Error: Training data not found at: {TRAINING_DATA_PATH}")
if not os.path.exists(INFERENCE_DATA_PATH):
    print(f"Warning: Inference data not found at: {INFERENCE_DATA_PATH}. Please ensure it exists or download manually.")

# ---------- Load model ----------
print("--- Loading FastLanguageModel ---")
max_seq_length = 1024
dtype = None
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    dtype=dtype,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    full_finetuning=False,
)

# Apply PEFT/LoRA
print("--- Applying PEFT/LoRA ---")
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ---------- Load and process training data ----------
print(f"--- Loading training data: {TRAINING_DATA_PATH} ---")
dataset = load_dataset(
    "csv",
    data_files=TRAINING_DATA_PATH,
    split="train"
)

def format_prompts(examples):
    convos = [json.loads(c) if isinstance(c, str) else c for c in examples["message"]]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True, num_proc=4)
print("--- Training data formatted ---")

# ---------- SFT Trainer ----------
print("--- Starting SFT training ---")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=51,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)
trainer_stats = trainer.train()
print("--- Training completed ---")

# ---------- Inference function ----------
prefix = "Provide a step-by-step deduction that identifies the correct response.\n\n"

def chat_with_model(model, tokenizer, user_content, system_prompt="You are a medical assistant, an expert in clinical reasoning and evidence-based diagnosis."):
    full_user_content = prefix + user_content
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_user_content},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="medium",
    ).to("cuda")

    output_ids = model.generate(**inputs, max_new_tokens=4096)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

# ---------- Inference and save results ----------
print(f"--- Loading inference data: {INFERENCE_DATA_PATH} and starting inference ---")
try:
    df = pd.read_csv(INFERENCE_DATA_PATH)
except FileNotFoundError:
    sys.exit(f"Error: Inference data not found at: {INFERENCE_DATA_PATH}")

if 'output' not in df.columns:
    df['output'] = ""

# Filter rows with empty output
rows_to_infer = df[df['output'].isna() | (df['output'].astype(str).str.strip() == '')]
total_to_infer = len(rows_to_infer)
print(f"✅ Total {len(df)} rows, {total_to_infer} rows have empty output and will be inferred.")

for original_idx, row in tqdm(rows_to_infer.iterrows(), total=total_to_infer, desc="Running model inference (real-time write)"):
    user_content = row['input']
    try:
        # Call model once
        output_text = chat_with_model(model, tokenizer, user_content)
        df.at[original_idx, 'output'] = output_text
    except Exception as e:
        print(f"\n❌ Warning: index {original_idx} inference failed, error: {e}.")
        df.at[original_idx, 'output'] = None

    # Save CSV after each inference (real-time write)
    df.to_csv(INFERENCE_DATA_PATH, index=False)

print(f"--- Inference completed. Results saved at: {INFERENCE_DATA_PATH} ---")
print("--- Script finished ---")
