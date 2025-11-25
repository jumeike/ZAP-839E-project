# **ZAP 839E – Course Project**

## **Overview**

This project develops **ZAP**, a zero-shot adversarial prompting benchmark designed to evaluate the jailbreak vulnerability of **multimodal large language models (MLLMs)**. We curate harmful-context images from **SPA-VL** and **BeaverTails-V**, organize them into high-risk categories (e.g., Illegal Activities, Security Threats, False Information), and pair them with handcrafted prompt templates to test whether models can be coerced into producing unsafe outputs **without any fine-tuning**. We evaluate several state-of-the-art MLLMs to measure how visual context and prompt structure influence safety failures. 

### **Datasets Used**

* **SPA-VL** – Hierarchical safety-labeled multimodal dataset
* **BeaverTails-V** – Harmful visual behavior dataset curated for safety alignment
  Both provide diverse, safety-sensitive visual scenarios across multiple domains.

### **Models Evaluated**

* **LLaMA3.2-VL (11B)**
* **Qwen2.5-VL (3B)**
* **Qwen2.5-VL (7B)**

These models are tested on curated images and adversarial prompts to quantify **Attack Success Rate (ASR)** across categories.

---

## **Prompt Templates**

We include category-specific prompt templates extracted from the project’s prompt collection.
Examples include templates for:

* **Animal Abuse / Environmental Damage**
* **Financial or Academic Fraud**
* **Privacy Violations**
* **False Information / Erosion of Trust**
* **Illegal Activities / Extremism**
* **Security Threats / Digital Crime**

---


## **Repository Contents**

```
datasets/             # Extracted category-specific image folders
scripts/              # Filtering and prompt-generation scripts
figures/              # Distribution plots and analysis figures
prompt_template/      # Prompt templates for all harm categories
responses/            # LLM responses to prompt injections from different categories
README.md             # Project documentation
```

---



