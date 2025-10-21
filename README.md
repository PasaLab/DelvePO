# DelvePO: Direction-Guided Self-Evolving Framework for Flexible Prompt Optimization

## 📖 Introduction

Welcome to the official implementation of **DelvePO: Direction-Guided Self-Evolving Framework for Flexible Prompt Optimization**! This framework is designed to optimize prompts for various tasks in a flexible and task-agnostic manner by decoupling prompts into distinct components and evolving them iteratively with memory-guided insights. DelvePO supports open- and closed-source large language models, such as GPT-4o-mini and DeepSeek-R1-Distill-Llama-8B, and has shown its effectiveness across diverse tasks.

Please refer to our **paper** for more experimental results and insights.

---

## 🚀 Quick Start

### ⚙️ Preparation

1. **Clone the Repository:**
   
   ```bash
   git clone https://anonymous.4open.science/r/DelvePO.git
   cd DelvePO
   ```
   
2. **Environment Setup:**
   Create and activate the environment using the following instructions:
   
   ```bash
   conda create -n delvepo python=3.10 -**y**
   conda activate delvepo
   pip install -r requirements.txt
   ```
   
3. **OpenAI API Key** (if needed):
   Add your OpenAI API key and other related settings to the `auth.yaml` file
   ```yaml
    api_key: YOUR_API_KEY
   ```

---

## 🏋️ Usage

### Training on Classification Tasks (CLS)
To train the framework on classification-based tasks such as CLS, run the script:

```bash
bash scripts/run_cls.sh
```

This script will execute the iterative evolution process using specified configurations, such as population size, number of epochs, and dataset.

---

## 🎮 Customization

You can customize parameters for running DelvePO by modifying the `run.py` script or using specific command-line arguments. The following are the most important parameters you might want to adjust:

### Arguments

- **`--population_size`**: The size of the prompt population.
- **`--epoch_size`**: The number of evolutionary epochs.
- **`--sample_num`**: The number of samples from the dataset used for evaluation.
- **`--factor_size`**: The size of the factor set, which consists of predefined prompt components (e.g., roles, task descriptions, output formats).
- **`--number_of_pairs`**: The size of the memory pair, which tracks component transformations for guiding evolution.
- **`--number_of_set`**: The size of the memory set, which stores top-performing prompts and metrics for optimization.
- **`--llm_type`** : The LLM used to evolve prompts. (e.g., `deepseek-r1-8b`, `qwen2.5-7b`, `gpt-4o-mini`, etc.) .
- **`--task`**: The type of task (e.g., classification `cls`, generation `nlg`, summarization `sum`).
- **`--dataset`**: The specific dataset used for evaluation (e.g., SST-5, SAMSum, etc.).
- **`--output_path`**: Directory where the output files will be saved.

---

## 🧠 How DelvePO Works

The core idea of DelvePO is to break down **prompts into components** (e.g., task roles, output format, constraints, etc.), which can be evolved and optimized separately. This modular design makes it easier to pinpoint which parts of the prompt need improvement.

### Key Features:
1. **Component-Based Prompts**:
   - Instead of treating a prompt as a single block of text, DelvePO splits it into smaller components. These components are optimized individually, allowing for fine-grained control and targeted evolution.

2. **Memory Mechanism**:
   - **Components Memory**: Tracks the performance of component value pairs before and after evolution. This helps focus on evolving the most impactful parts of the prompt.
   - **Prompts Memory**: Stores evolved prompts in descending order to guide future mutation and crossover. It comes in two flavors:
     - **Discrete Form**: Stores just the raw combinations of component values.
     - **Continuous Form**: Saves the full formatted prompts, preserving context.

By combining a **component-based design** with a **memory-driven optimization process**, DelvePO ensures continuous improvement in prompt quality while learning from past iterations. This makes the framework efficient and adaptable to a variety of tasks.

---

## 🔧 Code Structure

```plaintext
DelvePO/
├── DELVEPO.py          # Main DelvePO framework
├── run.py              # Main entry point for running experiments
├── scripts/            # Bash scripts for different tasks
├── data/               # Dataset folder with factors and split files
├── utils.py            # Utility functions and tools
├── llm_utils.py        # Tools for interacting with large language models
├── memory_update.py    # Memory module to log and refine factor sets
├── pattern.py          # Prompt templates and patterns
├── auth.yaml           # File to store API keys
└── requirements.txt    # Python dependencies
```

## ✏️ Example Template

Below is an example classification prompt template:

```plaintext
You are a <role>{role}</role>. Given the Sentence, your task is to <task_description>{task_description}</task_description>.
<Output format>: {output_format}
<Workflow>: {workflow}
<Example>: {examples}
```

During runtime, these placeholders (e.g., `<role>`, `<workflow>`) are dynamically replaced with factors designed to enhance task performance.

---

## 🤝 Acknowledgments
We built our codebase upon the following repositories. We greatly appreciate their open-source contributions!

- [APE](https://github.com/keirp/automatic_prompt_engineer)
- [EvoPrompt](https://github.com/beeevita/EvoPrompt)
- [promptbench](https://github.com/microsoft/promptbench)
