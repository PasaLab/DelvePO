from llm_utils import generate_metric, build_prompt
from pattern import *
from utils import *
import random

task = "cls"
if task == "cls":
    random.seed(5)
    test_data = read_lines("./data/cls/sst-5/test.txt", sample_indices=None)
    test_src = []
    test_tgt = []
    verbalizers = get_dataset_verbalizers("sst-5")
    for i, line in enumerate(test_data):
        try:
            cur_src, cur_tgt = line.split('\t')
        except:
            raise ValueError
        test_src.append(cur_src)
        test_tgt.append(verbalizers[int(cur_tgt)])
    sample_indices = random.sample(range(len(test_src)), 100) 
    print("sample_indices:", sample_indices)
    dev_src = [test_src[i] for i in sample_indices]
    dev_tgt =[test_tgt[i] for i in sample_indices]
    factor_types = ["role", "task_description", "output_format", "output_format", "example"]
    top_k_prompts = [] # top-k prompts after evolution
    for i, item in enumerate(top_k_prompts[:5]):
        prompt, dev_score = item[0], item[1]
        prompt_contiue = build_prompt(factor_types, PROMPT_For_cls, prompt)
        test_score = generate_metric(dev_src, dev_tgt, prompt, ["role", "task_description"], task="cls", dataset="sst-5")
        print(f"Prompt {i}: {prompt_contiue} - dev_score: {dev_score} - test_score: {test_score}")
