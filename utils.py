from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from easse.sari import corpus_sari
from mosestokenizer import *
from rouge import Rouge
import random
import yaml

def roulette_wheel_selection(population):
    base_weight = 1.0 / len(population) 
    weights = [tup[1] for tup in population]
    total_weight = sum(weights) + 1
    selection_probabilities = [(weight+1) / total_weight for weight in weights]
    selected_index = random.choices(range(len(population)), weights = selection_probabilities)[0]
    return population[selected_index][0]

def cal_sari(orig_sents, sys_sents, refs_sents):
    sari = corpus_sari(orig_sents=orig_sents,  
                sys_sents=sys_sents, 
                refs_sents=refs_sents)
    return sari

def cal_cls_score(pred_list, label_list,metric='acc'):
    pred_list = [p.lower() for p in pred_list]
    label_list = [l.lower() for l in label_list]
    if metric == 'f1':
        score = f1_score(label_list, pred_list, average='macro')
    elif metric == 'acc':
        score = accuracy_score(label_list, pred_list)
    elif metric == 'mcc':
        score = matthews_corrcoef(label_list, pred_list)
        score = (score + 1) / 2  # normalize to [0, 1]
    return score

def cal_rouge(output_texts, ref_texts, output=None):
    print("calculating rouge score...")
    rouge = Rouge()
    for i in range(len(output_texts)):
        if not output_texts[i].strip():
            output_texts[i] = "None"
    output_texts = [" ".join(MosesTokenizer('en')(sent)) for sent in output_texts]
    ref_texts = [" ".join(MosesTokenizer('en')(sent)) for sent in ref_texts]
    for i in range(len(output_texts)):
        if not output_texts[i].strip() or output_texts[i].strip().strip('.') == '':
            output_texts[i] = "None"
    assert len(output_texts) == len(ref_texts), f"Length mismatch: {len(output_texts)} vs {len(ref_texts)}"
    if output:
        with open(output, "a") as f:
            for i in range(len(output_texts)):
                f.write("\n============ new cal_rouge =============\n")
                f.write(f"({i}) Output: {output_texts[i]}\n({i}) Reference: {ref_texts[i]}\n\n")
    try:
        print("Calculating scores...")
        scores = rouge.get_scores(output_texts, ref_texts, avg=True, ignore_empty=True)
    except Exception as e:
        print(f"Error calculating ROUGE scores: {e}")
        scores = {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
    if output:
        with open(output, "a") as f:
            f.write("=========scores:======\n")
            f.write(f"ROUGE-1: {scores['rouge-1']['f']}\n")
            f.write(f"ROUGE-2: {scores['rouge-2']['f']}\n")
            f.write(f"ROUGE-L: {scores['rouge-l']['f']}\n")
            f.write("======================\n")
    return scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f'] 

def read_lines(file_, sample_indices=None):
    ret = []
    if sample_indices:
        sample_indices.sort()
        with open(file_, 'r') as f:
            for i, line in enumerate(f):
                if i in sample_indices:
                    ret.append(line.rstrip())
        return ret
    else:
        with open(file_, 'r') as f:
            lines = f.readlines()
        return [line.rstrip() for line in lines]

def get_dataset_verbalizers(dataset: str): # parse: int-> label
    verbalizers = None
    if dataset == "sst-5":
        verbalizers = ["terrible", "bad", "okay", "good", "great"]
    elif dataset == "subj":
        verbalizers = ["subjective", "objective"]
    elif dataset == "trec":
        verbalizers = ["Description", "Entity", "Expression", "Human", "Location", "Number"]
    else:
        raise ValueError("dataset not supported")
    return verbalizers

def load_cls_data(verbalizers=None, data_path="data/cls/sst-5/dev.txt", sample_nums=100, seed=5):
    random.seed(seed)
    test_data = read_lines(data_path, sample_indices=None)
    dev_src = []
    dev_tgt = []
    for i, line in enumerate(test_data):
        try:
            cur_src, cur_tgt = line.split('\t')
        except:
            raise ValueError
        dev_src.append(cur_src)
        dev_tgt.append(verbalizers[int(cur_tgt)])
    sample_indices = random.sample(range(len(dev_src)), sample_nums)
    print("%"*20, "cls data sample", "%"*20)
    print("sample_indices:", sample_indices)
    dev_src = [dev_src[i] for i in sample_indices]
    dev_tgt =[dev_tgt[i] for i in sample_indices]
    return dev_src, dev_tgt

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_markup_prompt(FactorType, prompt):
    markup_prompt = []
    for factor, value in zip(FactorType, prompt):
        tmp = "<"+factor+">" + value + "</"+factor+">"
        markup_prompt.append(tmp)
    return markup_prompt
