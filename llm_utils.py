from openai import OpenAI
from tqdm import tqdm
from pattern import *
from utils import *
import string
import time
import re

metric_time = 0.0

def llm_init(auth_file="auth.yaml", llm_type='deepseek-r1-8b', setting="default", logger=None):
    auth = read_yaml_file(auth_file)[llm_type][setting]
    try:
        global local_client
        base_url = auth["api_base"]
        api_key = auth["api_key"]
        if logger:
            logger.info(f"llm_init: llm_type = '{llm_type}', base_url = '{base_url}' api_key = '{api_key}'")
        # local_client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0)
        local_client = OpenAI(base_url=base_url, api_key=api_key)
        print("[llm_init] Success")
    except Exception as e:
        print("[llm_init] Error:", e)
        raise
    return True

def test_llm():
    local_client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY",)
    prompt = """
    tell me a short joke.
    """
    prompt = "give me 3 advice to lose weight."
    llm_model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    temperature = 0.5
    chat_completion = local_client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ],
        model = llm_model,
        temperature=temperature,
    )
    result = chat_completion.choices[0].message.content
    print(result)

def extract_seconds(text, retried=5):
    words = text.split()
    for i, word in enumerate(words):
        if "second" in word:
            return int(words[i - 1])
    return 60

def llm_query(prompt, llm_type="deepseek-r1-8b", temperature=0.5):
    retried = 0
    if llm_type == "deepseek-r1-8b":
        llm_model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    else:
        print("unsupported LLM")
    while True:
        try:
            chat_completion = local_client.chat.completions.create(
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    }
                ],
                model = llm_model,
                temperature=temperature,
                max_tokens=512,
            )
            result = chat_completion.choices[0].message.content
            break
        except Exception as e:
            error = str(e)
            print("[llm_query] retrying...", error)
            retried = retried + 1
    return result

def extract_result(response: str):
    """
        Args:
        response (str): "..... Let's finalize the promising factors based on both insights and the Current Prompt:
                        <res> role | task_description </res>"
        split sign: <res> </res> "|"  
    """
    end_think_pos = response.find('</think>')
    if end_think_pos != -1:
        remaining_res = response[end_think_pos + len('</think>'):]
    else:
        remaining_res = response  

    pattern = r'<res>(.*?)</res>' 

    matches = re.findall(pattern, remaining_res, re.DOTALL)  
    cleaned_matches = ""
    if matches:
        cleaned_matches = matches[-1].strip()
    else:
        cleaned_matches = remaining_res.strip() 
    return cleaned_matches

def extract_crossover_values(response: str):
    """
    Args:
        response (str): "..... Let's finalize the promising factors based on both insights and the Current Prompt:
                        <res> content1 | content2 </res>"
        split sign: <res> </res> "|"  
    """
    end_think_pos = response.find('</think>')
    if end_think_pos != -1:
        remaining_res = response[end_think_pos + len('</think>'):]
    else:
        remaining_res = response

    pattern = r'.*<res>(.*?)</res>'
    match = re.search(pattern, remaining_res, re.DOTALL)
    if match:
        content = match.group(1).strip()
    else:
        content = ""
    cleaned_matches = content.split("|")
    res = [s.strip() for s in cleaned_matches]
    tmp = res[:]
    for r in tmp:
        if len(r) == 0:
            res.remove(r)
    return res

def extract_factors(response: str, FactorType: list):
    """
    Extract components that should be mutated from the LLM's output
    Args:
        response (str): "..... Let's finalize the promising factors based on both insights and the Current Prompt:
                        <res> role | task_description </res>"
        split sign: <res> </res> "|"  
    """
    end_think_pos = response.find('</think>')
    if end_think_pos != -1:
        remaining_res = response[end_think_pos + len('</think>'):]
    else:
        remaining_res = response

    pattern = r'.*<res>(.*?)</res>'
    match = re.search(pattern, remaining_res, re.DOTALL)
    if match:
        content = match.group(1).strip()
    else:
        content = ""
    res = content.split("|")

    tmp = res[:]
    for r in tmp:
        if r not in FactorType:
            res.remove(r)
    tmp = res[:]
    res = []
    for r in tmp:
        if r not in res:
            res.append(r)
    return res

def extract_values_discrete(response: str):
    """
        Args:
        response (str): "..... 
                       Mutated Values: <res> <role>role1</role>, <task_description>task_description1</task_description></res>"
        split sign: <res></res>, <role></role>, <task_description></task_description>
    """
    end_think_pos = response.find('</think>')
    if end_think_pos != -1:
        remaining_res = response[end_think_pos + len('</think>'):]
    else:
        remaining_res = response
    pattern_res = r'<res>(.*?)</res>'
    matches = re.findall(pattern_res, remaining_res, re.DOTALL)

    pattern_values = r'<([a-zA-Z_]+)>(.*?)</\1>'
    matches = re.findall(pattern_values, matches[-1].strip(), re.DOTALL)

    res = [s[-1].strip() for s in matches]
    return res

def extract_values_continuous(response: str):
    """    
    Args:
        response (str): ".....Let's finalize the promising factors based on both insights and the Current Prompt:
                       Final Prompt:<prompt>You are a <role>role1</role>. Your task is to <task_description>task_description1</task_description>. To accomplish this, you need to <action>action1</action>. Please make sure to <requirements>requirements1</requirements> throughout the process.</prompt>"
        split sign: <prompt> </prompt>, <role></role>, <td> </td>
    """
    end_think_pos = response.find('</think>')
    if end_think_pos != -1:
        remaining_res = response[end_think_pos + len('</think>'):]
    else:
        remaining_res = response
    
    pattern_prompt = r'<prompt>(.*?)</prompt>'
    matches = re.findall(pattern_prompt, remaining_res, re.DOTALL)
    if len(matches) == 0:
        return []
    pattern_factor = r'<([a-zA-Z_]+)>(.*?)</\1>'
    matches = re.findall(pattern_factor, matches[-1].strip(), re.DOTALL)

    res = [s[-1].strip() for s in matches]
    return res

dataset_classes_list = {
    'sst-5': ['terrible', 'bad', 'okay', 'good', 'great'],
    'trec': ["Description", "Entity", "Expression", "Human", "Location", "Number"],
    'subj': ["subjective", "objective"],
}

def first_appear_pred(text, verbalizer_dict):
    text = text.lower()
    verbalizer_dict = [k.lower() for k in verbalizer_dict]
    for word in text.split():
        word = word.strip("`.,!?\"'()[]{}<>:;*/\\|")
        if word in verbalizer_dict:
            return word
    return ""

def generate_metric(dev_src, dev_tgt, prompt, factor_types, logger, llm_type="deepseek-r1-8b", task="cls", dataset="sst-5"):
    start_time2 = time.time()
    logger.info(f"Generating metric, task={task}, dataset={dataset}, prompt=[[{prompt}]]")
    global metric_time
    if task == "cls":
        hypos = []
        i = 0
        for dev_txt in tqdm(dev_src):
            data_with_prompt = build_prompt(factor_types + ["input"], PROMPT_For_cls_input, prompt + [dev_txt])
            result = llm_query(data_with_prompt, llm_type)
            final_result = extract_result(result)
            final_result = first_appear_pred(final_result, dataset_classes_list[dataset])
            hypos.append(final_result)
            i += 1
        score = cal_cls_score(hypos, dev_tgt, metric="acc")
    else:
        raise ValueError(f"Task {task} not supported in generate_metric()")
    end_time2 = time.time()
    metric_time = metric_time + (end_time2 - start_time2)
    logger.info(f"Metric generated, score = {score}, metric_time: {metric_time}s")
    return score

def build_prompt(factor_types, template, item):
    # print(f"factor_types: {factor_types}")
    # print(f"item: {item}")
    assert len(factor_types) == len(item), "The lengths of factor_types and item must be equal."
    mapping = dict(zip(factor_types, item))  # Map factor types to their corresponding values.
    formatter = string.Formatter()
    template_fields = [fname for _, fname, _, _ in formatter.parse(template) if fname]
    missing = [f for f in template_fields if f not in mapping]
    extra   = [f for f in mapping if f not in template_fields]
    if missing:
        raise ValueError(f"The template requires fields {missing}, but factor_types does not provide them.")
    if extra:
        raise ValueError(f"factor_types provides extra fields {extra} that are not used in the template.")
    return "<prompt>" + template.format(**mapping) + "</prompt>"