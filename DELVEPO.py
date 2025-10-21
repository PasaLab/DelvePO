from memory_update import MemoryUpdate
from collections import defaultdict
from time import strftime, gmtime
from datetime import datetime
from llm_utils import *
from pattern import *
from utils import *
import datetime
import logging
import string
import random
import time
import os

class DELVEPO:
    def __init__(self, args, verbalizers=None):
        # Initialization, unchanged during iteration
        print("Initializing DELVEPO...")
        self.init_time = time.time()
        self.population_size = args.population_size
        self.factor_set = args.factor_set
        self.factor_type = args.factor_type
        self.epoch_size = args.epoch_size
        self.parents_size = args.parents_size
        self.data_path = args.data_path
        self.output_path = args.output_path
        self.number_of_pairs = args.number_of_pairs
        self.number_of_set = args.number_of_set
        self.task = args.task
        self.population = []
        self.prompt2metric = defaultdict(float)
        # Working memory
        self.memory_set = []
        self.memory_pair = []
        self.dataset = args.dataset
        self.llm_type = args.llm_type
        # print(f"self.factor_type: {self.factor_type}")
        # print(f"self.factor_set: {self.factor_set}")

        if self.task == "cls":
            self.dev_src, self.dev_tgt = load_cls_data(
                verbalizers = verbalizers, data_path = self.data_path, sample_nums = args.sample_num, seed = args.seed
            )
            self.prompt_template = PROMPT_For_cls
        self.memory_update = MemoryUpdate(self.factor_type, self.number_of_pairs, self.number_of_set)
        self.valid = True
        self.setup_log(os.path.join(self.output_path, f"{self.task}_{self.dataset}"), 'DELVEPO')
        self.logger.info("=" * 50)
        self.logger.info("\n\t" + "\n\t".join(f"{k} = {v}" for k, v in vars(args).items()))
        if verbalizers:
            self.logger.info(f"\tverbalizers={verbalizers}")
        self.logger.info("=" * 50)
        llm_init(llm_type=args.llm_type, logger=self.logger)
        self.logger.info("start testing LLM")
        result = llm_query("Hello, who are you?", llm_type=args.llm_type)
        self.logger.info(f"result is [{result}]")
        self.logger.info("end testing LLM")
        self.logger.info("DELVEPO initialized.")

    def setup_log(self, log_path, log_name="basic"):
        print(f"Setting up log for {log_name}, log_path = '{log_path}'")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"{log_path}_{timestamp}.log"
        self.logger = logging.getLogger(log_name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)
            file_handler = logging.FileHandler(log_filename)
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s File \"%(filename)s\", line %(lineno)d, in %(funcName)s] [%(levelname)s] : %(message)s",
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            stream_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            if self.logger.hasHandlers():
                self.logger.handlers.clear()
            self.logger.addHandler(stream_handler)
            self.logger.addHandler(file_handler)

    def random_construct_prompt(self):
        ''' 
        Randomly sample factor values from the FactorSet to populate the prompt template.
        Calculate the resulting prompt's performance metric.
        '''
        self.logger.info("Constructing a random prompt with metrics...")
        prompt = []
        for factorLine in self.factor_set:
            choice = random.choice(factorLine) if factorLine else ""
            prompt.append(choice)
        
        promptStr = '-'.join(prompt)
        if self.prompt2metric[promptStr] == 0.0:
            metric = generate_metric(self.dev_src, self.dev_tgt, prompt, self.factor_type, llm_type=self.llm_type, task=self.task, dataset=self.dataset, logger=self.logger)
            self.prompt2metric[promptStr] = metric
            self.logger.info(f"Random prompt constructed: {prompt}, metric: {metric}")
            return prompt, metric
        else:
            prompt = []
            for factorLine in self.factor_set:
                choice = random.choice(factorLine) if factorLine else ""
                prompt.append(choice)            
            promptStr = '-'.join(prompt)
            metric = generate_metric(self.dev_src, self.dev_tgt, prompt, self.factor_type, task=self.task, dataset=self.dataset, llm_type=self.llm_type, logger=self.logger)
            self.prompt2metric[promptStr] = metric
            self.logger.info(f"Random prompt constructed: {prompt}, metric: {metric}")
            return prompt, metric

    def get_mutate_direction_v1(self, prompt, memory_pair):
        markup_prompt = get_markup_prompt(self.factor_type, prompt) 
        input4mutate = MUTATE_PATTERN_KEY_v1.format(memory_pair=memory_pair, prompt=markup_prompt, FactorType=self.factor_type)
        response = llm_query(input4mutate, self.llm_type)
        mutate_factors = extract_factors(response, FactorType=self.factor_type) 
        return mutate_factors
    
    def get_mutate_direction_v2(self, prompt1, prompt2, memory_pair):
        markup_prompt1 = get_markup_prompt(self.factor_type, prompt1)
        markup_prompt2 = get_markup_prompt(self.factor_type, prompt2) 
        input4mutate = MUTATE_PATTERN_KEY_v2.format(memory_pair=memory_pair, prompt1=markup_prompt1, prompt2=markup_prompt2, FactorType=self.factor_type)
        response = llm_query(input4mutate, self.llm_type)
        mutate_factors = extract_factors(response, FactorType=self.factor_type) 
        return mutate_factors

    def mutate_discrete_prompt(self, prompt, mutate_factors, memory_set):
        new_prompt = prompt[:]
        
        mutate_values_old = []
        for mutate_factor in mutate_factors:
            idx = self.factor_type.index(mutate_factor)
            mutate_values_old.append(prompt[idx])
        markup_values_old = get_markup_prompt(mutate_factors, mutate_values_old)

        memory_set_discrete = []
        for item in memory_set:
            item_discrete = get_markup_prompt(self.factor_type, item[0]) 
            memory_set_discrete.append([item_discrete, item[1]])
        input4mutate = MUTATE_PATTERN_value_case1.format(memory_set_discrete=memory_set_discrete, old_values=markup_values_old)

        response = llm_query(input4mutate, self.llm_type)
        mutate_values_new = extract_crossover_values(response)
        if len(mutate_values_new) != len(mutate_factors):   # The number of mutated values is inconsistent with the number of mutated factors
            mutate_values_new = []

            # Fallback: Randomly select existing values.
            for mutate_factor in mutate_factors:
                try:
                    idx = self.factor_type.index(mutate_factor)
                    factorLine = self.factor_set[idx]
                    random_value = random.choice(factorLine) if factorLine else ""
                    mutate_values_new.append(random_value)
                except ValueError:
                    print(f"Warning: Index = {idx}, Factor '{mutate_factor}' not found in FactorType. Skipping...")
                    continue
                except IndexError:
                    print(f"Warning: Index {idx} out of range for FactorSet. Skipping...")
                    continue
        for i, mutate_factor in enumerate(mutate_factors):
            # Replace the value in old_prompt with the new value to generate new_prompt
            idx = self.factor_type.index(mutate_factor)
            new_prompt[idx] = mutate_values_new[i]

        promptStr = '-'.join(new_prompt)
        if self.prompt2metric[promptStr] == 0.0:
            metric = generate_metric(self.dev_src, self.dev_tgt, new_prompt, self.factor_type, task=self.task, dataset=self.dataset,  llm_type=self.llm_type, logger=self.logger)
            self.prompt2metric[promptStr] = metric
            return new_prompt, metric
        else:
            # Evaluated; discarding this prompt and regenerating a new random prompt
            new_prompt, metric = self.random_construct_prompt()
            promptStr = '-'.join(new_prompt)
            self.prompt2metric[promptStr] = metric
            return new_prompt, metric

    def mutate_continuous_prompt(self, prompt, mutate_factors, memory_set):
        markup_factors = "<res> " + " | ".join(mutate_factors) + " </res>"
        prompt_continue = self.build_prompt(self.prompt_template, prompt)

        # Convert original prompt to natural language
        memory_set_continuous = []
        for item in memory_set:
            item_continuous = self.build_prompt(self.prompt_template, item[0])
            memory_set_continuous.append([item_continuous, item[1]]) 
        input4mutate = MUTATE_PATTERN_value_case2.format(memory_set_continuous=memory_set_continuous, prompt=prompt_continue, mutate_factors=markup_factors)
        response = llm_query(input4mutate, self.llm_type)
        new_prompt = extract_values_continuous(response)
        if len(new_prompt) != len(self.factor_type):     # The number of mutated values is inconsistent with the number of factors
            # Fallback: Randomly select existing values.
            new_prompt = prompt[:]
            for mutate_factor in mutate_factors:
                try:
                    idx = self.factor_type.index(mutate_factor)
                    factorLine = self.factor_set[idx]
                    random_value = random.choice(factorLine) if factorLine else ""
                    new_prompt[idx] = random_value
                except ValueError:
                    print(f"Warning: Index = {idx}, Factor '{mutate_factor}' not found in FactorType. Skipping...")
                    continue
                except IndexError:
                    print(f"Warning: Index {idx} out of range for FactorSet. Skipping...")
                    continue

        promptStr = '-'.join(new_prompt)
        if self.prompt2metric[promptStr] == 0.0:
            metric = generate_metric(self.dev_src, self.dev_tgt, new_prompt, self.factor_type, task=self.task, dataset=self.dataset, llm_type=self.llm_type, logger=self.logger)
            self.prompt2metric[promptStr] = metric
            return new_prompt, metric
        else:
            new_prompt, metric = self.random_construct_prompt()
            promptStr = '-'.join(new_prompt)
            self.prompt2metric[promptStr] = metric
            return new_prompt, metric

    def get_crossover_direction(self, prompt1, prompt2, mutate_factors, memory_pair):
        crossover_values_pair = []
        for crossover_factor in self.factor_type:
            if crossover_factor not in mutate_factors: 
                idx = self.factor_type.index(crossover_factor)
                markup_value1 = "<"+crossover_factor+">" + prompt1[idx] + "</"+crossover_factor+">"
                markup_value2 = "<"+crossover_factor+">" + prompt2[idx] + "</"+crossover_factor+">"
                crossover_values_pair.append([markup_value1, markup_value2]) 

        input4crossover = CROSSOVER_PATTERN_case3.format(memory_pair=self.memory_pair, old_values=crossover_values_pair)
        response = llm_query(input4crossover, self.llm_type)
        crossover_values = extract_crossover_values(response)
        if len(crossover_values) != (len(self.factor_type)-len(mutate_factors)):
            crossover_values = []
            for crossover_factor in self.factor_type:
                if crossover_factor not in mutate_factors:
                    idx = self.factor_type.index(crossover_factor)
                    factorLine = self.factor_set[idx]
                    random_value = random.choice(factorLine) if factorLine else ""
                    crossover_values.append(random_value)
        return crossover_values

    ''' crossover discrete prompt: Post-workflow (consider values) '''
    def crossover_discrete_prompt(self, prompt1, prompt2, mutate_factors, crossover_values, memory_set):
        mutate_values_pair = []
        for mutate_factor in mutate_factors:
            idx = self.factor_type.index(mutate_factor)
            markup_value1 = "<"+mutate_factor+">" + prompt1[idx] + "</"+mutate_factor+">"
            markup_value2 = "<"+mutate_factor+">" + prompt2[idx] + "</"+mutate_factor+">"
            mutate_values_pair.append([markup_value1, markup_value2])
        
        memory_set_discrete = []
        for item in memory_set:
            item_discrete = get_markup_prompt(self.factor_type, item[0]) 
            memory_set_discrete.append([item_discrete, item[1]])
        input4crossover = MUTATE_PATTERN_value_crossover_case3.format(memory_set_discrete = memory_set_discrete, old_values=mutate_values_pair)

        response = llm_query(input4crossover, self.llm_type)
        mutate_values_new = extract_crossover_values(response)
        if len(mutate_values_new) != len(mutate_factors):
            # Fallback: Randomly select existing values.
            mutate_values_new = []
            for mutate_factor in mutate_factors:
                try:
                    idx = self.factor_type.index(mutate_factor)
                    factorLine = self.factor_set[idx]
                    random_value = random.choice(factorLine) if factorLine else ""
                    mutate_values_new.append(random_value)
                except ValueError:
                    print(f"Warning: Index = {idx}, Factor '{mutate_factor}' not found in FactorType. Skipping...")
                    continue
                except IndexError:
                    print(f"Warning: Index {idx} out of range for FactorSet. Skipping...")
                    continue
        new_prompt = ["None" for _ in range(len(prompt1))] 
        i = 0
        for factor in self.factor_type:
            if factor not in mutate_factors:
                idx = self.factor_type.index(factor)
                new_prompt[idx] = crossover_values[i]
                i += 1

        for i, mutate_factor in enumerate(mutate_factors):
            # Select one from prompt1 and prompt2 to simulate the crossover process
            idx = self.factor_type.index(mutate_factor)
            new_prompt[idx] = mutate_values_new[i]

        promptStr = '-'.join(new_prompt)
        if self.prompt2metric[promptStr] == 0.0:
            metric = generate_metric(self.dev_src, self.dev_tgt, new_prompt, self.factor_type, task=self.task, dataset=self.dataset, llm_type=self.llm_type, logger=self.logger)
            self.prompt2metric[promptStr] = metric
            return new_prompt, metric
        else:
            new_prompt, metric = self.random_construct_prompt()
            promptStr = '-'.join(new_prompt)
            self.prompt2metric[promptStr] = metric
            return new_prompt, metric
    
    '''  crossover continuous prompt: Post-workflow (whole prompt, consider context)  '''    
    def crossover_continuous_prompt(self, prompt1, prompt2, mutate_factors, memory_set):
        if len(prompt1) != len(prompt2):
            return -1

        assert len(self.factor_type) == len(self.factor_set), "FactorType and FactorSet should have the same length"
        
        memory_set_continuous = []
        for item in memory_set:
            item_continuous = self.build_prompt(self.prompt_template, item[0])
            memory_set_continuous.append([item_continuous, item[1]]) 
        prompt1_continue = self.build_prompt(self.prompt_template, prompt1)
        prompt2_continue = self.build_prompt(self.prompt_template, prompt2)
        input4crossover =  MUTATE_PATTERN_value_crossover_case4.format(memory_set_continuous=memory_set_continuous, prompt1=prompt1_continue, mutate_factors=mutate_factors, prompt3=prompt2_continue)

        response = llm_query(input4crossover, self.llm_type)
        new_prompt = extract_values_continuous(response) 
        if len(new_prompt) != len(self.factor_type):
            # Fallback: Randomly select existing values.
            new_prompt = []
            for value1, value2 in zip(prompt1, prompt2):
                random_value = random.choice([value1, value2])
                new_prompt.append(random_value)
            for mutate_factor in mutate_factors:
                try:
                    idx = self.factor_type.index(mutate_factor)
                    factorLine = self.factor_set[idx]
                    random_value = random.choice(factorLine) if factorLine else ""
                    new_prompt[idx] = random_value
                except ValueError:
                    print(f"Warning: Index = {idx}, Factor '{mutate_factor}' not found in FactorType. Skipping...")
                    continue
                except IndexError:
                    print(f"Warning: Index {idx} out of range for FactorSet. Skipping...")
                    continue

        promptStr = '-'.join(new_prompt)
        if self.prompt2metric[promptStr] == 0.0:
            metric = generate_metric(self.dev_src, self.dev_tgt, new_prompt, self.factor_type, task=self.task, dataset=self.dataset, llm_type=self.llm_type, logger=self.logger)
            self.prompt2metric[promptStr] = metric
            return new_prompt, metric
        else:
            new_prompt, metric = self.random_construct_prompt()
            promptStr = '-'.join(new_prompt)
            self.prompt2metric[promptStr] = metric
            return new_prompt, metric

    def write_epoch(self, epoch: int, step:int, population, mutate_factors=None, operation_type=None, epoch_time=0.0):
        if self.output_path is None:
            return
        ave_metric = 0.0
        output = f"{self.output_path}/epoch_{epoch}.txt"
        with open(output, "a") as f:
            f.write(f"Epoch: {epoch}, Steps: {step}\n")
            for idx, (prompt, metric) in enumerate(population):
                promptidx = '-'.join(prompt)
                f.write(f"({idx}) Prompt: {prompt}, Metric: {metric}\n\n")
            total_metric = sum(metric for _, metric in population)
            best_metric = max(metric for _, metric in population)
            f.write(f"operation_type: {operation_type}\n")
            f.write(f"mutate_factors: {mutate_factors}\n")
            f.write(f"Epoch time: {epoch_time:.4f} seconds\n")
            ave_metric = total_metric / len(population)
            f.write(f"Average metric: {ave_metric}\n")
            f.write(f"Best metric: {best_metric}\n")     
        if epoch == self.epoch_size - 1:
            with open(f"{self.output_path}/train_result.txt", "a", encoding="utf-8") as rf:
                rf.write(f"Final average metric: {ave_metric}\n")
                rf.write(f"Final best metric: {best_metric}\n")

    def build_prompt(self, prompt_template, item):
        """
        item: Component values that should be filled into prompt_template
        """
        factor_types = self.factor_type
        assert len(factor_types) == len(item), "The lengths of factor_types and item must be the same."
        mapping = dict(zip(factor_types, item))  # Map to a dictionary
        # Extract all placeholders from the template
        formatter = string.Formatter()
        template_fields = [fname for _, fname, _, _ in formatter.parse(prompt_template) if fname]
        # Check for missing or extra fields
        missing = [f for f in template_fields if f not in mapping]
        extra   = [f for f in mapping if f not in template_fields]
        if missing:
            raise ValueError(f"The template requires the fields {missing}, but factor_types does not provide them.")
        if extra:
            raise ValueError(f"factor_types provides extra fields {extra} that the template does not use.")
        # Replace and generate the prompt
        return "<prompt>" + prompt_template.format(**mapping) + "</prompt>"

    def run_self_evolution(self):
        for _ in range(self.population_size):  
            prompt, metric = self.random_construct_prompt()
            self.population.append([prompt, metric])
        result_output_path = f"{self.output_path}/train_result.txt"
        with open(f"{self.output_path}/init_pop.txt", "w", encoding="utf-8") as f, \
             open(result_output_path, "a", encoding="utf-8") as rf:
            total_metric = 0.0
            best_metric = 0.0
            for idx, (prompt, metric) in enumerate(self.population):
                promptidx = '-'.join(prompt)
                self.prompt2metric[promptidx] = metric
                total_metric += metric
                if metric > best_metric:
                    best_metric = metric
                f.write(f"({idx}) Prompt: {prompt}, Metric: {metric}\n\n")
            ave_metric = total_metric / len(self.population)
            f.write(f"Average metric: {ave_metric}\n")
            f.write(f"Best metric: {best_metric}\n")
            rf.write(f"Initial average metric: {ave_metric}\n")
            rf.write(f"Initial best metric: {best_metric}\n")
        self.memory_set = sorted(self.population, key=lambda x: x[1], reverse=True)
        choice2des = {"1": "mutate discrete prompt", "2": "mutate continuous prompt", "3": "crossover discrete prompt", "4": "crossover continuous prompt"}
        best_prompts, best_metrics = [], []
        start_time = time.time()
        total_time = 0.0
        for epoch in range(self.epoch_size):
            self.population = sorted(self.population, key=lambda x: x[1], reverse=True)
            best_unit = self.population[0]
            self.logger.info('################# Epoch {} started, Best metric: {} #################'.format(epoch, best_unit[1])) 
            self.logger.info("=======population===========")
            self.logger.info(self.population)
            self.logger.info("=======FactorSet============")
            self.logger.info(self.factor_set)
            if len(best_prompts) == 0 or best_unit[1] > best_metrics[-1]:
                best_prompts.append(best_unit[0])
                best_metrics.append(best_unit[1])
            child_pool, metric_pool = [], []
            step = 0
            epoch_time = 0.0
            # Mutate and crossover until the new generation reaches population_size.
            while step < self.population_size:
                cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                choiceId = random.choice([1, 2, 3, 4])
                self.logger.info('Epoch {} Step {} started | Choice: {} | Description: {}'.format(epoch, step, choiceId, choice2des[str(choiceId)]))
                self.logger.info("%" * 20 + "memory_pair" + "%" * 20)
                self.logger.info(self.memory_pair)
                self.logger.info("%" * 20 + "memory_set" + "%" * 20)
                self.logger.info(self.memory_set)
                self.logger.info("%" * 40)
                match choiceId:
                    case 1: #! mutate discrete prompt
                        prompt = roulette_wheel_selection(self.population)
                        self.logger.info("========== Current prompt ===========")
                        self.logger.info(prompt)
                        promptidx = '-'.join(prompt)
                        self.logger.info(self.prompt2metric[promptidx])

                        mutate_factors = self.get_mutate_direction_v1(prompt, self.memory_pair)
                        if len(mutate_factors) == 0:
                            mutate_factors = random.choice(self.factor_type) # Randomness, prefer to exploration
                        if isinstance(mutate_factors, str):
                            mutate_factors = [mutate_factors]
                        self.logger.info("========== mutate_factors ===========")
                        self.logger.info(mutate_factors)

                        new_prompt, new_metric = self.mutate_discrete_prompt(prompt, mutate_factors, self.memory_set)
                        child_pool.append(new_prompt)
                        metric_pool.append(new_metric)

                        self.memory_pair = self.memory_update.memory_pair_update(self.memory_pair, mutate_factors, prompt, self.prompt2metric[promptidx], new_prompt, new_metric)

                    case 2: #! mutate continuous prompt
                        prompt = roulette_wheel_selection(self.population)
                        self.logger.info("========== Current prompt ===========")
                        self.logger.info(prompt)
                        promptidx = '-'.join(prompt)
                        self.logger.info(self.prompt2metric[promptidx])

                        mutate_factors = self.get_mutate_direction_v1(prompt, self.memory_pair)
                        if len(mutate_factors) == 0:
                            mutate_factors = random.choice(self.factor_type) # Randomness, prefer to exploration
                        if isinstance(mutate_factors, str):
                            mutate_factors = [mutate_factors]
                        self.logger.info("========== mutate_factors ===========")
                        self.logger.info(mutate_factors)

                        new_prompt, new_metric = self.mutate_continuous_prompt(prompt, mutate_factors, self.memory_set)
                        child_pool.append(new_prompt)
                        metric_pool.append(new_metric)

                        self.memory_pair = self.memory_update.memory_pair_update(self.memory_pair, mutate_factors, prompt, self.prompt2metric[promptidx], new_prompt, new_metric)
                    
                    case 3: #! crossover discrete prompt（Value-based mutation）
                        prompt1 = roulette_wheel_selection(self.population)
                        promptidx1 = '-'.join(prompt1)
                        prompt2 = roulette_wheel_selection(self.population)
                        promptidx2 = '-'.join(prompt2)
                        while promptidx1 == promptidx2:
                            prompt2 = roulette_wheel_selection(self.population)
                            promptidx2 = '-'.join(prompt2)
                        self.logger.info("========== Current prompt ===========")
                        self.logger.info(prompt1)
                        self.logger.info(self.prompt2metric[promptidx1])
                        self.logger.info(prompt2)
                        self.logger.info(self.prompt2metric[promptidx2])

                        mutate_factors = self.get_mutate_direction_v2(prompt1, prompt2, self.memory_pair) 
                        if isinstance(mutate_factors, str):
                            mutate_factors = [mutate_factors]
                        self.logger.info("========== mutate_factors ===========")
                        self.logger.info(mutate_factors)
                        crossover_values = []
                        if len(mutate_factors) < len(self.factor_type):
                            crossover_values = self.get_crossover_direction(prompt1, prompt2, mutate_factors, self.memory_pair)
                        self.logger.info("========== crossover direction ===========")
                        self.logger.info(crossover_values)

                        if len(mutate_factors) == 0:
                            new_prompt = crossover_values
                            new_metric = generate_metric(self.dev_src, self.dev_tgt, new_prompt, self.factor_type, task=self.task, dataset=self.dataset, llm_type=self.llm_type, logger=self.logger)
                        else:
                            new_prompt, new_metric = self.crossover_discrete_prompt(prompt1, prompt2, mutate_factors, crossover_values, self.memory_set)
                        child_pool.append(new_prompt)
                        metric_pool.append(new_metric)

                        self.memory_pair = self.memory_update.memory_pair_update(self.memory_pair, mutate_factors, prompt1, self.prompt2metric[promptidx1], new_prompt, new_metric)
                        self.memory_pair = self.memory_update.memory_pair_update(self.memory_pair, mutate_factors, prompt2, self.prompt2metric[promptidx2], new_prompt, new_metric)
                        
                    case 4:  #! crossover continuous prompt（Whole prompt mutation: The given prompt is delimited by a template with multiple hypertext markers）
                        prompt1 = roulette_wheel_selection(self.population)
                        promptidx1 = '-'.join(prompt1)
                        prompt2 = roulette_wheel_selection(self.population)
                        promptidx2 = '-'.join(prompt2)
                        while promptidx1 == promptidx2:
                            prompt2 = roulette_wheel_selection(self.population)
                            promptidx2 = '-'.join(prompt2)
                        self.logger.info("========== Current prompt ===========")
                        self.logger.info(prompt1)
                        self.logger.info(self.prompt2metric[promptidx1])
                        self.logger.info(prompt2)
                        self.logger.info(self.prompt2metric[promptidx2])
                        
                        mutate_factors = self.get_mutate_direction_v2(prompt1, prompt2, self.memory_pair) 
                        if isinstance(mutate_factors, str):
                            mutate_factors = [mutate_factors]
                        self.logger.info("========== mutate_factors ===========")
                        self.logger.info(mutate_factors)

                        new_prompt, new_metric = self.crossover_continuous_prompt(prompt1, prompt2, mutate_factors, self.memory_set)
                        child_pool.append(new_prompt)
                        metric_pool.append(new_metric)

                        self.memory_pair = self.memory_update.memory_pair_update(self.memory_pair, mutate_factors, prompt1, self.prompt2metric[promptidx1], new_prompt, new_metric)
                        self.memory_pair = self.memory_update.memory_pair_update(self.memory_pair, mutate_factors, prompt2, self.prompt2metric[promptidx2], new_prompt, new_metric)

                self.memory_set = self.memory_update.memory_set_update(self.memory_set, new_prompt, new_metric)
                self.logger.info("===========new_prompt=============")
                self.logger.info(new_prompt)
                self.logger.info("===========new_metric=============")
                self.logger.info(new_metric)
                end_time = time.time()
                step_time = end_time - start_time - metric_time
                step_show_time = strftime("%H:%M:%S", gmtime(step_time))
                epoch_time += step_time
                self.logger.info('Epoch {} Step {} finished | cost time: {}'.format(epoch, step, step_show_time))
                step += 1
                start_time = time.time()
            
            #! Update module: Update population
            idxs = []
            for i in range(self.population_size):
                promptStr = '-'.join(self.population[i][0])
                idxs.append(promptStr)
            for i in range(self.population_size):
                child, metric = child_pool[i], metric_pool[i]
                childStr = '-'.join(child)
                if childStr not in idxs:
                    self.population.append([child, metric])
                    idxs.append(childStr)
            self.population = sorted(self.population, key=lambda x: x[1], reverse=True) # Descending order
            self.population = self.population[:self.population_size] # Keep the first half to update population
            self.factor_set = self.memory_update.factor_set_update(self.factor_set, self.population) # co-evolve
            total_time += epoch_time
            epoch_show_time = strftime("%H:%M:%S", gmtime(epoch_time))
            metrics = [row[1] for row in self.population]
            ave_score, best_score = sum(metrics) / len(metrics), max(metrics)
            epoch_output_path = f"{self.output_path}/epoch_{epoch}.txt"
            self.logger.info('################# Epoch {} finished, Ave score: {}, Best score: {}, cost time: {} #################'.format(epoch, ave_score, best_score, epoch_show_time))
            self.logger.info(f"Writing epoch {epoch} to {epoch_output_path}")
            try:
                self.write_epoch(
                    epoch=epoch,
                    step=step,
                    population=self.population,
                    mutate_factors=mutate_factors,
                    operation_type=choice2des[str(choiceId)],
                    epoch_time=epoch_time
                )
                self.logger.info(f"Epoch {epoch} data written successfully to {epoch_output_path}")
            except Exception as e:
                self.logger.error(f"Error writing epoch data to file. Exception occurred: {e}")
                pass
        
        ave_epoch_time = total_time / self.epoch_size
        evol_time = time.time() - self.init_time
        with open(result_output_path, "a", encoding="utf-8") as rf:
            rf.write(f"Average time for each epoch: {ave_epoch_time:.2f}s\n")
            rf.write(f"Total evolution time: {evol_time:.2f}s\n")
        self.logger.info(f"Average time for each epoch: {ave_epoch_time}s")
        return best_metrics, best_prompts