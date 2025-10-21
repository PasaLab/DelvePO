from utils import get_markup_prompt

class MemoryUpdate:
    def __init__(self, FactorType, number_of_pairs, number_of_set):
        self.factor_type = FactorType
        self.number_of_pairs = number_of_pairs
        self.number_of_set = number_of_set

    def memory_pair_update(self, memory_pair, mutate_factors, cur_prompt, cur_metric, new_prompt, new_metric):
        tmp_prompt, tmp_new = [], []
        for mutate_factor in mutate_factors:
            idx = self.factor_type.index(mutate_factor)
            tmp_prompt.append(cur_prompt[idx])
            tmp_new.append(new_prompt[idx])
        markup_prompt, markup_new_prompt = get_markup_prompt(mutate_factors, tmp_prompt), get_markup_prompt(mutate_factors, tmp_new) 
        if [markup_prompt, markup_new_prompt] in memory_pair or [markup_new_prompt, markup_prompt] in memory_pair:
            return memory_pair
        while len(memory_pair) >= self.number_of_pairs:
            memory_pair.pop(0)
        if cur_metric > new_metric:
            memory_pair.append([markup_prompt, markup_new_prompt])
        else:
            memory_pair.append([markup_new_prompt, markup_prompt])
        return memory_pair
    
    def memory_set_update(self, memory_set, new_prompt, new_metric):
        left, right = 0, len(memory_set)
        while left < right:
            mid = (left + right) // 2
            if memory_set[mid][1] > new_metric:
                left = mid + 1
            else:
                right = mid
        memory_set.insert(left, [new_prompt, new_metric])
        if len(memory_set) > self.number_of_set:
            memory_set = memory_set[:self.number_of_set]
        return memory_set
    
    def factor_set_update(self, FactorSet, population):
        pop_size = len(population)
        top_half = population[:pop_size // 2]
        for pop in top_half:
            for i, value in enumerate(pop[0]):
                if value not in FactorSet[i]:
                    FactorSet[i].append(value)
        return FactorSet