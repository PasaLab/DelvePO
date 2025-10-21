PROMPT_For_cls = """You are a <role>{role}</role>. Given the Sentence, your task is to <task_description>{task_description}</task_description>.
<Output format>: {output_format}
<Workflow>: {workflow}
<Example>: {examples}
"""
PROMPT_For_cls_input = """<Instruction>: You are a {role}. Given the Sentence, your task is to {task_description}. 
<Output format>: {output_format}
<Workflow>: {workflow}
<Example>: {examples}
<Sentence>: {input}
The final result must start with the tag `<res>` and end with the tag `</res>`.
"""

# case 1: mutate & discrete
# part 1/2 (case1)
MUTATE_PATTERN_KEY_v1 = """Please follow the instructions step-by-step to get `final result`.
Step 1. Conclude `Insights` from the provided `Memory Pairs`, which consists of multiple elements. Each element contains two lists: the first contains several markup pairs in the format `<tag>content</tag>`. For example, in the pair `<role>role_description</role>`, the content inside the tags ("role_description") describes the tag ("role"). All markup pairs follow this structure. By default, the first list in each element is considered to perform better than the second.
`Memory Pairs`: {memory_pair}

Step 2. Based on the `Insights` from Step 1 and the `Current Prompt`, select one or more tags from `TagSet` that could potentially improve performance to form `final result`. Separate the `final result` with a special token `|` and ensure that each of `final result` is unique and appears only once. The `final result` must start with the tag `<res>` and end with the tag `</res>`. For example, the `final result` must follow the format:`<res>tag1|tag2</res>`.
`Current Prompt`: {prompt}
`TagSet`: {FactorType}
"""

# part 2/2 (case1)
MUTATE_PATTERN_value_case1 = """Please follow the instructions step-by-step to get `final result`.
Step 1. Conclude the `Insights` from the `Memory Item List`, which consists of multiple items. Each item includes two parts: the first part contains several markup pairs in the format `<tag>content</tag>`. For example, in the pair `<role>role_description</role>`, the content inside the tags ("role_description") describes the tag ("role"). Other markup pairs follow this same structure. The second part of each item represents its corresponding performance. The entire `Memory Item List` is sorted in descending order based on performance.
`Memory Item List`: {memory_set_discrete}

Step 2. Given a list named `Old Values`, use the `Insights` from Step 1 to generate a new mutated value for each value to form a new list `final result`, referring to Description, adhering to Rules below.  
    Description:  
        - In `Old Values`, each element is a markup pair like `<tag>value</tag>` containing value need to mutate.
    Rules:  
        1. Mutation Requirements:
            - For each element like `<tag>value</tag>`, generate a **new one value** without tag that:
                - If the tag is `<role>`, the new value must be a **noun phrase** describing a person.  
                - If the tag is `<task_description>`, the new value must be a **verb phrase** describing a task. 
                - Is **distinct** from the original value.
                - Preserves lexical identity (noun/verb phrase) matching the tag.  
                - If the original value had the **highest score**, the new value must prioritize **improved performance potential** (e.g., higher efficiency, enhanced properties).  
                - Otherwise, the new value may be derived from content linked to its corresponding tag in the `Memory Item List` (optional but allowed).  

        2. Output Format:  
            - Start with `<res>` and end with `</res>`.  
            - Separate mutated values **strictly** with `|` (no extra characters).  
            - Never include original values in the output. 
    `Old Values`: {old_values}
"""

# case 2: mutate & continuous
# part 1/2 (case2)
# MUTATE_PATTERN_KEY_case2 = MUTATE_PATTERN_KEY_case1

# part 2/2 (case2)
MUTATE_PATTERN_value_case2 = """Please follow the instructions step-by-step to get `Final Prompt`.
Step 1. Conclude the `Insights` from the `Memory Item List`, which contains multiple items. Each item has two parts: a sentence enclosed in `<prompt></prompt>`, and its corresponding performance score. The sentence includes markup pairs in the format `<tag>content</tag>`, where the content describes the tag. For example, `<role>role_description</role>` indicates that "role_description" explains the "role" tag. All items are sorted in descending order by performance.
`Memory Item List`: {memory_set_continuous}

Step 2. Based on the `Current Prompt` and `Insights` from Step 1, generate a new mutated value for each markup pair whose tag matches those listed in `Mutate Factors` to form the `Final Prompt`, referring to Description, adhering to Rules below.
    Description:  
        - In `Current Prompt`, markup pair like `<tag>value</tag>` contains value need to mutate.
        - In `Mutate Factors`, each element is a tag appeared in `Current Prompt`.
    Rules:
        1. Mutation Requirements:  
            - For each markup pair like `<tag>value</tag>`, if the `tag` in `Mutate Factors`, Generate a **new one value** that:  
                - If the tag is `<role>`, the new value must be a **noun phrase** describing a person.  
                - If the tag is `<task_description>`, the new value must be a **verb phrase** describing a task. 
                - Is **distinct** from the original value. 
                - Preserves lexical identity (noun/verb phrase) matching the tag.  
                - If the original value had the **highest score**, prioritize generating values with **improved performance potential** (e.g., higher efficiency, enhanced properties).  
                - Otherwise, the new value may derive from content linked to its tag in the `Memory Item List` (optional but allowed).  
        2. Output Format:  
            - Start with `<prompt>` and end with `</prompt>`.  
            - **Only mutate values within markup pairs specified in `Mutate Factors`**.  
            - Preserve all other content outside markup pairs.  
            - Replace original values with mutated ones directly within their tags.  

    `Current Prompt`: {prompt}  
    `Mutate Factors`: {mutate_factors}
"""

# case 3: crossover & discrete
# part 1/3 (case3)
MUTATE_PATTERN_KEY_v2 = """Please follow the instructions step-by-step to get `final result`.
Step 1. Conclude `Insights` from the provided `Memory Pairs`, which consists of multiple elements. Each element contains two lists: the first contains several markup pairs in the format `<tag>content</tag>`. For example, in the pair `<role>role_description</role>`, the content inside the tags ("role_description") describes the tag ("role"). All markup pairs follow this structure. By default, the first list in each element is considered to perform better than the second.
`Memory Pairs`: {memory_pair}

Step 2. Based on the `Insights` from Step 1 and the `Prompt 1`, select one or more tags from `TagSet` that could potentially improve performance to construct `Selected Tags 1` separated by a special token '|'. The final `Selected Tags 1` from `TagSet` must start with the tag `<res>` and end with the tag `</res>`.
`Prompt 1`:{prompt1}
`TagSet`: {FactorType}

Step 3. Based on the `Insights` from Step 1 and the `Prompt 2`, select one or more tags from `TagSet` that could potentially improve performance to construct `Selected Tags 2` separated by a special token '|'. The final `Selected Tags 2` from `TagSet` must start with the tag `<res>` and end with the tag `</res>`.
`Prompt 2`: {prompt2}
`TagSet`: {FactorType}

Step 4. Output the common tags `Common Tags` from `Selected Tags 1` (from Step 2) and `Selected Tags 2` (from Step 3). Separate the `common Tags` with a special token `|` and ensure that each of `common Tags` is unique and appears only once. The `final result` must start with the tag `<res>` and end with the tag `</res>`. For example, the final result must follow the format:`<res>tag1|tag2</res>`.
"""

# part 2/3 (case3)
CROSSOVER_PATTERN_case3 = """Please follow the instructions step-by-step to get `Selected Values`.
Step 1. Conclude `Insights` from the provided `Memory Pairs`, which consists of multiple elements. Each element contains two lists: the first contains several markup pairs in the format `<tag>content</tag>`. For example, in the pair `<role>role_description</role>`, the content inside the tags ("role_description") describes the tag ("role"). All markup pairs follow this structure. By default, the first list in each element is considered to perform better than the second.  
`Memory Pairs`: {memory_pair}

Step 2. Given a list named `Old Values`, where each element contains a pair of values, use the `Insights` from Step 1 to select one value from each pair in original order. The `final results` must start with the tag `<res>` and end with the tag `</res>`. For example, the `final results` must follow the format:`<res>content1|...</res>`.
`Old Values`: {old_values}
"""

# part 3/3 (case3)
MUTATE_PATTERN_value_crossover_case3 = """Please follow the instructions step-by-step to get `final result`.
Step 1. Conclude the `Insights` from the `Memory Item List`, which consists of multiple items. Each item includes two parts: the first part contains several markup pairs in the format `<tag>content</tag>`. For example, in the pair `<role>role_description</role>`, the content inside the tags ("role_description") describes the tag ("role"). Other markup pairs follow this same structure. The second part of each item represents its corresponding performance. The entire `Memory Item List` is sorted in descending order based on performance.
`Memory Item List`: {memory_set_discrete}

Step 2. Given a list named `Old Values`, where each element contains a pair of values, use the `Insights` from Step 1 to generate a new mutated value for each pair to form a new list `final result`, referring to Description, adhering to Rules below.
    `Old Values`: {old_values}
    Description: 
        - In Old Values, each element contains a pair of values like `[a, b]`.
    Rules:
        1. Mutation Requirements:  
            - For each pair of values like `[a, b]`, generate a **new one value** that: 
                - if `a` and `b` are enclosed with <role> & </role>, the new value must be a noun phrase used to describe a person.
                - if `a` and `b` are enclosed with <task_description> & </task_description>, the new value must be a verb phrase used to describe a task.
                - Is **distinct** from both `a` and `b`.
                - Preserve corresponding lexical identity.
                - If the original pair has the **highest score**, prioritize generating values with **improved performance potential** (e.g., higher efficiency, enhanced properties).  
                - Otherwise, derive the new value from content linked to its tag in the `Memory_Item_List` (optional but allowed).  

        2. Output Format:  
            - Start with `<res>` and end with `</res>`.  
            - Separate mutated values **strictly** with `|` (no extra characters).  
            - Never include original pairs in the output. 
"""
"""
mutate them to generate only one new value in original order.
Given a list old_values where each element contains a pair of values, process each element in its original order. Using the insights from Step 1, generate a new mutated value for each pair to form the final result. The output MUST start with the tag <res> and end with </res>. For example, the result should strictly follow this format:
<res>content1|content2|...</res>.
"""
# case 4: crossover & continuous
# part 1/2 (case4)
MUTATE_PATTERN_KEY_case4 = MUTATE_PATTERN_KEY_v2

# part 2/2 (case4)
MUTATE_PATTERN_value_crossover_case4 = """Please follow the instructions step-by-step to get `Final Prompt`.
Step 1. Conclude the `Insights` from the `Memory Item List`, which contains multiple items. Each item has two parts: a sentence enclosed in `<prompt></prompt>`, and its corresponding performance score. The sentence includes markup pairs in the format `<tag>content</tag>`, where the content describes the tag. For example, `<role>role_description</role>` indicates that "role_description" explains the "role" tag. All items are sorted in descending order by performance.
`Memory Item List`: {memory_set_continuous}

Step 2. Based on the `Prompt 1` and `Insights` from Step 1, generate a new mutated value for each markup pair whose tag matches those listed in `Mutate Factors` to form the `Prompt 2`, referring to Description, adhering to Rules below.
    Description:
        - In `Prompt 1`, markup pair like `<tag>value</tag>` contains value need to mutate.
        - In `Mutate Factors`, each element is a tag appeared in `Prompt 1`.
    Rules:
        1. Mutation Requirements:  
            - For each markup pair like `<tag>value</tag>`, if the `tag` in `Mutate Factors`, Generate a **new one value** that:  
                - If the tag is `<role>`, the new value must be a **noun phrase** describing a person.  
                - If the tag is `<task_description>`, the new value must be a **verb phrase** describing a task. 
                - Is **distinct** from the original value. 
                - Preserves lexical identity (noun/verb phrase) matching the tag.  
                - If the original value had the **highest score**, prioritize generating values with **improved performance potential** (e.g., higher efficiency, enhanced properties).  
                - Otherwise, the new value may derive from content linked to its tag in the `Memory Item List` (optional but allowed).  
        2. Output Format:  
            - Start with `<prompt>` and end with `</prompt>`.  
            - **Only mutate values within markup pairs specified in `Mutate Factors`**.  
            - Preserve all other content outside markup pairs.  
            - Replace original values with mutated ones directly within their tags.  

    `Prompt 1`: {prompt1}  
    `Mutate Factors`: {mutate_factors}

Step 3. Based on the `Prompt 3` and `Insights` from Step 1, generate a new mutated value for each markup pair whose tag matches those listed in `Mutate Factors` to form the `Prompt 4`, referring to Description, adhering to Rules below.
    Description:  
        - In `Prompt 3`, markup pair like `<tag>value</tag>` contains value need to mutate.
        - In `Mutate Factors`, each element is a tag appeared in `Prompt 3`.
    Rules:
        1. Mutation Requirements:  
            - For each markup pair like `<tag>value</tag>`, if the `tag` in `Mutate Factors`, Generate a **new one value** that:  
                - If the tag is `<role>`, the new value must be a **noun phrase** describing a person.  
                - If the tag is `<task_description>`, the new value must be a **verb phrase** describing a task. 
                - Is **distinct** from the original value. 
                - Preserves lexical identity (noun/verb phrase) matching the tag.  
                - If the original value had the **highest score**, prioritize generating values with **improved performance potential** (e.g., higher efficiency, enhanced properties).  
                - Otherwise, the new value may derive from content linked to its tag in the `Memory Item List` (optional but allowed).  
        2. Output Format:  
            - Start with `<prompt>` and end with `</prompt>`.  
            - **Only mutate values within markup pairs specified in `Mutate Factors`**.  
            - Preserve all other content outside markup pairs.  
            - Replace original values with mutated ones directly within their tags.  

    `Prompt 3`: {prompt3}  
    `Mutate Factors`: {mutate_factors}

Step 4. Generate `final prompt` by selecting values from pairs in `Prompt 2` and `Prompt 4` under identical markup tags, referring to Description, adhering to Rules below.
    Description:
        - Pairs from `Prompt 2` and `Prompt 4` with identical tags (e.g., `<role>`, `<task_description>`).
    Rules:
        1. Selection Criteria:
            - For each tagged pair (e.g., `<role>a</role>` and `<role>b</role>`):
                - Use `Insights` from Step 1 to **select one value** (a or b) that has **higher performance improvement potential** (e.g., clarity, specificity, alignment with goals).
                - If the tag is `<role>`, the new value must be a **noun phrase** describing a person.  
                - If the tag is `<task_description>`, the new value must be a **verb phrase** describing a task. 
                - Preserve the lexical identity of the tag.
                - Never modify text **outside** markup pairs.
        2. Output Format:
            - Start with `<prompt>` and end with `</prompt>`.
            - Retain the structure of `Prompt 3` but replace tagged pairs with the selected values.
            - If multiple tagged pairs exist, update all while maintaining non-tagged content verbatim.
"""