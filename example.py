from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

import whowhatbench

max_new_tokens = 128
model_small_id = "facebook/opt-125m"
model_small = AutoModelForCausalLM.from_pretrained(model_small_id)
tokenizer_small = AutoTokenizer.from_pretrained(model_small_id)

model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
test_data = ["Who is author of The Lord of the Rings?",
             "Who was the first king of Norway?",
             "Who is the most famous scientist in the field of artificial intelligence?",
             "What is the Bronze Age?",
             "What is the Gold Age?"
             ]
evaluator = whowhatbench.Evaluator(base_model=model, tokenizer=tokenizer_small, test_data=test_data)

all_metrics_per_question, all_metrics = evaluator.score(model_small)

print(all_metrics_per_question)
print(all_metrics)

metrics = ["similarity", "SDT norm"]

for metric in metrics:
    worst_examples = evaluator.worst_examples(top_k=5, metric=metric)
    print("Metric: ", metric)
    for e in worst_examples:
        print("\t=========================")
        print(f"\t{metric}: ", e[metric])
        print("\tPrompt: ", e["prompt"])
        print("\tSource Model:\n ", "\t" + e["source_model"])
        print("\tOptimized Model:\n ", "\t" + e["optimized_model"])
