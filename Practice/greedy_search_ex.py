from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
)

tokenizer = AutoTokenizer.from_pretrained('gpt-2')
model = AutoModelForCausalLM.from_pretrained('gpt-2')

# set pad_token_id to eos_token_id because GPT-2 does not have a EOS token
model.config.pad_token_id = model.config.eos_token_id

input_prompt = 'Today is beautiful day, and'
input_ids = tokenizer(input_prompt, return_tensor = 'pt').input_ids

# instantiate logits processors
logit_processor = LogitsProcessorList([
    MinLengthLogitsProcessor(15, eos_token = model.config.eos_token_ids)
])

outputs = model.greedy_search(input_ids, logits_processor = logit_processor)
print('Generated:', tokenizer.batch_decode(outputs, skip_special_tokens = True))
