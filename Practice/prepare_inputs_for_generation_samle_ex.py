from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
)

tokenizer = AutoTokenizer.from_pretrained('gpt-2')
model = AutoModelForCausalLM.from_pretrained('gpt-2')

# set pad_token_id to eos_token_id because GPT2 does not have a EOS token
model.config.pad_token_id = model.config.eos_token_id

input_prompt = 'Today is a beautiful day, and'
input_ids = tokenizer(input_prompt, return_tensors = 'pt').input_ids

# instantiate logits processors
logits_processor = LogitsProcessorList([
    MinLengthLogitsProcessor(15, eos_token_id = model.config.eos_token_id),
])

logits_warper = LogitsProcessorList([
    TopKLogitsWarper(50),
    TemperatureLogitsWarper(0.7),
])

outputs = model.sample(input_ids, logits_processor = logits_processor, logits_warper = logits_warper)
print('Generated:', tokenizer.batch_decode(outputs, skip_special_tokens = True))
