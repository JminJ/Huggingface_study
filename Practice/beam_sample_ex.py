from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    BeamSearchScorer,
)

import torch
from transformers.generation_logits_process import LogitsWarper

tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')

encoder_input_str = "translate English to German: How old are you?"
encoder_input_ids = tokenizer(encoder_input_str, return_tensor = 'pt').input_ids

# lets run beam search using 3 beams
num_beams = 3
# define decoder start token ids
input_ids = torch.ones((num_beams, 1), device = model.device, dtype = torch.long)
input_ids = input_ids * model.config.decoder_start_token_id

# add encoder_output to model keyword arguments
model_kwargs = {
    'encoder_outputs': model.get_decoder()(encoder_input_ids.repeat_interleave(num_beams, dim = 0), return_dict = True)
}

# instantiate beam scorer
beam_scorer = BeamSearchScorer(
    batch_size = 1,
    max_length = model.config.max_length,
    num_beams = num_beams,
    device = model.device,
)

# instantiate beam processors
logits_processor = LogitsProcessorList([
    MinLengthLogitsProcessor(5, eos_token_id = model.config.eos_token_id)
])

logits_warper = LogitsProcessorList([
    TopKLogitsWarper(50),
    TemperatureLogitsWarper(0.7),
])

outputs = model.beam_sample(
    input_ids, beam_scorer, logits_processor = logits_processor, logits_warper = logits_warper, **model_kwargs
)

print('Generated : ', tokenizer.batch_decode(outputs, skip_special_tokens = True))