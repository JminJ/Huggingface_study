from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    HammingDiversityLogitsProcessor,
    BeamSearchScorer,
)

import torch

tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')

encoder_input_str = 'translate Engilsh to German: How old are you?'
encoder_input_ids = tokenizer(encoder_input_str, return_tensors = 'pt').input_ids

# lets run dicerse beam search using 6 beams
num_beams = 6
# define decoder start token ids
input_ids = torch.ones((num_beams, 1), device = model.device, dtype = torch.long)
input_ids = input_ids * model.config.decoder_start_token_id

# add encoder_outputs to model keyword arguments
model_kwargs = {
    'encoder_outputs':model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim = 0), return_dict = True)
}

# instantiate beam scorer
beam_scorer = BeamSearchScorer(
    batch_size = 1,
    max_length = model.config.max_length,
    num_beams = num_beams,
    device = model.device,
    num_beam_group = 3
)

# instantiate logits processors
logits_processor = LogitsProcessorList([
    HammingDiversityLogitsProcessor(5.5, num_beams = 6, num_beam_group = 3),
    MinLengthLogitsProcessor(5, eos_token_id = model.config.eos_token_id),
])

