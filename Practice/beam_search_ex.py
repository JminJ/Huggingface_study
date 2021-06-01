from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    BeamSearchScorer,
)
import torch

tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')

encoder_input_str = 'translate English to German: How old are you?'
encoder_input_ids = tokenizer(encoder_input_str, return_tensors = 'pt').input_ids

# lets run beam search using 3 beams
num_beams = 3
# define decoder start token ids
input_ids = torch.ones((num_beams, 1), device = model.device, dtype = torch.long)
input_ids = input_ids * model.config.decoder_start_token_id

# add encoder_outputs to model keyword argyments
model_kwargs = {
    'encoder_outputs': model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim = 0))
}

# instantiate beam scorer
beam_scorer = BeamSearchScorer(
    batch_size = 1,
    num_beams = num_beams,
    device = model.device,
)

# instantiate logit processors
logits_processor = LogitsProcessorList([
    MinLengthLogitsProcessor(5, eos_token_id = model.config.eos_token_id),
])

outputs = model.beam_search(input_ids, beam_scorer, logits_processor = logits_processor, **model_kwargs)
print('Generated:', tokenizer.batch_decode(outputs, skip_special_tokens = True))

