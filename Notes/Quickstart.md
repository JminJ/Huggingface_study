## Main concepts
이 library는 각 model을 위한 class들은 세 가지 type으로 구성되어 있다.
* model classes : 예를 들어 BertModel은 해당 library에서 제공하는 선 학습 된 weight들과 20가지 이상의 pytorch model들로 구성되어 있다.
* configuration classes : 해당 class는 model을 구성하기 위한 모든 parameter들을 적재하고 있다.
* tokenizer classes : 각 model을 위한 vocabularry를 적재하고 있고 model에 들어갈 token embeddings index들의 list 안의 문자열들을 encoding/decoding하는 방법들을 제공한다.

모든 classes들은 두 가지 방법들을 사용함으로써 선 학습된 instance들로부터 가져오거나 저장을 할 수 있다.
* from_pretrained() : library에서 제공하거나 user의 local에서 적재하는 pretrained version으로 부터 model/configuration/tokenizer을 인스턴스화할 수 있다.
* save_pretrained() : model/configuration/tokenizer을 저장한다. from_pretrained()를 통해 불러올 수 있다.

## Quick tour: Usage
Bert와 GPT2 class들을 사용하는 예제를 살펴본다.

각 model에 대한 더 많은 예제는 API reference통해 확인하는 것을 추천한다.

### BERT example
BertTokenizer를 사용해 tokenize된 input을 준비한다.
```python
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
# OPTIONAL
import logging
logging.basicConfig(level = logging.INFO)

# pre-train된 model tokenizer를 가져온다 (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased)

# Tokenize input
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)

# 이후 'BertForMaskedLM'을 통해 예측을 수행할 Mask token
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['[CLS]','who','was','jim','henson','?','[SEP]','jim','[MASK]','was','a','pupper','##eer','[SEP]']

# token들을 vocabulary index들로 변환한다.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# 첫 번째 문장와 두 번째 문장에 따라 index를 설정한다.
segments_ids = [0,0,0,0,0,0,0,1,1,1,1,1,1,1]

# input들을 Pytorch tensor로 변환한다.
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
```
이번에는 BertModel으로 hidden_state안의 우리 input들을 encode하는 방법을 살펴보자.
```python
# pre-train model을 불러온다 (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# DropOut module들을 비활성화 하기 위해 model을 evaluation mode로 만든다.
# 이는 evaluation 동안 재현 가능한 결과를 가지는데 매우 중요하다.
model.eval()

# 만약 GPU를 가지고 있다면, 모든것을 cuda에 넣어라.
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

# 각 layer에서 hidden state를 예측한다.
with torch.no_grad():
    # input들의 더 자세한 설명을 원한다면 model들의 docstring을 보는 것을 추천한다.
    outputs = model(tokens_tensor, token_type_ids = segments_tensors)
    # Transformes model들은 언제나 tuple을 output으로 내놓는다.
    # 우리의 경우, 첫 element는 Bert model의 마지막 layer의 hidden state이다.
    encoded_layers = outputs[0]

# 우리는 input sequence를 FloatTensor(batch_size, sequence length, model hidden dimension)으로 encode해 왔다.
assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), model.config.hidden_size)
```
그리고 어떻게 BertForMaskedLM을 사용해 masking될 token을 예측하는 방법을 알아볼 것이다.
```python
# pre-train된 model을 불러온다 (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# 만약 GPU를 가지고 있다면, 모든 것을 cuda에 넣는다
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

# 모든 token들을 예측한다.
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids = segments_tensors)
    predictions = outputs[0]

# 'henson'을 예측할 수 있었는지 확인한다.
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == 'henson'
```
### OpenAI GPT-2 example


