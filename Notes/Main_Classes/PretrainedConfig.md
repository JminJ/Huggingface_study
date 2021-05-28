# transformers.PretrainedConfig
------
## 형태 및 설명
* transformers.PretrainedConfig(***kwargs*)
* 모든 configulation class들의 기반이 되는 class.
* 모든 model에 들어가는 일반적인 parameter들과 configulation을 loadding /downloading/saving하는 방법들을 다룬다.

> Note
>> configulation file은 disk에 저장되거나 불러와질 수 있다. model을 초기화하기 위해 이 file을 불러오고 사용할 때 **model의 weight는 불러와지지 않는다**.

**Class attributes (파생된 class들에 의해 override된다)**
* model_type(*str*) - 모델 유형에 대한 identifier, JSON file로 직렬화되고 AutoConfig에서 올바른 개체를 다시 만드는데 사용된다.
* is_composition(*bool*) - config class가 multiple sub-config들로 구성되어있는지를 정한다. 이 경우, config는 두개 이상의 PretrainedConfug type의 config들로 초기화 된다. like: **EncoderDecoderConfig** 또는 **RagConfig**.
* keys_to_ignore_at_inference(*List[str]*) - 추론을 하는 동안 model outputs dictionary를 볼 때 기본적으로 무시할 key들의 list.

**Common attribures (모든 subclass들에 존재한다.)**
* vocab_size(*int*) - vocabulary안에 있는 token들의 수, 이는 또한 embedding matrix의 첫 번째 dimension이다(model은 VIT같은 text modality를 가지고 있지 않기에 이 attribute는 찾을 수 없을 것이다).
* hidden_size(*int*) - model의 hidden size.
* num_attention_heads(*int*) - model의 multi-head-attention layer들에서 사용되는 attention head들의 수.
* num_hidden_layers(*int*) - model안의 block들의 수.

## Parameters
* name_or_path(str, *optional*, defaults to "") - configuration이 이러한 방법으로 생성되었을 경우, from_pretraind() 또는 from_pretrained()에 전달 될 문자열을 pretrained_model_name_or_path로 저장한다.
* output_hidden_states(bool, *optional*, defaults to False) - model이 모든 hidden-state를 return할지를 설정한다.
* output_attentions(bool, *optional*, defaults to False) - model이 모든 attention들을 return할지를 설정한다.
* return_dict(bool, *optional*, defaults to True) - model이 **ModelOutput**을 plain tuple 대신 return할지를 설정한다.
* is_encoder_decoder(bool, *optional*, defaults to False) - model이 encoder/decoder를 사용하는지 하지 않는지를 설정한다.
* is_decoder(bool, *optional*, defaults to False) - model이 decoder로 사용돠는지 여부(이 경우 encoder로 사용된다).
* add_cross_attention(bool, *optional*, defaults to False) - cross-attention layer들이 model에 추가되어야 하는지 여부. Note, 이 옵션은 오직 AUTO_MODELS_FOR_CASUAL_LM 안의 모든 model들로 이루어진 *:class:~transformers.EncoderDecoderModel class* 사이의 decoder model들로 사용될 수 있는 model들에 관련되어 있다.
* tie_encoder_decoder(bool, *optional*, defaults to False) - 모든 encoder weight들이 그들의 동일한 decoder weight들에 연결해야 하는지 여부. 이는 encoder와 decoder에 완벽히 동일한 parameter 이름들을 요구한다.
* prune_heads(Dict[int, Lists[int]], *optional*, defaults to {}) - model의 Pruned head들. key들은 선택한 layer index 및 관련 값, 해당 layer에서 prune할 head 목록이다.
* chunk_size_feed_forward(int, *optional*, defaults to 0) - resudial attention block 안의 feed forward layer의 chunk size. chunk siae가 0일 때의 의미는 feed forward layer가 chunk를 수행하지 않는다는 의미다. 
**sequence 생성을 위한 parameter들**
* max_length(int, *optional*, defaults to 20) - model의 generate method에서 기본으로 사용될 최대 길이
* min_length(int, *optional*, defaults to 10) - model의 generate method에서 기본으로 사용될 최소 길이
* do_sample(bool, *optional*, defaults to False) - model의 generate method에서 기본적으로 사용될 Flag. sampling을 사용할지 말지를 결정 ; 다른 경우는 greedy decoding을 사용한다.
* early_stopping(bool, *optional*, defaults to False) - model의 generate method에서 기본적으로 사용될 Flag. beam search를 배치 당 최소 num_beams 문장들이 완료되면 beam search를 중지할지 여부.
* num_beams(int, *optional*, defaults to 1) - model의 generate method에서 기본적으로 사용되는 bean search를 위한 beam들의 수. 1은 beam search를 사용하지 않는 다는 뜻과 같다.
* num_beam_groups(int, *optional*, defaults to 1) - 


