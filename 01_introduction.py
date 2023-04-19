# %% [markdown]
# <table align="left"><tr><td>
# <a href="https://colab.research.google.com/github/rickiepark/nlp-with-transformers/blob/main/01_introduction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="코랩에서 실행하기"/></a>
# </td></tr></table>

# %%
# 코랩을 사용하지 않으면 이 셀의 코드를 주석 처리하세요.
# !git clone https://github.com/rickiepark/nlp-with-transformers.git
# %cd nlp-with-transformers
# from install import *
# install_requirements(chapter=1)

# %% [markdown]
# # 트랜스포머스 소개

# %% [markdown]
# <img alt="transformer-timeline" caption="The transformers timeline" src="https://github.com/rickiepark/nlp-with-transformers/blob/main/images/chapter01_timeline.png?raw=1" id="transformer-timeline"/>

# %% [markdown]
# ## 인코더-디코더 프레임워크

# %% [markdown]
# <img alt="rnn" caption="Unrolling an RNN in time." src="https://github.com/rickiepark/nlp-with-transformers/blob/main/images/chapter01_rnn.png?raw=1" id="rnn"/>

# %% [markdown]
# <img alt="enc-dec" caption="Encoder-decoder architecture with a pair of RNNs. In general, there are many more recurrent layers than those shown." src="https://github.com/rickiepark/nlp-with-transformers/blob/main/images/chapter01_enc-dec.png?raw=1" id="enc-dec"/>

# %% [markdown]
# ## 어텐션 메커니즘

# %% [markdown]
# <img alt="enc-dec-attn" caption="Encoder-decoder architecture with an attention mechanism for a pair of RNNs." src="https://github.com/rickiepark/nlp-with-transformers/blob/main/images/chapter01_enc-dec-attn.png?raw=1" id="enc-dec-attn"/> 

# %% [markdown]
# <img alt="attention-alignment" width="500" caption="RNN encoder-decoder alignment of words in English and the generated translation in French (courtesy of Dzmitry Bahdanau)." src="https://github.com/rickiepark/nlp-with-transformers/blob/main/images/chapter02_attention-alignment.png?raw=1" id="attention-alignment"/> 

# %% [markdown]
# <img alt="transformer-self-attn" caption="Encoder-decoder architecture of the original Transformer." src="https://github.com/rickiepark/nlp-with-transformers/blob/main/images/chapter01_self-attention.png?raw=1" id="transformer-self-attn"/> 

# %% [markdown]
# ## NLP의 전이 학습

# %% [markdown]
# <img alt="transfer-learning" caption="Comparison of traditional supervised learning (left) and transfer learning (right)." src="https://github.com/rickiepark/nlp-with-transformers/blob/main/images/chapter01_transfer-learning.png?raw=1" id="transfer-learning"/>  

# %% [markdown]
# <img alt="ulmfit" width="500" caption="The ULMFiT process (courtesy of Jeremy Howard)." src="https://github.com/rickiepark/nlp-with-transformers/blob/main/images/chapter01_ulmfit.png?raw=1" id="ulmfit"/>

# %% [markdown]
# ## 허깅 페이스 트랜스포머스

# %% [markdown]
# ## 트랜스포머 애플리케이션 둘러보기

# %%
text = """Dear Amazon, last week I ordered an Optimus Prime action figure \
from your online store in Germany. Unfortunately, when I opened the package, \
I discovered to my horror that I had been sent an action figure of Megatron \
instead! As a lifelong enemy of the Decepticons, I hope you can understand my \
dilemma. To resolve the issue, I demand an exchange of Megatron for the \
Optimus Prime figure I ordered. Enclosed are copies of my records concerning \
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

# %% [markdown]
# ### 텍스트 분류

# %%
from transformers import pipeline

classifier = pipeline("text-classification")

# %%
import pandas as pd

outputs = classifier(text)
pd.DataFrame(outputs)    

# %% [markdown]
# ### 개체명 인식

# %%
ner_tagger = pipeline("ner", aggregation_strategy="simple")
outputs = ner_tagger(text)
pd.DataFrame(outputs)    

# %% [markdown]
# ### 질문 답변

# %%
reader = pipeline("question-answering")
question = "What does the customer want?"
outputs = reader(question=question, context=text)
pd.DataFrame([outputs])    

# %% [markdown]
# ### 요약

# %%
summarizer = pipeline("summarization")
outputs = summarizer(text, max_length=60, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])

# %% [markdown]
# ### 번역

# %%
translator = pipeline("translation_en_to_de", 
                      model="Helsinki-NLP/opus-mt-en-de")
outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print(outputs[0]['translation_text'])

# %% [markdown]
# ### 텍스트 생성

# %%
from transformers import set_seed
set_seed(42) # 동일 결과를 재현하기 위해 지정

# %%
generator = pipeline("text-generation")
response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
prompt = text + "\n\nCustomer service response:\n" + response
outputs = generator(prompt, max_length=200)
print(outputs[0]['generated_text'])

# %% [markdown]
# ## 허깅 페이스 생태계

# %% [markdown]
# <img alt="ecosystem" width="500" caption="An overview of the Hugging Face ecosystem of libraries and the Hub." src="https://github.com/rickiepark/nlp-with-transformers/blob/main/images/chapter01_hf-ecosystem.png?raw=1" id="ecosystem"/>

# %% [markdown]
# ### 허깅 페이스 허브

# %% [markdown]
# <img alt="hub-overview" width="1000" caption="The models page of the Hugging Face Hub, showing filters on the left and a list of models on the right." src="https://github.com/rickiepark/nlp-with-transformers/blob/main/images/chapter01_hub-overview.png?raw=1" id="hub-overview"/> 

# %% [markdown]
# <img alt="hub-model-card" width="1000" caption="A example model card from the Hugging Face Hub. The inference widget is shown on the right, where you can interact with the model." src="https://github.com/rickiepark/nlp-with-transformers/blob/main/images/chapter01_hub-model-card.png?raw=1" id="hub-model-card"/> 

# %% [markdown]
# ### 허깅 페이스 토크나이저

# %% [markdown]
# ### 허깅 페이스 데이터셋

# %% [markdown]
# ### 허깅 페이스 액셀러레이트

# %% [markdown]
# ## 트랜스포머의 주요 도전 과제

# %% [markdown]
# ## 결론


