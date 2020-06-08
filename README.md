## Dialogue_generator


#### Model
Bert Encoder, GPT2 Decoder를 사용하는 seq2seq 대화 생성 모델  
Adapter module 사용. 자세한 내용은 [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf) 참조.

#### Data

두 모델의 tokenizer가 다르기 때문에 dataset은 형태소 분석을 한 것, 하지 않은 것 모두 사용해야함  
형태소 분석이 된 source, 형태소 분석을 하지 않은 target을 input으로 줌

train : valid+test = 9 : 1 로 나눔, test는 15개

dataset은 name_{train,valid,test}{,_tag}의 형태로 같은 폴더에 존재해야 함

#### How to install

```sh
git clone https://github.com/dladustn95/Dialogue_generator.git
cd Dialogue_generator
pip install -r requirements.txt
pip install .
```


##### Requirements

* gluonnlp >= 0.8.3
* mxnet
* sentencepiece >= 0.1.6
* torch >= 1.4.0
* pytorch-ignite
* transformers==2.5.1
* tensorboardX==1.8
* tensorflow  # for tensorboardX
* pytorch-pretrained-bert

---

### How to use

```
sh train.sh
```

