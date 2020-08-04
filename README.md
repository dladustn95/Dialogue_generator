## Dialogue_generator



## Data
Source|Target 형태로 txt파일 구성.

아래의 형태로 같은 경로에 데이터가 존재해야 함.
Name_train.txt / Name_train_keyword.txt
Name_valid.txt / Name_valid_keyword.txt
Name_test.txt / Name_test_keyword.txt

## How to install

```sh
git clone https://github.com/dladustn95/Dialogue_generator.git
cd Dialogue_generator
pip install -r requirements.txt
pip install .
```
  
pytorch_pretrained_bert 패키지 폴더에 pretrain_bert안에 있는 tokenization2.py 파일 이동  
현재 폴더에 KoBert 실행을 위한 파일 추가 (vocab.korean.rawtext.list, pytorch_model.bin, bert_config.json)

### Requirements

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

```sh
python train.py --dataset_path DATAPATH/Name
python infer.py --dataset_path DATAPATH/Name --model_checkpoint MODELPATH/MODEL.pth

```

