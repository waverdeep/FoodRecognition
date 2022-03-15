# Food Recognition/ Ingredient Recognition
- 사진 안에 있는 음식이 어떤 음식인지 인식할 수 있는 인공지능 모델 개발
- 사진 안에 있는 식재료가 어떤 식재료인지 인식할 수 있는 인공지능 모델 개발 

## Getting Started
You can start on any computer that can learn deep learning.
If you want to learn fast, use GPU-workstation.
### Training Prerequisites
```
python version == 3.6.9
```

```
torch~=1.9.1
torchvision~=0.10.1
Pillow~=7.0.0
natsort~=7.0.1
sklearn~=0.0
scikit-learn~=0.24.2
tqdm~=4.42.0
numpy~=1.18.1
tensorflow~=2.2.0
tensorflow-gpu~=2.2.0
tensorboard~=2.7.0
matplotlib~=3.1.2
```

```
pip install -r requirements.txt
```

## How can I request Checkpoint?
If you respond to GoogleForms, we will share the download link within a few days. 
Currently, the shared checkpoint is ResNet152.
 - [Resquest JIT Traced Checkpoint](https://forms.gle/AqxwTx6owSvMk6Su9)
 - [Request Pytorch Checkpoint](https://forms.gle/T18o5EKRERcDe2tR6)

## Running the Test
 - Use TorchScript
```
test_model = torch.jit.load('./jit_traced_torch_model_name.pt', map_location='cpu')
sample_data = torch.randn(1, 3, 512, 512) # (1, channel, width, height)
out_data = test_model(sample_data)
``` 
 - pytorch
```
python inference.py --config configure_name.json --image image_name.jpg --label labels.jpg
```


## Quick Start Training Guide
1. You need to create a configuration first.
2. Then execute the following command:
```
python train.py --configuration configuration_name.json
```

## Dataset
- [KOREA AI HUB DATASET](https://aihub.or.kr/aidata/13594) - 한국 음식 이미지 데이터셋 
- [Fruit and Vegetable Image Dataset](https://www.kaggle.com/kritikseth/fruit-and-vegetable-image-recognition) - 과일과 채소 데이터셋
- [Vegetable Image Dataset](https://www.kaggle.com/misrakahmed/vegetable-image-dataset) - 채소 데이터셋
- Private Dataset (INGD_V1, INGD_V2)

## Support TorchScript
 - [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) - TorchScript 소개 및 튜토리얼
 - [Deep Java Library Pytorch](https://docs.djl.ai/jupyter/load_pytorch_model.html) - Pytorch용 Deep Java Library Engine Provider 소개 및 튜토리얼


## Baseline Results - Food Recognition
| Pretrained Model | Accuracy | Loss      | epoch | note                                    |
|------------------|----------|-----------|-------|-----------------------------------------|
| VGG16            | 0.077    | 5.001     | -     | early stop, the performance is terrible |
| RESNET50         | 81.94    | 0.78      | 60    | early stop,                             |
| RESNET152        | 73.77    | 0.973     | 20    | comming soon!                           |
| WIDERESNET50_2   | 72.52    | 0.998     | 20    | comming soon!                           |
| MOBILENET V2     | 81.96    | 0.72      | 240   | cool, stop training                     |
| DENSENET121      | 45.94    | 4.3338e+7 | 40    | early stop,                             |


## Baseline Results - Ingredient Recognition 
| Pretrained Model | Accuracy | Loss | epoch | note                 | dataset                          | num of class |
|------------------|----------|------|-------|----------------------|----------------------------------|--------------|
| VGG16            | -        | -    | -     | poor accuracy        | INGD_V1 (private)                | 58           |
| RESNET50         | -        | -    | -     | poor accuracy        | INGD_V1 (private)                | 58           |
| RESNET152        | 95.44    | 0.68 | 250   | fruits and vegs only | Food and Vegetable Image Dataset | 58           |
| RESNET152        | 92.19    | 0.41 | 376   | nice accuracy        | INGD_V1 (private)                | 58           |
| WIDERESNET50_2   | -        | -    | -     | poor accuracy        | INGD_V1 (private)                | 58           |
| MOBILENET V2     | 82.55    | 0.70 | 282   | not bad              | INGD_V1 (private)                | 58           |
| DENSENET121      | -        | -    | -     | poor accuracy        | INGD_V1 (private)                | 58           |

## Stage2 Result - Ingredient Recognition
| Pretrained Model | Accuracy | Loss | epoch | note            | dataset           | num of class |
|------------------|----------|------|-------|-----------------|-------------------|--------------|
| RESNET152        | 83.03    | 0.71 | 40    | now available!  | INGD_V2 (private) | 238          |
| MOBILENET V2     |          |      |       | comming soon!   | INGD_V2 (private) | 238          |




## Built With
* [waverDeep](https://github.com/waverDeep) - model architecture, setup train/test pipeline

## Computing resources
| GPU RESOURCE              | RAM     | COUNT | NOTE                 |
|---------------------------|---------|-------|----------------------|
| NVIDIA TITAN RTX          | 24G     | 2     | training             |
| NVIDIA GeForce GTX 1080TI | 12G     | 1     | develop, test or etc |

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/waverDeep/FoodRecognition/blob/main/LICENSE) file for details
