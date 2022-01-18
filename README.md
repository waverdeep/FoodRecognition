# Food Recognition/ Ingredient Recognition
- 사진 안에 있는 음식이 어떤 음식인지 인식할 수 있는 인공지능 모델 개발
- 사진 안에 있는 식재료가 어떤 식재료인지 인식할 수 있는 인공지능 모델 개발 

## Getting Started
You can start on any computer that can learn deep learning.
If you want to learn fast, use GPU-workstation.
### Prerequisites
```
torch==1.9.0
torchvision==0.9.0
tensorboard==2.2.2
tensorflow-gpu==2.2.0
```

### Installing
```
python setup.py
```

## Dataset
- [KOREA AI HUB DATASET](https://aihub.or.kr/aidata/13594) - 한국 음식 이미지 데이터셋 
- [Fruit and Vegetable Image Dataset](https://www.kaggle.com/kritikseth/fruit-and-vegetable-image-recognition) - 과일과 채소 데이터셋
- [Vegetable Image Dataset](https://www.kaggle.com/misrakahmed/vegetable-image-dataset) - 채소 데이터셋

## Running the tests
```
python inference.py
```

## Quick Start Guide
```
python train_supervised.py
```
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
| Pretrained Model | Accuracy | Loss | epoch | note          |
|------------------|----------|------|-------|---------------|
| VGG16            |          |      |       | comming soon! |
| RESNET50         |          |      |       | comming soon! |
| RESNET152        | 45.01    | 1.96 | 7     | training now! |
| WIDERESNET50_2   |          |      | -     | comming soon! |
| MOBILENET V2     |          |      | -     | comming soon! |
| DENSENET121      |          |      | -     | comming soon! |

## Built With
* [waverDeep](https://github.com/waverDeep) - model architecture, setup train/test pipeline

## Computing resources
| GPU RESOURCE              | RAM     | COUNT | NOTE                 |
|---------------------------|---------|-------|----------------------|
| NVIDIA TITAN RTX          | 24G     | 2     | training             |
| NVIDIA GeForce GTX 1080TI | 12G     | 1     | develop, test or etc |

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/waverDeep/FoodRecognition/blob/main/LICENSE) file for details
