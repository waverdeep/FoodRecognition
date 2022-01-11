# FoodRecognition/ Multi-label Food Classification
사진안에 있는 음식이 어떤 음식인지 인식할수 있는 인공지능 모델 개발

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
## Running the tests
```
python inference.py
```

## Quick Start Guide
```

```
## Baseline Results
| Pretrained Model | Accuracy | Loss      | epoch | note                                    |
|------------------|----------|-----------|-------|-----------------------------------------|
| VGG16            | 0.077    | 5.001     | -     | early stop, the performance is terrible |
| RESNET50         | 81.94    | 0.78      | 60    | early stop,                             |
| RESNET152        | 73.77    | 0.973     | 20    | comming soon!                           |
| WIDERESNET50_2   | 72.52    | 0.998     | 20    | comming soon!                           |
| MOBILENET V2     | 81.96    | 0.72      | 240   | cool, stop training                     |
| DENSENET121      | 45.94    | 4.3338e+7 | 40    | early stop,                             |


## Built With
* [waverDeep](https://github.com/waverDeep) - model architecture, setup train/test pipeline

## Computing resources
| GPU RESOURCE              | RAM     | COUNT | NOTE                 |
|---------------------------|---------|-------|----------------------|
| NVIDIA TITAN RTX          | 24G     | 2     | training             |
| NVIDIA GeForce GTX 1080TI | 12G     | 1     | develop, test or etc |

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/waverDeep/FoodRecognition/blob/main/LICENSE) file for details
