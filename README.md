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
| Pretrained Model | Accuracy | Loss      | epoch | note                                   |
|------------------|----------|-----------|-------|----------------------------------------|
| VGG16            | 0.077    | 5.001     | -     | early stop, the performance is terrible |
| RESNET50         | 81.94    | 0.78      | 60    | early stop,                            |
| MOBILENET V2     | 81.96    | 0.72      | 240   | cool                                   |
|  DENSENET121     | 45.94    | 4.3338e+7 | 40    | early stop,                            |

## Built With
* [waverDeep](https://github.com/waverDeep) - model architecture, setup train/test pipeline

## Computing resources
| GPU RESOURCE              | RAM     | COUNT |
|---------------------------|---------|-------|
| NVIDIA TITAN RTX          | 24G     | 2     |
| NVIDIA GeForce GTX 1080TI | 12G     | 1     |
## Contribution
Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us. / [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) 를 읽고 이에 맞추어 pull request 를 해주세요.

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://gist.github.com/PurpleBooth/LICENSE.md) file for details / 이 프로젝트는 MIT 라이센스로 라이센스가 부여되어 있습니다. 자세한 내용은 LICENSE.md 파일을 참고하세요.
