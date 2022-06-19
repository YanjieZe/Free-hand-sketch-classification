# Free-Hand Sketch Classification


## Run
See `scripts/` for training scripts.

## Our dataset: QuickDraw (both png and svg)
Our dataset is uploaded to google drive. The link is:
- [png file](https://drive.google.com/file/d/1CVu5CljixuK9mjiQUEIVjY7rlSSdRzWu/view?usp=sharing)
- [svg file](https://drive.google.com/file/d/1tizohsP9u97Ql-ORg2Koezmq4LNWfxiG/view?usp=sharing)


## Results

| alg | epoch | acc |
| --- | --- | --- |
| sketch r2cnn | 20 | 7.0368 |
| sketch r2cnn | 40 | 6.6208 |
| sketch r2cnn | 60 | 6.304 |
| vit | 20 | 64.1568 |
| vit | 40 | 66.416 |
| vit | 60 | 64.816 |
| vit | 80 | 61.5632 |
| vit | 100 | 61.6688 |
| vit | test | 61.9742 |
| densenet | 1 | 81.2528 |
| densenet | 2 | 83.2272 |
| densenet | 3 | 83.9488 |
| densenet | 4 | 84.5344 |
| densenet | 5 | 84.6576 | 
| densenet | 6 | 85.072 | 
| densenet | 7 | 85.2192 |
| densenet | 8 | 85.4304 |
| resnet18 | 20 | 77.576 |
| resnet18 | 40 | 76.1104 |
| resnet18 | 60 | 75.3152 |
| resnet18 | 80 | 75.0816 |
| resnet18 | 100 | 75.0112 |
| resnet18 | test | 75.2336 |
| resnet34 | 20 | 77.3008 |
| resnet34 | 40 | 76.1072 |
| resnet34 | 50 | 75.936 |
| resnet50 | 20 | 76.7776 |
| resnet50 | 40 | 76.0992 |
| VGG16 | 20 | 80.6224 |
| VGG16 | 40 | 79.56 |
| VGG16 | 60 | 79.8224 |


