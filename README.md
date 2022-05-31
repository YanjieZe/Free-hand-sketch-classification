# Free-Hand Sketch Classification

TODO List:
- [ ] logger
- [ ] data augmentation
- [ ] tensorboard

## Run
```
bash scripts/train.sh
```


## Our dataset (.png)
Our dataset is uploaded to google drive. The link is:
```
https://drive.google.com/file/d/1CVu5CljixuK9mjiQUEIVjY7rlSSdRzWu/view?usp=sharing
```

### Dataset Process

Before training an RPCL-pix2seq, you first need a pixel-formed sketch dataset translated from [QuickDraw dataset](https://quickdraw.withgoogle.com/data). Each sketch image is in **48x48x1**. The provided `seq2png.py` is used to create the required dataset. You are able to build your own pixel-formed dataset based on QuickDraw dataset with
``python seq2png.py``, and it follows an example usage.

```
python seq2png.py --input_dir=dataset/quickdraw --output_dir=dataset/quickdraw_png --png_width=28
```



gdown https://drive.google.com/uc?id=1tizohsP9u97Ql-ORg2Koezmq4LNWfxiG --output=dataset/quickdraw.zip

Each category of sketch images will be packaged in a single `.npz` file, and it will take about 30 to 60 minutes for each file translation. You might need the `svgwrite` python module, which can be installed as

```
conda install -c omnia svgwrite=1.1.6
```