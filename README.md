# Saptial and Temporal Difference Network for Real-time salient object detection

## Introduction
Salient object detection (SOD) and Video SOD (VSOD) have benefited from recent advances in deep convolutional neural networks (CNNs). However, top-performing large-scale models require considerable computational cost, making it hard to deploy them on resource-constrained devices. In this paper, we present SDCNet and STDCNet, lightweight architectures for SOD and VSOD that achieve state-of-the-art results with real-time inference speeds on embedded devices. SDCNet leverages pixel difference convolutions (PDC) to enrich the feature representation with image gradient information. Our proposed difference convolution reparameterization (DCR) strategy is effective in capturing the effect of multiple PDC operators at the computational cost of a single convolutional operator with no additional parameters. We additionally propose a novel spatiotemporal difference convolution (STDC) to complement standard 3D convolutions used for VSOD. STDCs similarly benefit from DCR and enable our resulting STDCNet to achieve high temporal consistency in VSOD predictions. Extensive experiments on six SOD and three VSOD datasets highlight the superior accuracy-runtime trade-offs that SDCNet and STDCNet achieve.

<div align=center>
<img src="https://user-images.githubusercontent.com/18327074/225762927-bbb3de74-2287-4db3-b143-61d2538a0f0d.png" width="80%"><br>
</div>

Coding style is based on [Pixel Difference Convolution](https://github.com/zhuoinoulu/pidinet).

## Environment
- Ubuntu 20.04 + cuda 11.7
- RTX 3090 x 2
- python 3.8, pytorch 1.12

*Other versions may also work~ :)*

## Dataset

#### create data folders
```bash
# please change the dir for ROOTDIR to where you want to store your data
ROOTDIR=/to/rootdir
mkdir ${ROOTDIR}/vsod
mkdir ${ROOTDIR}/sod
```

#### Download SOD datasets
- Training on [DUTS-TR](http://saliencydetection.net/duts/).
- Testing on [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [PASCAL-S](http://cbs.ic.gatech.edu/salobj/), [DUT-O](http://saliencydetection.net/dut-omron/), [SOD](https://www.elderlab.yorku.ca/resources/salient-objects-dataset-sod/), [HKU-IS](https://i.cs.hku.hk/~yzyu/research/deep_saliency.html), [DUTS-TE](http://saliencydetection.net/duts/).

For each dataset, please put the RGB images in folder `images` and ground truth images in `GT` (create them if there are no such folders), root them at ${ROOTDIR}/sod.

#### Download VSOD datasets
- [DAVSOD and DAVIS](https://github.com/DengPingFan/DAVSOD)
- VOS: [link1, where we downloaded it](https://github.com/Roudgers/DCFNet), or [link2](http://cvteam.net/projects/TIP18-VOS/VOS.html) (but please reorganize the files in its directory after downloading according to dataset/{train,val,test}\_video\_lst.txt)

Remember to unzip/unrar them to ${ROOTDIR}/vsod, and change the folder name for DAVSOD with the following scripts:
```bash
mv ${ROOTDIR}/vsod/Training\ Set ${ROOTDIR}/vsod/DAVSOD_Training_Set
mv ${ROOTDIR}/vsod/Validation\ Set ${ROOTDIR}/vsod/DAVSOD_Validation_Set
mv ${ROOTDIR}/vsod/Easy-35 ${ROOTDIR}/vsod/DAVSOD_Test_Set_Easy_35
```


## Evaluation

- Checkpoints of trained models can be found in [checkpoints](checkpoints).
- Saliency maps or our models can be downloaded at [saliency maps](https://github.com/isl-org/stdc-net/releases/download/v1.0/saliencyMaps.zip).

#### SOD (SDCNet w/o Imagenet pretraining)
```bash
ROOTDIR=/to/rootdir
testdata='ECSSD+PASCAL-S+SOD+DUT-O+HKU-IS+DUTS-TE'
#testdata='ECSSD' # if you want to evaluate on only a single dataset
size=320
exp='sdcnet_from_scratch'

# Difference Convolution Reparameterization (DCR) first,
# generating saliency maps second,

python generate_salmaps_sod.py --model sdcnet --inference-config baseline --train-config sdcnet -j 4 --gpu 0 --datadir ${ROOTDIR}/sod --testdata ${testdata} --savedir results/$exp --evaluate checkpoints/${exp}.pth --size ${size}

# calcualting metrics finally.

cd Evaluation
bash eval_sod.sh ${testdata} ${exp} ${ROOTDIR}/sod
```

#### SOD (SDCNet-A w/ Imagenet pretraining)
```bash
ROOTDIR=/to/rootdir
testdata='ECSSD+PASCAL-S+SOD+DUT-O+HKU-IS+DUTS-TE'
#testdata='ECSSD' # if you want to evaluate on only a single dataset
size=384
exp='sdcneta_from_pretrained'

# Difference Convolution Reparameterization (DCR) first,
# generating saliency maps second,

python generate_salmaps_sod.py --model sdcneta --bn --inference-config baseline --train-config sdcnet-a -j 4 --gpu 0 --datadir ${ROOTDIR}/sod --testdata ${testdata} --savedir results/$exp --evaluate checkpoints/${exp}.pth --size ${size}

# calcualting metrics finally.

cd Evaluation
bash eval_sod.sh ${testdata} ${exp} ${ROOTDIR}/sod
```

#### VSOD (STDCNet w/o Imagenet pretraining)
```bash
ROOTDIR=/to/rootdir
testdata='DAVSOD'
#testdata='VOS'
#testdata='DAVIS'
size=256
exp='stdcnet_from_scratch'
pos=_${testdata}
tid='cv cd ad'

# Difference Convolution Reparameterization (DCR) first,
# generating saliency maps second,

python generate_salmaps_vsod.py --model stdcnet --inference-config baseline --train-config sdcnet -j 4 --gpu 0 --datadir ${ROOTDIR} --testdata ${testdata} --savedir results/$exp --size $size --evaluate checkpoints/${exp}${pos}.pth --tid ${tid}

# calcualting metrics finally.

cd Evaluation
bash eval_vsod.sh ${testdata} ${exp} ${ROOTDIR}
```

#### VSOD (STDCNet-A w/ Imagenet pretraining)
```bash
ROOTDIR=/to/rootdir
testdata='DAVSOD'
#testdata='VOS'
#testdata='DAVIS'
size=256
exp='stdcneta_from_pretrained'
pos=_${testdata}
tid='cv cd ad'

# Difference Convolution Reparameterization (DCR) first,
# generating saliency maps second,

python generate_salmaps_vsod.py --model stdcneta --inference-config baseline --train-config sdcnet-a -j 4 --gpu 0 --datadir ${ROOTDIR} --testdata ${testdata} --savedir results/$exp --size $size --evaluate checkpoints/${exp}${pos}.pth --tid ${tid}

# calcualting metrics finally.

cd Evaluation
bash eval_vsod.sh ${testdata} ${exp} ${ROOTDIR}
```

Note: For calculating metrics for VSOD models, the [matlab evaluation tool](https://github.com/DengPingFan/DAVSOD/tree/master/EvaluateTool) can also be used. We implement it with python in above scripts which support GPU and ouput the same results much faster.



## Training

```bash
ROOTDIR=/to/rootdir

# Train SDCNet, w/o ImageNet pretraining
python train_sod.py --model sdcnet --config sdcnet --resume --gpu 0,1 --datadir ${ROOTDIR}/sod --savedir results/exp1

# Train SDCNetA, w/ ImageNet pretraining
python train_sod.py --model sdcneta --config sdcnet-a --resume --gpu 0,1 --datadir ${ROOTDIR}/sod --preload checkpoints/sdcneta_imagenet_pretrained_backbone.pth --savedir results/exp2

# Train STDCNet-A with suitable hyperparameters for DAVSOD, w/ ImageNet pretraining
python train_vsod.py --model stdcneta --config sdcnet-a --resume --gpu 0,1 --datadir ${ROOTDIR} --lr-reduce 0.01 --preload checkpoints/sdcneta_imagenet_pretrained_backbone.pth --savedir results/exp3

# Train STDCNet-A with suitable hyperparameters for VOS/DAVIS, w/ ImageNet pretraining
python train_vsod.py --model stdcneta --config sdcnet-a --resume --gpu 0,1 --datadir ${ROOTDIR} --lr-reduce 0.1 --preload checkpoints/sdcneta_imagenet_pretrained_backbone.pth --savedir results/exp4

# Train STDCNet with suitable hyperparameters for DAVSOD, w/o ImageNet pretraining (here, lr_reduce is 0.1, stage1 backbone can be obtained from SDCNet trained on DUTS-TR)
python train_vsod.py --model stdcnet --config sdcnet --resume --gpu 0,1 --datadir ${ROOTDIR} --lr-reduce 0.1 --preload checkpoints/sdcnet_stage1_backbone.pth --savedir results/exp5

# Train STDCNet with suitable hyperparameters for VOS/DAVIS, w/o ImageNet pretraining (here, lr_reduce is 0.01, stage1 backbone can be obtained from SDCNet trained on DUTS-TR)
python train_vsod.py --model stdcnet --config sdcnet --resume --gpu 0,1 --datadir ${ROOTDIR} --lr-reduce 0.01 --preload checkpoints/sdcnet_stage1_backbone.pth --savedir results/exp6
```

## Testing FPS

```bash
## Test speed of SDCNet, SDCNet-A
shape=320
python speed_sod.py --model sdcnet --config baseline -j 1 --gpu 0 --size $shape
python speed_sod.py --model sdcneta --config baseline -j 1 --gpu 0 --size $shape


## Test speed of STDCNet, STDCNet-A
shape=256
python speed_vsod.py --model stdcnet --config baseline -j 1 --gpu 0 --size $shape
python speed_vsod.py --model stdcneta --config baseline -j 1 --gpu 0 --size $shape
```

## Acknowledgement
Repositories by which the code writing is inspired:

- [PiDiNet](https://github.com/zhuoinoulu/pidinet)
- [Evaluation-on-salient-object-detection](https://github.com/Jun-Pu/Evaluation-on-salient-object-detection)
- [DCFNet](https://github.com/Roudgers/DCFNet)
- [DAVSOD](https://github.com/DengPingFan/DAVSOD)

Friendly colleagues at Intel Lab.
