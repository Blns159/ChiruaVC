# ChiruaVC: Vietnamese Voice conversation
Download [WavLM-Large](https://github.com/microsoft/unilm/tree/master/wavlm) and put it under directory 'wavlm/'
Download [HiFi-GAN model](https://github.com/jik876/hifi-gan) and put it under directory 'hifigan/' (for training with SR only)

```
cd ChiruaVC
conda env create -f environment.yml
python preprocess.py
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/freevc.json -m freevc
```
