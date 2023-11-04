# Segment Shards

This repository is the official implementation of Segment Shards: Prompt Batch Adversarial Attack with Momentum to Segment Anything Model.

## Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

env : Ubuntu 18.04  python 3.8  Pytorch 1.8.1+cu111

## Attacking

To generate the adversarial example(s) in the paper, run this command:

- PBA

  ```bash
  python attack_pba.py --config config/PBA/Apollo_10_816b.yaml
  ```
- SPA

  ```bash
  python attack.py --config config/SPA/Apollo_10_816b.yaml
  ```
- PA

  ```bash
  python attack.py --config config/PA/Apollo_10_864b.yaml
  ```
- NPA

  ```bash
  python attack_npa.py --config config/NPA/Apollo_10_816b.yaml
  ```

> ps. You can use the config file to change the detial of attacking

## Evaluation

To evaluate the adversarial examples generated, run:

```bash
# calculate miou
python eval_miou.py --config config/PBA/eval_Apollo_10_816b_816b.yaml
```

```bash
# calculate ssim
python eval_ssim.py --config config/PBA/Apollo_10_816b.yaml
```

> the output of evaluation is a file named *.csv

## Pre-trained Models

You can download pretrained models here:

- [SAM vit_b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- [SAM vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

Then, put them into checkpoints directory, your project tree may like:

```
<root of code>
├── checkpoints
│   ├── sam_vit_b01ec64.pth
│   └── sam_vit_h_4b8939.pth
├── add_permutations.ipynb
├── attack_pba.py
├── attack_npa.py
├── attack.py
|...

```

## Download Datasets

You can download datasets from [Apollo Dataset link](https://apolloscape.auto/trajectory.html#to_download_href), [BDD100k Dataset link](https://bair.berkeley.edu/blog/2018/05/30/bdd/), [CBCL Dataset link](http://cbcl.mit.edu/software-datasets/streetscenes/).

You can put these data in advDataset directory. Your project tree may like:

```
<root of code>
├── checkpoints
│   ├── sam_vit_b01ec64.pth
│   └── sam_vit_h_4b8939.pth
├── advDataset
│   ├── advApollo
│   |   └──...
│   └── advBDD
│       └──...
├── add_permutations.ipynb
├── attack_pba.py
├── attack_npa.py
├── attack.py
|...

```

## Contributing

> The Segment Shards project was made with the help of [Segment Anything project](https://github.com/facebookresearch/segment-anything).
