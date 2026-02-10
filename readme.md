# PISCO: Precise Video Instance Insertion with Sparse Control

This repo hosts the official implementation of PISCO: Precise Video Instance Insertion with Sparse Control

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2602.08277)
[![Project Page](https://img.shields.io/badge/Project-Page-1f72ff.svg?style=for-the-badge)](https://xiangbogaobarry.github.io/PISCO/)
[![Development Tools](https://img.shields.io/badge/GitHub-Development_Tools-2ea44f.svg?style=for-the-badge)](https://github.com/XiangboGaoBarry/PISCO-Development-Tools)
[![Model-14B](https://img.shields.io/badge/HuggingFace-14B-orange.svg?style=for-the-badge)](https://huggingface.co/xiangbog/PISCO-14B/tree/main)
[![Model-1.3B](https://img.shields.io/badge/HuggingFace-1.3B-orange.svg?style=for-the-badge)](https://huggingface.co/xiangbog/PISCO-1.3B/tree/main)
<!-- [![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange.svg?style=for-the-badge)](https://github.com/taco-group/PISCO) -->


### TODO list

- [x] Release Inference Code
- [x] Release Development Tools
- [ ] Release Training Code
- [ ] Release Training Set

### Installation

```bash
conda create -n pisco python=3.12
conda activate pisco
# Install the correct version of torch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Install deepspeed if training 14B model
pip install deepspeed
```


### Inference

```bash
# 1.3B 480p
python inference/pretrained/infer_1.3B.py
# 1.3B 720p
python inference/pretrained/infer_1.3B_720p.py
# 14B 480p
python inference/pretrained/infer_14B.py
# 14B 720p
python inference/pretrained/infer_14B_720p.py
```


[![Star History Chart](https://api.star-history.com/svg?repos=taco-group/PISCO&type=Date)](https://star-history.com/#taco-group/PISCO&Date)


### Acknowledgments

This repo is built upon the [Diffsynth-Studio](https://github.com/modelscope/DiffSynth-Studio) codebase. Thanks to the authors for their great work!