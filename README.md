# SemBA

## Semantic-based Bayesian Attention for Human Scanpath Prediction

Code tested in Python 3.10.12 and nvidia-cudnn-cu12

- Clone this repository:
```
git clone https://github.com/vislab-tecnico-lisboa/SemBA.git
```

- Create a Python virtual environment:
```
cd SemBA
python -m venv semba_env
source semba_env/bin/activate
```

- Install requirements and dependencies:
```
pip install -r requirements.txt
```

- Test the code using one the provided examples:
```
python search.py -f examples/bottle.jpg -t bottle -d dfine -l 4 -b 160

```

## Reference
This repository contains code for the scanpath prediction pipeline described in the following papers. Please use the following citations if you use this code.

```bibtex
@article{semba_fast,
title = {SemBA-FAST: Semantic-based Bayesian attention applied to foveal active visual search tasks},
journal = {Neurocomputing},
volume = {673},
pages = {132860},
year = {2026},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2026.132860},
url = {https://www.sciencedirect.com/science/article/pii/S0925231226002572},
author = {João Luzio and Alexandre Bernardino and Plinio Moreno},
}

@InProceedings{luzio2025,
  author={Luzio, João and Bernardino, Alexandre and Moreno, Plinio},
  booktitle={2025 IEEE International Conference on Development and Learning (ICDL)}, 
  title={Human Scanpath Prediction in Target-Present Visual Search with Semantic-Foveal Bayesian Attention}, 
  year={2025},
  volume={},
  number={},
  pages={1-8},
  keywords={Visualization;Uncertainty;Computational modeling;Biological system modeling;Semantics;Predictive models;Benchmark testing;Probabilistic logic;Turning;Bayes methods},
  doi={10.1109/ICDL63968.2025.11204443}
}
```