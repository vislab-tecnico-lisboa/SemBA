# Semantic-based Bayesian Attention Framework

![Static Badge](https://img.shields.io/badge/Python-3.10.12-blue)

*SemBA* (**Semantic-based Bayesian Attention**) is a top-down probabilistic framework for human scanpath prediction in visual search. It leverages pretrained deep object detectors to extract semantic cues at each fixation and fuses them over time using a Bayesian update mechanism based on Dirichlet beliefs. At every step, *SemBA* builds and updates class-specific semantic belief maps, selecting the next fixation by maximizing the posterior probability of the target class while applying inhibition of return. By integrating artificial foveation to mimic retinal eccentricity and reduce computational cost, *SemBA* achieves biologically inspired, cost-efficient active perception without relying on human scanpath supervision, closely approximating human search behavior on benchmarks such as [COCO-Search18](https://sites.google.com/view/cocosearch/).

*SemBA* integrates the **Multi-Scale Fovea** mechanism, that approximates the dynamics of the human field-of-view. At each fixation, it crops several increasingly larger regions around the gaze point, downsamples them to the same size, and runs object detection on each scale. This reduces computational cost while preserving central detail and introducing realistic peripheral uncertainty.

The code contained in this repository was developed and tested on *Python 3.10.12*.

## Installation

- Clone the repository into a local directory:
```
git clone https://github.com/vislab-tecnico-lisboa/SemBA.git
cd SemBA
```

- Create a Python (3.10.12) virtual environment (optional, yet recommended):
```
python -m venv semba_env
source semba_env/bin/activate
```

- Install requirements and dependencies:
```
pip install -r requirements.txt
```

- Test the code using one of the provided examples:
```
python search.py -f examples/bottle.jpg -t bottle -d dfine -l 4 -b 160
```

## Reference
This repository contains code for the SemBA pipeline, described in the following papers: 
```bibtex
@InProceedings{luzio2025,
  author={Luzio, João and Bernardino, Alexandre and Moreno, Plinio},
  booktitle={2025 IEEE International Conference on Development and Learning (ICDL)}, 
  title={Human Scanpath Prediction in Target-Present Visual Search with Semantic-Foveal Bayesian Attention}, 
  year={2025},
  pages={1-8},
  doi={10.1109/ICDL63968.2025.11204443}
}

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
```
Please use these citations if you come to work with this code, our *SemBA* framework, or the *Multi-Scale Fovea* mechanism.

Please directly contact [*João Luzio*](https://sites.google.com/view/joaoluzio) at ```joaoluzio14@tecnico.ulisboa.pt``` for any further enquiry.