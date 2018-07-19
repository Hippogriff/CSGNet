# CSGNet: Neural Shape Parser for Constructive Solid Geometry
This repository contains code accompaning the paper: [CSGNet: Neural Shape Parser for Constructive Solid Geometry, CVPR 2018](https://arxiv.org/abs/1712.08290).

Here we only include the code for 2D CSGNet. Code for 3D is available on this [repository](https://github.com/Hippogriff/3DCSGNet).

![](docs/image.png)
### Dependency
- Python 3.*
- Please use conda env using environment.yml file.
  ```bash
  conda env create -f environment.yml -n CSGNet
  source activate CSGNet
  ```

### Data
- Synthetic Dataset:

    Download the synthetic [dataset](https://www.dropbox.com/s/ud3oe7twjc8l4x3/synthetic.tar.gz?dl=0) and CAD [Dataset](https://www.dropbox.com/s/d6vm7diqfp65kyi/cad.h5?dl=0). Pre-trained model is available [here](https://www.dropbox.com/s/0f778edn3sjfabp/models.tar.gz?dl=0). Synthetic dataset is provided in the form of program expressions, instead of rendered images. Images for training, validation and testing are rendered on the fly. The dataset is split in different program lengths.
    ```bash
    tar -zxvf synthetic.tar.gz -C data/
    ```

- CAD Dataset

    Dataset is provided in H5Py format.
    ```bash
    mv cad.h5 data/cad/
    ```

### Supervised Learning
- To train, update `config_synthetic.yml` with required arguments. Default arguments are already filled. Then run:
    ```python
    python train_synthetic.py
    ```

- To test, update `config_synthetic.yml` with required arguments. Default arguments are already filled. Then run:
    ```python
    # For top-1 testing
    python test_synthetic.py
    ```
    ```python
    # For beam-search-k testing
    python test_synthetic_beamsearch.py
    ```

### RL fintuning
- To train a network using RL, fill up configuration in `config_cad.yml` or keep the default values and then run:
    ```python
    python train_cad.py
    ```
    Make sure that you have trained a network used Supervised setting first.

- To test the network trained using RL, fill up configuration in `config_cad.yml` or keep the default values and then run:
  ```python
  # for top-1 decoding
  python test_cad.py
  ```
  ```python
  # beam search decoding
  python test_cad_beamsearch.py
  ```
  For post processing optmization of program expressions (visually guided search), set the flag `REFINE=True` in the script `test_cad_beam_search.py`, although it is little slow. For saving visualization of beam search use `SAVE_VIZ=True`

- To optmize some expressions for cad dataset:
  ```
  # To optmize program expressions from top-1 prediction
  python refine_cad.py path/to/exp/to/optmize/exp.txt  path/to/directory/to/save/exp/
  ```
  Note that the expression files here should only have 3k expressions corresponding to the 3k test examples from the CAD dataset.

- To optmize program expressions from top-1 prediction
  ```
  python refine_cad_beamsearch.py path/to/exp/to/optmize/exp.txt  path/to/directory/to/save/exp/
  ```
  Note that the expression files here should only have 3k x beam_width expressions corresponding to the 3k test examples from the CAD dataset.

- To visualize generated expressions (programs), look at the script `visualize_expressions.py`


### Cite:
```bibtex
@InProceedings{Sharma_2018_CVPR,
author = {Sharma, Gopal and Goyal, Rishabh and Liu, Difan and Kalogerakis, Evangelos and Maji, Subhransu},
title = {CSGNet: Neural Shape Parser for Constructive Solid Geometry},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```

### Contact
To ask questions, please [email](mailto:gopalsharma@cs.umass.edu).
