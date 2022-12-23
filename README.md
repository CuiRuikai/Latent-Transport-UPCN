# Energy-Based Residual Latent Transport for Unsupervised Point Cloud Completion

Created by [Ruikai Cui](https://ruikai.ink/), [Shi Qiu](https://shiqiu0419.github.io/), [Saeed Anwar](https://saeed-anwar.github.io/), [Jing Zhang](https://jingzhang617.github.io/), [Nick Barnes](http://users.cecs.anu.edu.au/~nmb/)

[[arXiv]](https://arxiv.org/abs/2211.06820) [[poster]](https://ruikai.ink/static/files/latent-transport-upcn.pdf) [[Supplymentry]](https://ruikai.ink/static/files/Supplementary_Material.pdf)

---
## To-do
 + [ ] More instruction coming soon

## Usage
### Requirements

#### Building Pytorch Extensions for Chamfer Distance, PointNet++ and kNN

### Dataset
Preprocessed 3D-EPN dataset can be downloaded from [[Google Drive]](https://drive.google.com/file/d/1TxM8ZhaKEZWWSnakU2KGBLAO0pRnKDKo/view?usp=sharing)
Place the dataset in ```./data/EPN3D/```
### Training

train the model
```
python main.py --config ./cfgs/EPN3D_models/ResEBM.yaml --exp_name exp_name
```

## License
MIT License

## Acknowledgements

Our code is inspired by [PoinTr](https://github.com/yuxumin/PoinTr).

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{cui2022ltupcn,
  title={Energy-Based Residual Latent Transport for Unsupervised Point Cloud Completion},
  author={Cui, Ruikai and Qiu, Shi and Anwar, Saeed and Zhang, Jing and Barnes, Nick},
  booktitle={BMVC},
  year={2022}
}
```
