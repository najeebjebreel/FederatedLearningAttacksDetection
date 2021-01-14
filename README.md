# Detection of Byzantine Attacks in Federated Learning 

This repository is a primary version of the source code and models of the paper Efficient Detection of Byzantine Attacks in Federated Learning Using Last Layer Biases. The repository uses PyTorch to implement the experiments.

## Paper 

[Efficient Detection of Byzantine Attacks in Federated Learning Using Last Layer Biases] (https://crises-deim.urv.cat/web/docs/publications/lncs/1117.pdf)
</br>
[Najeeb Moharram Jebreel](https://crises-deim.urv.cat/)<sup>1</sup>, [Josep Domingo-Ferrer](https://crises-deim.urv.cat/)<sup>1</sup>, [David Sánchez](https://crises-deim.urv.cat/)<sup>1</sup>, [Alberto Blanco-Justicia](https://crises-deim.urv.cat/)<sup>1</sup>
</br>
<sup>1 </sup> Universitat Rovira i Virgili, Department of Computer Engineering and Mathematics, CYBERCAT-Center for
Cybersecurity Research of Catalonia, UNESCO Chair in Data Privacy, Av. Països Catalans 26, 43007 Tarragona,
Catalonia
</br>

## Content
The repository contains one main jupyter notebook: `Experiments.IPYNB` in each data set folder. These notebooks can be used to train, predict, and fine-tune models. 

Additionally, this repo contains some images from different distributions that used to embed the watermarks.

The code supports training and evaluating on [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) and [MNIST](http://yann.lecun.com/exdb/mnist/) datasets.


## Dependencies

[Python 3.6](https://www.anaconda.com/download)

[PyTorch 1.6](https://pytorch.org/)

## Citation 
If you find our work useful please cite:

@inproceedings{jebreel2020efficient,
  title={Efficient Detection of Byzantine Attacks in Federated Learning Using Last Layer Biases},
  author={Jebreel, Najeeb and Blanco-Justicia, Alberto and S{\'a}nchez, David and Domingo-Ferrer, Josep},
  booktitle={International Conference on Modeling Decisions for Artificial Intelligence},
  pages={154--165},
  year={2020},
  organization={Springer}
}

## Funding
This research was funded by the European Commission (projects H2020-871042 “SoBigData++” and
603 H2020-101006879 “MobiDataLab”), the Government of Catalonia (ICREA Acadèmia Prizes to J. Domingo-Ferrer
604 and D. Sánchez, FI grant to N. Jebreel and grant 2017 SGR 705), and the Spanish Government (projects
605 RTI2018-095094-B-C21 “Consent” and TIN2016-80250-R “Sec-MCloud”).




