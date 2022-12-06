# Neural Message Passing for Quantum Chemistry

Based on the article proposed by Gilmer *et al.* [1] and the code of [nmp_qc](https://github.com/priba/nmp_qc.git) project from Pau Riba and Anjan Dutta, we explored some other methods to improve the performance. 
The methodology includes two parts: Feature Engineering and Network Architecture Design.

- Feature Engineering

- Network Architecture Design

## Installation

    $ pip install -r requirements.txt

## Run the script

    $ python main.py
    
    $ python main.py --epochs 100 --mpnn --e_rep 6

    $ python main.py --epochs 100 --mpnnattn --method_attn 3 --num_heads 8 --e_rep 6

    (method_attn: 1, 2, 3, 4, 5 supported now)
    (e_rep: 1, 2, 3, 4, 5, 6 supported now)
    
## Installation of rdkit

Running any experiment using QM9 dataset needs installing the [rdkit](http://www.rdkit.org/) package, which can be done 
following the instructions available [here](http://www.rdkit.org/docs/Install.html)

## Data

The data used in this project can be downloaded [here](https://github.com/wei0py/mpnn/tree/master/data).

## Bibliography

- [1] Gilmer *et al.*, [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf), arXiv, 2017.
- [2] Petar Veličković *et al.*, [Graph attention networks](https://arxiv.org/pdf/1710.10903), arXiv, 2017.
- [3] Brody *et al.*, [How Attentive are Graph Attention Networks?](https://arxiv.org/pdf/2105.14491.pdf), arXiv, 2022.


## Authors

* Buyu Zhang (@wei0py)
* Zheyu Lu (@Nsigma-Bill)