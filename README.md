# Neural Message Passing for Quantum Chemistry

Based on the article proposed by Gilmer *et al.* [1] and the code of [nmp_qc](https://github.com/priba/nmp_qc.git) project from Pau Riba and Anjan Dutta, we explored some other methods to improve the performance. 
The methodology includes 

## Installation

    $ pip install -r requirements.txt
    $ python main.py
    $ python main.py --mpnn
    $ python main.py --ggnn
    $ python main.py --epochs 4 --mpnnattn --method_attn 1 --num_heads 4 --e_rep 4
    (method 1,2,3, 4, 5 supported now)
    (e_rep: 1-5 supported now)
    
## Installation of rdkit

Running any experiment using QM9 dataset needs installing the [rdkit](http://www.rdkit.org/) package, which can be done 
following the instructions available [here](http://www.rdkit.org/docs/Install.html)

## Data

The data used in this project can be downloaded [here](https://github.com/priba/nmp_qc/tree/master/data).

## Bibliography

- [1] Gilmer *et al.*, [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf), arXiv, 2017.



## Authors

* 