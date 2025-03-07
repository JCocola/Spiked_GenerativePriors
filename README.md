# Nonasymptotic Guarantees for Low-Rank Matrix Recovery with Generative Priors  

This repository contains the accompanying code for the NeurIPS paper:  

**_Nonasymptotic Guarantees for Low-Rank Matrix Recovery with Generative Priors_**  

**⚠ Note:** This repository does *not* include the full paper—only the code used for experiments. The full paper can be accessed at the following link:  
🔗 [NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/hash/ad62cfd33e3870262d6bf5331c1f13b0-Abstract.html)  

## Requirements  

The code is written in Python and relies on PyTorch. The following libraries are required:  

- Python 3  
- PyTorch  
- NumPy  
- Matplotlib  

The provided notebooks are ready to run on Google Colab.  

## Running Synthetic Experiments  

The following two files were used to produce Figure 1 in the paper:  
- `Scaling_Wishart.ipynb`  
- `Scaling_Wigner.ipynb`  

A run of Algorithm 1 is demonstrated in the file `Recovery_Wishart.ipynb`.  

## Citation  

If you find this work useful, please cite:  

```bibtex
@article{cocola2020nonasymptotic,
  title={Nonasymptotic guarantees for spiked matrix recovery with generative priors},
  author={Cocola, Jorio and Hand, Paul and Voroninski, Vlad},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={15185--15197},
  year={2020}
}
```  
