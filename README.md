
# Dynamic Graph Transformer with Correlated Spatial-Temporal Positional Encoding (CorDGT)

![Static Badge](https://img.shields.io/badge/Conference-WSDM2025-FF8C00)

This is the official implementation of WSDM 2025 paper: 
> Zhe Wang, Sheng Zhou, Jiawei Chen, Zhen Zhang, Binbin Hu, Yan Feng, Chun Chen, Can Wang. *Dynamic Graph Transformer with Correlated Spatial-Temporal Positional Encoding*. [[arXiv link]](https://arxiv.org/abs/2407.16959)


## Datasets Downloading and Preprocessing
All the datasets can be downloaded [here](https://zenodo.org/records/7213796#.Y1cO6y8r30o) and put them into `processed` folder. Then run the following command: 
```
python process.py 
```
We use the dense `npy` format to save the features in binary format. If edge features or nodes features are absent, it will be replaced by a vector of zeros. Then put your data under `processed` folder.



## Training and Evaluation Commands

* Using CorDGT for Dynamic Link Prediciton: 
```
# Enron 
python -u learn_edge.py -d enron --uniform --bs 100 --n_degree 20 1 --n_head 6

# UCI
python -u learn_edge.py -d uci --bs 100 --uniform --n_degree 32 1 --n_head 6 --alpha 1 --beta 0.1 
```

## Requirements
* python >= 3.8.0
* torch >= 1.9.1
* Full Dependency list is in `requirements.txt`

## Acknowledgments
Portions of the code utilized in this project are based on [TGAT](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs). We are grateful for their contributions. 

## Citations
If you find the paper useful in your research, please consider citing:
```
@inproceedings{wang2024dynamic,
  title={Dynamic Graph Transformer with Correlated Spatial-Temporal Positional Encoding},
  author={Wang, Zhe and Zhou, Sheng and Chen, Jiawei and Zhang, Zhen and Hu, Binbin and Feng, Yan and Chen, Chun and Wang, Can},
  booktitle={Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining},
  year={2025}
}
```







