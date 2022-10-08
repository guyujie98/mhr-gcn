## Requirements
```
deeprobust==0.2.4
dgl==0.4.3
torch==1.11.0
ipdb==0.13.2
scipy==1.8.0
numpy==1.20.3
pandas==1.4.1
torch-geometric==2.0.4
scikit_learn==1.0.2
networkx==2.7.1
```

## Installation
```
pip install -r requirements.txt
```
# Run codes
'--runs 10' means that we repeat 10 times 

## Under Graph Structure Attacks (GSAs)
On different datasets: epochs 201 for cora and citeseer and  301 for pubmed.  
You can report our results tuning the '--ptb_rate'  
perturbation rate of meta in [0.05,0.1,0.15,0.2,0.25]  
perturbation rate of random in [0.2,0.4,0.6,0.8,1.0]  
perturbation number of sga in [1,2,3,4,5]  
```
# performance of MHR-GCN on clean graphs
python main.py --attack meta  --dataset cora --threshold 0.1  --epochs 201 --runs 10 --device cuda:0 
# performance of MHR-GCN under non-targeted attacks
python main.py --attack meta --ptb_rate 0.25 --dataset cora --threshold 0.1  --epochs 201 --runs 10 --device cuda:0 --conduct_attack
# performance of MHR-GCN under random attacks
python main.py --attack random --ptb_rate 1.0 --dataset cora --threshold 0.1  --epochs 201 --runs 10 --device cuda:0 --conduct_attack
# performance of MHR-GCN on target nodes under no attacks
python main.py --attack sga --ptb_rate 5 --dataset cora --threshold 0.1 --epochs 201 --runs 10 --device cuda:0 
# performance of MHR-GCN under targeted attacks
python main.py --attack sga --ptb_rate 5 --dataset cora --threshold 0.1 --epochs 201 --runs 10 --device cuda:0 --conduct_attack
```
