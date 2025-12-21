# GNN4CuZrMG
## Graph Neural Network used to characterize metallic glass

### Instructions for setup
Note: you may require system specific modules to be loaded if you work on a supercomputing cluster

#### step 1: clone repository from GitHub
```
git clone https:/github.com/ejgurniak/GNN4CuZrMG
```
#### step 2: create a conda environment in which to run the code
```
conda env create --name gnn-mg -f environment.yml
```
If the above does not work, try replacing "conda" with "mamba"
#### step 3: activate the environment
```
conda activate gnn-mg
```
#### step 4: add torch geometric package with pip
```
pip install torch_geometric
```
### running the code
