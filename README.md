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
## running the code
### training
Before running the code, a dataset of interest must be extracted, for example:
```
tar -xzvf ./datasets/2.2nm_simulation_cells/240each/fold1.tar.gz
```
to run the code:
```
python train.py ./datasets/2.2nm_simulation_cells/fold1 model_checkpoints log.model GANN heterogeneous
```
#### explanation
train.py: script to run training

./datasets/2.2nm_simulation_cells/240each/fold1: path to a folder with training samples

note: make sure the number of samples in the folder equals train_size + val_size in custom_config.yaml

model_checkpoints: name of folder where model checkpoints (model parameters) are saved

log.model: text file that saves training loss, validation loss, validation accuracy

GANN: tells code to run Graph Attention Network

heterogeneous: tells code to do two-body re-scaling

### prediction
```
python predict.py ./target ./model_checkpoints/model_best.pt GANN heterogeneous
```
