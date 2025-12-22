import json
import os
import sys
from glob import glob
from random import *
from tempfile import TemporaryFile

# from utilis import ROC_AUC_multiclass
import numpy as np
import torch
import yaml
from natsort import natsorted
from pymatgen.io.vasp.inputs import Poscar
import torch.nn as nn

# from graph import CrystalGraphDataset, prepare_batch_fn

# from ignite.contrib.handlers.tensorboard_logger import *
from settings import Settings

validation_data = sys.argv[1]

try:
    checkpoint_best = sys.argv[2]
except:
    raise Exception("model parameters are needed for prediction")

try:
    current_model = sys.argv[3]
except:
    raise Exception("make sure you use the same model as was used in training")

try:
    is_heterogeneous = sys.argv[4]
except:
    is_heterogeneous = "homogeneous"
    print("NOTE: heterogeneity not chosen, defaulting to homogeneous version")
    print("NOTE: make sure you chose homogeneous for the training")

if is_heterogeneous == "heterogeneous":
    print("heterogeneous graph chosen, input features will be modified based on Cu-Zr")
    from hetgraph import CrystalGraphDataset, prepare_batch_fn
else:
    print("homogeneous graph chosen, no modification of input features")
    from graph import CrystalGraphDataset, prepare_batch_fn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


poscars = natsorted(glob("{}/*.POSCAR".format(validation_data)))


# In[28]:


if os.path.exists("custom_config.yaml"):
    with open('custom_config.yaml', 'r') as file:
        custom_dict = yaml.full_load(file)
        settings = Settings(**custom_dict)
else:
    settings = Settings()


# In[26]:

classification_dataset = []
struct_label = []

for file in poscars:
    label = file.split("/")[-1].replace(".POSCAR", "")
    poscar = Poscar.from_file(file)
    struct_label.append(label)

    dictionary = {
        'structure': poscar.structure,
        'target': np.array(
            [1]
        ),  # dummy target where target of the vaidation data is not known
    }

    classification_dataset.append(dictionary)


# In[27]:


# -----------data loading-----------------------


graphs = CrystalGraphDataset(
    classification_dataset,
    neighbors=settings.neighbors,
    rcut=settings.rcut,
    delta=settings.search_delta,
)



# # Load model

# In[30]:



if current_model == "GANN":
    print("GANN chosen")
    from model import CEGAN
    net = CEGAN(
        settings.gbf_bond,
        settings.gbf_angle,
        n_conv_edge=settings.n_conv_edge,
        n_conv_angle=settings.n_conv_angle,
        h_fea_edge=settings.h_fea_edge,
        h_fea_angle=settings.h_fea_angle,
        n_classification=settings.n_classification,
        pooling=settings.pooling,
        embedding=True,
        num_MLP=settings.num_MLP,
    )
elif current_model == "GIN":
    print("GIN chosen")
    from model import GIN
    net = GIN(
        settings.gbf_bond,
        settings.gbf_angle,
        n_conv_edge=settings.n_conv_edge,
        n_conv_angle=settings.n_conv_angle,
        h_fea_edge=settings.h_fea_edge,
        h_fea_angle=settings.h_fea_angle,
        n_classification=settings.n_classification,
        neigh=settings.neighbors,
        pooling=settings.pooling,
        embedding=True,
        num_MLP=settings.num_MLP,
    )
elif current_model == "SAGE":
    print("GraphSAGE chosen")
    from model import mySAGE
    net = mySAGE(
        settings.gbf_bond,
        settings.gbf_angle,
        n_conv_edge=settings.n_conv_edge,
        n_conv_angle=settings.n_conv_angle,
        h_fea_edge=settings.h_fea_edge,
        h_fea_angle=settings.h_fea_angle,
        n_classification=settings.n_classification,
        neigh=settings.neighbors,
        pool=settings.pooling,
        embedding=True,
        dropout_prob=settings.dropout_prob,
        num_MLP=settings.num_MLP,
    )
elif current_model == "RGCN":
    print("RGCN chosen")
    from model import myRGCN
    net = myRGCN(
        settings.gbf_bond,
        settings.gbf_angle,
        n_classification=settings.n_classification,
        neigh=settings.neighbors,
        pool=settings.pooling,
        embedding=True,
    )

net.to(device)
# print(f"net.device = {net.device}")

best_checkpoint = torch.load(
    checkpoint_best, map_location=torch.device(device)
)


net.load_state_dict(best_checkpoint['model'])
weight_dim1 = 0
weight_dim2 = 0
# Print the model's state dictionary (state_dict)
# for param_tensor in net.state_dict():
#     if param_tensor == 'out.weight':
#         print(f"param_tensor = {param_tensor}, size = {net.state_dict()[param_tensor].size()}")
#         weight_dim1, weight_dim2 = net.state_dict()[param_tensor].shape
#         print(f"weight_dim1 = {weight_dim1}")
#         print(f"weight_dim2 = {weight_dim2}")
#     if param_tensor == 'out.bias':
#         print(f"param_tensor = {param_tensor}, size = {net.state_dict()[param_tensor].size()}")
for param_tensor in net.state_dict():
    print(f"param_tensor = {param_tensor}, size = {net.state_dict()[param_tensor].size()}")

# try to hard-code these params, just for this layer
my_out_weight = net.state_dict()['out.weight']
print(f"my_out_weight.shape = {my_out_weight.shape}")
my_out_bias = net.state_dict()['out.bias']
print(f"my_out_bias.shape = {my_out_bias.shape}")

my_bn_weight = net.state_dict()['bn.weight']
print(f"my_bn_weight.shape = {my_bn_weight.shape}")
my_bn_bias = net.state_dict()['bn.bias']
print(f"my_bn_bias.shape = {my_bn_bias.shape}")

# In[ ]:


# In[31]:

# make sure the model is in evaluation mode
net.eval()


predictions = {}


for i in range(graphs.size):
    outdata = graphs.collate([graphs[i]])
    label = struct_label[i]

    x, y = prepare_batch_fn(outdata, device, non_blocking=False)
    predict, embedding = net(x)

    predictions[label] = {
        "class": np.argmax(predict.cpu().detach().numpy(),axis=1).tolist(),
        "embeddings": embedding.cpu().detach().numpy().tolist(),
    }


# In[30]:

with open('predictions.json', 'w') as f:
    json.dump(predictions, f)

def pool(atom_fea, crys_idx):

    # print(crystal_atom_idx)

    summed_fea = [
        torch.mean(
            atom_fea[idx_map[0] : idx_map[1], :], dim=0, keepdim=True
        )
        for idx_map in crys_idx
    ]
    return torch.cat(summed_fea, dim=0)

print(f"graphs.size = {graphs.size}")
# open the file with the saved crys_fea
in_file = open(f"crys_fea.txt", "r")
myline = in_file.readline()
mylist = myline.split()
crys_idx = torch.tensor([[0, 0]])
# need to initialize num_atom, num_features outside if statement
num_atom = 0
num_features = 0
if mylist[0] == 'num_atom' and mylist[3] == 'num_features':
    print("read crys_fea.txt file")
    print(mylist)
    crys_idx[0][1] = int(mylist[2])
    # try increasing crys_idx[0][1] by one
    # crys_idx[0][1] += 1
    num_atom = int(mylist[2])
    num_features = int(mylist[5])
crys_fea = torch.tensor([[0.0]*num_features]*num_atom)
print(f"inside predict.py, crys_fea.shape = {crys_fea.shape}")
print(f"just read crys_fea.txt, crys_idx = {crys_idx}")
# myline = in_file.readline()
# mylist = myline.split()
for i in range(num_atom):
    data_line = in_file.readline()
    # print(data_line)
    data_list = data_line.split(' ')
    # print(len(data_list))
    # print(data_list)
    for j in range(num_features):
        # print(data_list[j])
        crys_fea[i][j] = float(data_list[j])
print(f"inside predict.py, crys_fea[500][500] = {crys_fea[500][500]}")
# print(mylist[0])
# re-do the pooling on the crys_fea we read from a file
# get data from the rest of the graphs
for my_graph_index in range(1, graphs.size, 1):
    # print(f"my_graph_index = {my_graph_index}")
    my_line = in_file.readline()
    num_atom1 = 0
    num_features1 = 0
    mylist = myline.split()
    if mylist[0] == 'num_atom' and mylist[3] == 'num_features':
        # print(f"found data for another graph")
        num_atom1 = int(mylist[2])
        num_features1 = int(mylist[5])
        if num_features != num_features1:
            print("ERROR, this will not work, everyone must have same number of features")
        else:
            print("OK, everyone has the same number of features")
        crys_fea1 = torch.tensor([[0.0]*num_features1]*num_atom1)
        crys_idx[0][1] += num_atom1
        # need to fill in crys_fea1
        for i in range(num_atom1):
            data_line = in_file.readline()
            data_list = data_line.split(' ')
            for j in range(num_features1):
                crys_fea1[i][j] = float(data_list[j])
        crys_fea = torch.cat([crys_fea, crys_fea1], dim=0)
print(f"new crys_idx = {crys_idx}")
print(f"crys_fea.shape = {crys_fea.shape}")
in_file.close()
# debug file to see if crys_fea get corrupted
# debug_file = open("debug_crys_fea.txt", "w")
# dim1, dim2 = crys_fea.shape
# print(f"dim1 = {dim1}, dim2 = {dim2}")
# for i in range(dim1):
#     for j in range(dim2):
#         debug_file.write(str(crys_fea[i][j].item()))
#         debug_file.write(' ')
#     debug_file.write("\n")
# debug_file.close()
# cyrs_fea  are still the same after reading
crys_fea = crys_fea.to(device)
crys_idx = crys_idx.to(device)
# print(f"crys_fea.device = {crys_fea.device}")
# print(f"crys_idx.device = {crys_idx.device}")
# crys_fea_pooled = pool(crys_fea, crys_idx)
# print(f"crys_fea_pooled.device = {crys_fea_pooled.device}")
# dim1, dim2 = crys_fea_pooled.shape
# print(f"dim1 = {dim1}, dim2 = {dim2}")
# debug_file = open("debug_pooled_crys_fea.txt", "w")
# for i in range(dim1):
#     for j in range(dim2):
#         debug_file.write(str(crys_fea_pooled[i][j].item()))
#         debug_file.write(' ')
#     debug_file.write("\n")
# debug_file.close()
# crys_fea are different after pooling
# for idx_map in crys_idx:
#     print(f"idx_map[0] = {idx_map[0]}")
#     print(f"idx_map[1] = {idx_map[1]}")
# print(f"after reading crys_fea, crys_fea_pooled.shape = {crys_fea_pooled.shape}")
# # print(f"crys_fea_pooled = {crys_fea_pooled}")

# # need to do layer-norm manually with pre-determined weights and bias that we grabbed earlier
# # 1- intialize
# bn = nn.LayerNorm(crys_fea_pooled.shape)
# # 2 - grab the assigned weights and biases
# with torch.no_grad():
#     bn.weight.copy_(my_bn_weight)
#     bn.bias.copy_(my_bn_bias)

# # freeze the parameters, may not be needed because we are not in training anyways
# bn.weight.requires_grad = False
# bn.bias.requires_grad = False
# bn.weight = bn.weight.to(device)
# print(f"bn.weight.device = {bn.weight.device}")
# bn.bias = bn.bias.to(device)
# print(f"bn.bias.device = {bn.bias.device}")
# # bn = bn.to(device)
# # print(f"bn.device = {bn.device}")
# crys_fea_bn = bn(crys_fea_pooled)

# conv_to_fc_softplus = nn.Softplus()
# crys_fea_softplus = conv_to_fc_softplus(crys_fea_bn)
# print(f"after softplus: crys_fea_softplus.size() = {crys_fea_softplus.size()}")
# # print(f"crys_fea_softplus = {crys_fea_softplus}")

# my_output = torch.matmul(crys_fea_softplus, torch.transpose(my_out_weight, 0, 1)) + my_out_bias
# print(f"my_output.size() = {my_output.size()}")

# predictions_v2 = {}

# predictions_v2[0] = {
#     "class": np.argmax(my_output.cpu().detach().numpy(),axis=1).tolist(),
#     "embeddings": crys_fea_softplus.cpu().detach().numpy().tolist(),
# }

# with open('predictions_v2.json', 'w') as f:
#     json.dump(predictions_v2, f)

# try writing everything as a new model
class GANN_crys_fea(nn.Module):
    """
    takes crys_fea crystal features and runs the rest of the steps from CEGANN
    """
    def __init__(
        self,
        bn_weight,
        bn_bias,
        output_weight,
        output_bias,
        n_classification,
        my_num_fea,
    ):

        super(GANN_crys_fea, self).__init__()

        self.bn = nn.LayerNorm(my_num_fea)
        self.conv_to_fc_softplus = nn.Softplus()
        self.out = nn.Linear(my_num_fea, n_classification)
    
    def forward(self, data_crys_fea, data_crys_idx):
        data_crys_fea = pool(data_crys_fea, data_crys_idx)
        data_crys_fea = self.bn(data_crys_fea)
        data_crys_fea = self.conv_to_fc_softplus(data_crys_fea)
        my_output = self.out(data_crys_fea)
        return my_output, data_crys_fea 

second_net = GANN_crys_fea(
    bn_weight=my_bn_weight,
    bn_bias = my_bn_bias,
    output_weight=my_out_weight,
    output_bias=my_out_bias,
    n_classification=settings.n_classification,
    my_num_fea=num_features,
)

second_net.to(device)
# print(f"second_net.device = {second_net.device}")

with torch.no_grad():
    second_net.bn.weight.copy_(my_bn_weight)
    second_net.bn.bias.copy_(my_bn_bias)
    second_net.out.weight.copy_(my_out_weight)
    second_net.out.bias.copy_(my_out_bias)

dummy_output, dummy_embedding = second_net(crys_fea, crys_idx)
print(f"dummy_output.shape = {dummy_output.shape}")
print(f"dummy_embedding.shape = {dummy_embedding.shape}")

predictions_v3 = {}

predictions_v3[0] = {
    "class": np.argmax(dummy_output.cpu().detach().numpy(),axis=1).tolist(),
    "embeddings": dummy_embedding.cpu().detach().numpy().tolist(),
    }

with open('predictions_v3.json', 'w') as f:
    json.dump(predictions_v3, f)