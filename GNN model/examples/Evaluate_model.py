import torch
from model import eGNN, train_model
from pathlib import Path
import sys
from torch_geometric.loader import DataLoader
from model import evaluate_loss, find_invariant_basis
from torch import nn
from torch_geometric.data import Data

#dataset_filename = "dataset.pt"
atom_dim = 19 
edge_dim = 21
dev = torch.device('cuda:0') 
# dev = torch.device('cpu')

restart = False
modelfile = 'model.pt'
if (len(sys.argv) > 1):
   modelfile = sys.argv[1]
print("# Restart from "+modelfile)
net = torch.load(modelfile)

# formation.pt  gap.pt  input_AMX.pt 
# lattice.pt  mask_coord.pt  mask_edge.pt
#datadir = "/media/weiwei/disk2/GNN/VODP/Data"
datadir = "./"
edge_index = torch.load(datadir+"/mask_edge.pt")
edge_vec   = torch.load(datadir+"/mask_coord.pt")
atomx      = torch.load("./input_AMX.pt")
#eform = list(torch.load(datadir+"/formation.pt").to(dev))
#egap  = list(torch.load(datadir+"/gap.pt").to(dev))
nmat = len(edge_index)
dataset = []
for imat in range(nmat):
   matgraph = Data() 
   matgraph.edge_index = torch.transpose( edge_index[imat].to(dev), 0, 1 )
   matgraph.edge_vec   = edge_vec[imat].to(torch.float32).to(dev)
   matgraph.x = atomx[imat].to(dev)
   rc = torch.linspace(0.0, 5.0, edge_dim).to(dev)
   nedge = len(matgraph.edge_vec)
   edgenorm = torch.norm(matgraph.edge_vec, dim=1, keepdim=False)
   # (nedge, nc)
   edge_RBF = torch.exp( -(edgenorm.repeat(edge_dim,1).T - rc)**2 / 0.1 )
   matgraph.edge_attr  = edge_RBF
   #print(matgraph.num_nodes, matgraph.num_edges, matgraph.edge_attr.shape)
   dataset.append(matgraph)

batchsize = min(500, nmat)
dloader = DataLoader( \
  tuple(dataset), \
  batchsize, shuffle=False)

loss_fn = nn.MSELoss()
err = evaluate_loss(net, dloader, loss_fn, dev, verbose=True, plot=False)
#print((err)**0.5)

