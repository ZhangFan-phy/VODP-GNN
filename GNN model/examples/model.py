# uncompyle6 version 3.9.1
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.15 (default, Nov  4 2022, 20:59:55) 
# [GCC 11.2.0]
# Embedded file name: /media/weiwei/disk2/GNN/VODP/model.py
# Compiled at: 2024-06-16 14:54:09
# Size of source mod 2**32: 12401 bytes
import h5py, numpy as np, torch
from torch import nn
import sys, os
from torch_geometric.nn import MessagePassing
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import pool
from time import perf_counter
from torch_geometric.utils import unbatch
from torch_geometric.data import Data
import matplotlib.pyplot as plt

def CellShell3D(n):
    if (n > 10):
        # complain:
        print("Number of unit cells to check is ", (2 * n) ** 3)
        print("Could be slow to run.")
    x = np.arange(-n, n)
    y = np.arange(-n, n)
    z = np.arange(-n, n)
    # print(x,y,z)
    count = 0
    shell = []
    # Layer 1: 2^3-0^3 = 8 cells
    # Layer 2: 4^3-2^3 = 56 cells
    # Layer 3: 6^3-4^3 = 152 cells
    # Layer 4: 8^3-6^3 = 296 cells
    # ...
    # Layer n: 2n^3-(2n-1)^3 = (2ii+2)^3-(2ii)^3 = 8*(3*ii^2+3*ii+1)
    for ii in range(n):
      # nsh = 8*(3*(ii+1)**2-3*(ii+1)+1) # number of cells in each shell
      nsh = 8 * (3 * ii ** 2 + 3 * ii + 1)  # number of cells in each shell
      shell.append(np.zeros((nsh, 3)))
    shell_count = np.zeros(n, dtype=np.int32)
    for xc in x:
      for yc in y:
        for zc in z:
          for ii in range(n):
            if ((not (xc in range(-ii, ii) and yc in range(-ii, ii) and zc in range(-ii, ii)))
                  and (xc in range(-ii - 1, ii + 1) and yc in range(-ii - 1, ii + 1) and zc in range(-ii - 1,
                                                                                                     ii + 1))):
              shell[ii][shell_count[ii], 0:3] = np.array([xc, yc, zc])
              shell_count[ii] += 1
              # print(shell_count)
              break
    for ish in range(n):
      # print(shell_count[ish], len(shell[ish]))
      assert (shell_count[ish] == len(shell[ish]))
    return shell_count, shell

def find_invariant_basis(latvec):
    max_nshell = 3
    basis = np.zeros((3,3))
    # latvec is a 3x3 ndarray
    assert (isinstance(latvec, np.ndarray))
    assert (np.shape(latvec) == (3,3))
    shell_count, shell = CellShell3D(max_nshell)
    crysR = np.concatenate(shell, axis=0)
    # print("crysR: \n", crysR, ". ")
    allR = np.array(crysR) @ latvec 
    Rnorm = np.linalg.norm(allR, axis=1)    
    srtidx = np.argsort(Rnorm)
    # print(allR[srtidx])
    nR = len(srtidx)
    if (Rnorm[0] < 1e-8):
        basis[0] = allR[srtidx[0]]
        istart = 1
    else:
        basis[0] = allR[srtidx[1]] 
        istart = 2
    # print(istart, nR)
    for ii in range(istart, nR):
        cos_theta = np.dot( allR[srtidx[ii]], basis[0] )/np.linalg.norm(basis[0])/Rnorm[srtidx[ii]]
        # print(ii, " cos_theta ", cos_theta)
        if (np.isclose(cos_theta, 1.0) or np.isclose(cos_theta, -1.0)):
            continue
        else:
            basis[1] = allR[srtidx[ii]]
            if (cos_theta > 0.0):
                basis[1] = -basis[1]
            break
    # print(ii, " basis[0] x basis[1] ", np.cross(basis[0], basis[1]))
    for jj in range(ii+1, nR):
        vol = np.dot( allR[srtidx[jj]], np.cross(basis[0], basis[1]) )
        # print(" jj ", jj, allR[srtidx[jj]], vol)
        if (np.abs(vol) < 1e-8):
            continue
        else:
            if(vol > 0):
                basis[2] = allR[srtidx[jj]]
            else:
                basis[2] = -allR[srtidx[jj]]
            break 
    return basis 

class AtomLayer(MessagePassing):
    def __init__(self, m_atom_in, m_atom_out, m_edge, device=torch.device("cpu")):
        super().__init__(aggr="mean")
        self.m_atom_in = m_atom_in
        self.m_atom_out = m_atom_out
        self.m_msg = 32
        self.sqrt_mmsg = self.m_msg**0.5
        self.m_hidden = 32
        self.m_edge = m_edge
        self.device = device
        self.LnQ = nn.Linear(m_atom_in, self.m_msg, device=self.device)
        self.LnK = nn.Linear(m_atom_in, m_atom_in, device=self.device)
        self.LnV = nn.Linear(m_atom_in, m_atom_in, device=self.device)
        self.LnE = nn.Linear(m_edge, m_edge, device=self.device)
        self.sigma = nn.Sequential(
            nn.Linear(m_atom_in + m_edge, self.m_hidden), 
            nn.LeakyReLU(), 
            nn.Linear(self.m_hidden, self.m_msg), 
            nn.LeakyReLU()
        ).to(self.device)
        self.phi_msg = nn.Sequential(
            nn.Linear(m_atom_in * 2 + m_edge, self.m_hidden), 
            nn.LeakyReLU(), 
            nn.Linear(self.m_hidden, self.m_msg), 
            nn.LeakyReLU()
        ).to(self.device)
        self.phi_out = nn.Sequential(
            nn.Linear(self.m_msg + m_atom_in, self.m_hidden), 
            nn.LeakyReLU(), 
            nn.Linear(self.m_hidden, m_atom_out)
        ).to(self.device)
        self.alpha = nn.Sequential(
            nn.Linear(6 + 2 * m_atom_in, self.m_hidden), 
            nn.LeakyReLU(), 
            nn.Linear(self.m_hidden, 1)
        ).to(self.device)
        self.beta = nn.Sequential(
            nn.Linear(6 + 2 * m_atom_in, self.m_hidden), 
            nn.LeakyReLU(), 
            nn.Linear(self.m_hidden, 1)
        ).to(self.device)

    def forward(self, atomx, edge_index, edge_attr, chi, edge_vec):
        cmsg = self.propagate(
          edge_index = edge_index, 
          atomx = atomx,
          chi = chi,
          edge_attr = edge_attr,
          edge_vec = edge_vec)
        return self.phi_out(torch.cat([cmsg[:, :self.m_msg], atomx], dim=1)), cmsg[:, self.m_msg:]

    def message(self, atomx_j, atomx_i, chi_j, chi_i, edge_attr, edge_vec):
        # print(edge_attr.shape, atomx_j.shape)
        w = torch.sigmoid(self.LnQ(atomx_i) * self.sigma(torch.cat((self.LnE(edge_attr), self.LnK(atomx_j)), dim=1)) / self.sqrt_mmsg)
        msg = self.phi_msg(torch.cat([self.LnE(edge_attr), self.LnV(atomx_i), self.LnV(atomx_j)], dim=1))
        tempv = torch.cat((
         (edge_vec * chi_i).sum(dim=1, keepdim=True),
         (edge_vec * chi_j).sum(dim=1, keepdim=True),
         (chi_i * chi_j).sum(dim=1, keepdim=True),
         torch.norm(edge_vec, dim=1, keepdim=True),
         torch.norm(chi_i, dim=1, keepdim=True),
         torch.norm(chi_j, dim=1, keepdim=True),
         atomx_j, atomx_i),
          dim=1)
        a = self.alpha(tempv)
        b = torch.sigmoid(self.beta(tempv))
        return torch.cat([w * msg, a * edge_vec + b * chi_j], dim=1)


class eGNN(torch.nn.Module):

    def __init__(self, atom_dim=18, edge_dim=21, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.hidden = 64
        self.AtomLayer1 = AtomLayer(m_atom_in=atom_dim,
          m_atom_out=atom_dim, m_edge=edge_dim,
          device=(self.device))
        self.AtomLayer2 = AtomLayer(m_atom_in=atom_dim,
          m_atom_out=atom_dim, m_edge=edge_dim,
          device=(self.device))
        self.AtomLayer3 = AtomLayer(m_atom_in=atom_dim,
          m_atom_out=atom_dim, m_edge=edge_dim,
          device=(self.device))
        self.phi1 = nn.Sequential(
            nn.Linear(atom_dim, self.hidden), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden, self.hidden), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden,1)
        ).to(self.device)
        self.phi2 = nn.Sequential(
            nn.Linear(atom_dim, self.hidden), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden, self.hidden), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden,1)
        ).to(self.device)
        self.phi3 = nn.Sequential(
            nn.Linear(atom_dim, self.hidden), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden, self.hidden), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden,1)
        ).to(self.device)
        self.psi1 = nn.Sequential(
            nn.Linear(atom_dim, self.hidden), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden, self.hidden), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden,1)
        ).to(self.device)
        self.psi2 = nn.Sequential(
            nn.Linear(atom_dim, self.hidden), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden, self.hidden), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden,1)
        ).to(self.device)
        self.psi3 = nn.Sequential(
            nn.Linear(atom_dim, self.hidden), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden, self.hidden), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden,1)
        ).to(self.device)
        self.mlp = nn.Sequential(
            nn.Linear( 9 + 3 * atom_dim, 2 * self.hidden), 
            nn.LeakyReLU(), nn.Linear(2 * self.hidden, self.hidden), 
            nn.LeakyReLU(), nn.Linear(self.hidden, self.hidden // 2), 
            nn.LeakyReLU(), nn.Linear(self.hidden // 2, 1)
        ).to(self.device)
        self.reduce = nn.Sequential(
            nn.Linear(7, 16), nn.SiLU(), nn.Linear(16, 1)
        ).to(self.device)

    def forward(self, material_graph):
        atom_dim = material_graph.x.shape[1]
        n_atom = material_graph.num_nodes
        if type(material_graph) == type(Batch()):
            n_graphs = material_graph.num_graphs
        else:
            n_graphs = 1
        material_graph.chi = torch.zeros((n_atom, 3), device=(self.device))
        material_graph.x, material_graph.chi = \
            self.AtomLayer1( 
                material_graph.x, material_graph.edge_index, material_graph.edge_attr, 
                material_graph.chi, material_graph.edge_vec )
        ft = pool.global_mean_pool(material_graph.x, material_graph.batch)
        ch = pool.global_mean_pool(material_graph.chi, material_graph.batch)
        atmax1  = pool.global_max_pool (self.phi1(material_graph.x), material_graph.batch)
        atmean1 = pool.global_mean_pool (self.psi2(material_graph.x), material_graph.batch)

        material_graph.x, material_graph.chi = \
            self.AtomLayer2(
                material_graph.x, material_graph.edge_index, material_graph.edge_attr, 
                material_graph.chi, material_graph.edge_vec )
        ft = torch.cat((
          ft, pool.global_mean_pool(material_graph.x, material_graph.batch)),
          dim=1)
        ch = torch.cat((
          ch, pool.global_mean_pool(material_graph.chi, material_graph.batch)),
          dim=1)
        atmax2  = pool.global_max_pool (self.phi2(material_graph.x), material_graph.batch)
        atmean2 = pool.global_mean_pool (self.psi2(material_graph.x), material_graph.batch)

        material_graph.x, material_graph.chi = \
            self.AtomLayer3(
                material_graph.x, material_graph.edge_index, material_graph.edge_attr, 
                material_graph.chi, material_graph.edge_vec )
        ft = torch.cat((
         ft, pool.global_mean_pool(material_graph.x, material_graph.batch)),
          dim=1)
        ch = torch.cat((
          ch, pool.global_mean_pool(material_graph.chi, material_graph.batch)),
          dim=1).view(n_graphs, -1, 3)
        atmax3  = pool.global_max_pool (self.phi3(material_graph.x), material_graph.batch)
        atmean3 = pool.global_mean_pool (self.psi3(material_graph.x), material_graph.batch)

        tempv = torch.bmm(ch, torch.transpose(ch, 1, 2)).reshape((-1, 9))
        atsum = self.mlp(torch.cat((tempv, ft), dim=1))
        y_pred = self.reduce(torch.cat((atmax1, atmax2, atmax3, atmean1, atmean2, atmean3, atsum), dim=1))
        return y_pred

def evaluate_loss(net, data_iter, loss_fn, device=torch.device("cpu"), verbose=False, plot=False):
    """Evaluate the loss for a given dataset"""
    net.eval()
    cumulate_loss = 0
    ncount = 0
    #Ytrue = []
    Ypred = []
    for mat_graph in data_iter:
        mat_graph = mat_graph.to(device)
        y_hat = net(mat_graph)
        #loss = loss_fn(y_hat, y)
        #cumulate_loss += float(loss)
        ncount += 1
        #Ytrue += list(y.to("cpu").detach().numpy())
        Ypred += list(y_hat.to("cpu").detach().numpy())
    else:
        if verbose:
            for t in Ypred:
                print(" %10.5f " %  t.item())
        if plot:
            plt.scatter(Ytrue, Ypred)
            plt.savefig("Pred_vs_True.pdf", dpi=100)
        return cumulate_loss / ncount

def train_epoch(net, train_iter, loss_fn, optimizer, device=torch.device("cpu"), verbose=False):
    ibatch = 0
    net.train()
    metric = [
     0.0, 0.0]
    t0 = perf_counter()
    for mat_graphs, y in train_iter:
        mat_graphs = mat_graphs.to(device)
        ibatch += 1
        if verbose:
            if ibatch < 4:
                tstart = perf_counter()
        y_hat = net(mat_graphs)
        loss = loss_fn(y_hat, y)
        if isinstance(optimizer, torch.optim.Optimizer):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metric[0] += float(loss.sum())
        metric[1] += 1
        if verbose:
            if ibatch < 4:
                tend = perf_counter()
                sys.stdout.write(" ## Training batch {} takes {:.4f} s; ".format(ibatch, tend - tstart))
            t1 = perf_counter()
            if verbose:
                print("Training a epoch takes {:.4f} s.".format(t1 - t0))
        return metric[0] / metric[1]

def train_model(net, mat_graphs, label, num_epochs=1200, bs=200, lr=0.06, print_every_nEpoch=5, device=torch.device("cpu")):
    loss_fn_mse = nn.MSELoss()
    loss_fn_mae = nn.L1Loss()
    nsamp = len(mat_graphs)
    batch_size = min(bs, nsamp)
    ntrain = int(nsamp * 0.9)
    train_iter = DataLoader((tuple(zip(mat_graphs[:ntrain], label[:ntrain]))),
      batch_size,
      shuffle=True)
    test_iter = DataLoader((tuple(zip(mat_graphs[ntrain:], label[ntrain:]))),
      batch_size,
      shuffle=True)
    optimizer = torch.optim.Adam((net.parameters()), lr=lr)
    print(" n_epoch   train_RMSE     validation_RMSE   train_MAE  validation_MAE ")
    for epoch in range(num_epochs):
        train_epoch(net, train_iter, loss_fn_mse, optimizer, device, False)
        train_loss_mse = evaluate_loss(net, train_iter, loss_fn_mse, device)
        train_loss_mae = evaluate_loss(net, train_iter, loss_fn_mae, device)
        #if epoch == 0 or (epoch + 1) % print_every_nEpoch == 0:
        if epoch == 1999:
            verbose = True
        else:
            verbose = False
        test_loss_mse = evaluate_loss(net, test_iter, loss_fn_mse, device, verbose)
        test_loss_mae = evaluate_loss(net, test_iter, loss_fn_mae, device, verbose)
        if epoch == 0 or (epoch + 1) % print_every_nEpoch == 0:
            print("%8d %12.6f %12.6f %12.6f %12.6f " % (epoch,
             train_loss_mse**0.5, test_loss_mse**0.5, train_loss_mae, test_loss_mae))

if __name__ == "__main__":
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    atom_dim = 19
    datadir = "/media/weiwei/disk2/GNN/VODP/Data"
    edge_index = torch.load(datadir + "/mask_edge.pt")
    edge_vec = torch.load(datadir + "/mask_coord.pt")
    atomx = torch.load(datadir + "/input_AMX.pt")
    lattvec = torch.load(datadir + "/lattice.pt")
    eform = list(torch.load(datadir + "/formation.pt"))
    egap = list(torch.load(datadir + "/gap.pt"))
    nmat = len(edge_index)
    dataset = []
    for imat in range(nmat):
        matgraph = Data()
        matgraph.edge_index = torch.transpose(edge_index[imat].to(dev), 0, 1)
        matgraph.edge_vec = (edge_vec[imat] / torch.norm((edge_vec[imat]), dim=1, keepdim=True)).to(torch.float32).to(dev)
        matgraph.x = atomx[imat].to(dev)
        dataset.append(matgraph)
    print("Test the eGNN model")
    model = eGNN(atom_dim=atom_dim,
      device=dev)
    train_iter = DataLoader((tuple(zip(dataset, egap))), 20, shuffle=False)
    for X, Y in train_iter:
        Y_hat = model(X)
        break
    for yp, y in zip(Y_hat, Y):
        print(" %10.5f %10.5f " % (yp.item(), y.item()))

