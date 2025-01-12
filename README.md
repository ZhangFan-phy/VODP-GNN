# GNN for mixed VODPs (Introduction)
The transformer-inspired graph neural network(GNN) can accurantely predict the bandgaps and formation energy of mixed vacancy ordered double perovskites (VODPs), which can directly map unrelaxed structures to physical properties. The unrelaxed supercells only include the compositions and the particular arrangement of elements in the mixed system. This scheme can allow us to skip both the costly structural relaxation and property evalution steps.

![image](https://github.com/user-attachments/assets/4f6d9552-aa15-4239-9cf5-6086a369fa3d)


# Installation
Our model was implemented with the Pytorch and PyG (PyTorch Geometric) ML frameworks.

# Model training
The main script to train model is `model.py` in `GNN model` folder. A user needs the following information to train a model: 1) `input_AMX.pt` with the atomic information containing electron shell configuration, electronegativity, ionic radius and the element site. 2) `mask_edge.pt` with edge incex. 3) `mask_coord.pt` with the distance vector. 4) `gap.pt` with the DFT bandgap. 5) `formation.pt` with the DFT formation energy. 

We provide an example to predict the DFT band gap of mixed VODP with 144-atoms supercell in `GNN model/examples` folder using the trained model. 
