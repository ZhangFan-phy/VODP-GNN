import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import math
import copy
import os
from itertools import repeat
from torch.nn.functional import pad
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp.inputs import Poscar
import torch.nn.functional as F

def append_unique(lst, num):
    if num not in lst:
        lst.append(num)

# Dictionary of element symbols to electronic structures
element_electron_structures = {
    'Cs': [2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 0, 0, 1],
    'Rb': [2, 2, 6, 2, 6, 10, 2, 6, 0, 1, 0, 0, 0, 0],
    'Ti': [2, 2, 6, 2, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0],
    'Zr': [2, 2, 6, 2, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0],
    'Pd': [2, 2, 6, 2, 6, 10, 2, 6, 10, 0, 0, 0, 0, 0],
    'Sn': [2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 2, 0, 0, 0],
    'Te': [2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 4, 0, 0, 0],
    'Hf': [2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 2, 2],
    'Pt': [2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 9, 1],
    'Cl': [2, 2, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Br': [2, 2, 6, 2, 6, 10, 2, 5, 0, 0, 0, 0, 0, 0],
    'I': [2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 5, 0, 0, 0] }

#electronegativity dictionary
element_electronegativity = {
    'Cs': [0.659], 'Rb': [0.706], 'Ti': [1.38], 'Zr': [1.32], 'Pd': [1.58],
    'Sn': [1.824], 'Te': [2.158], 'Hf': [1.16], 'Pt': [1.72], 'Cl': [2.869],
    'Br': [2.685], 'I': [2.359] }

#ionic radius
element_ionic_radius={'Cs': [1.67], 'Rb': [1.52], 'Ti': [0.605], 'Zr': [0.72], 'Pd': [0.615],
    'Sn': [0.69], 'Te': [0.52], 'Hf': [0.58], 'Pt': [0.625], 'Cl': [1.81],
    'Br': [1.96], 'I': [2.2]}

first_group_elements = set(['Cs', 'Rb'])
second_group_elements = set(['Ti', 'Zr', 'Pd', 'Sn', 'Te', 'Hf', 'Pt'])
third_group_elements = set(['Cl', 'Br', 'I'])

#with open('./new_perb-all.txt','r') as file:
#    chemical_formulas = [line.split()[0] for line in file]


input_AMX = []
mask_list = []
coord_list = []
lattice_list = []

for i in range(1,21):
    structure_path = os.path.join("./new_POSCAR_"+ str(i))
    poscar = Poscar.from_file(structure_path)
    structure = poscar.structure
    lattice = structure.lattice.matrix
    lattice_list.append(lattice)
    composition = structure.composition.get_el_amt_dict()
    
    element_list1 = ["Cs", "Rb"]
    element_list2 = ["Ti", "Zr", "Pd", "Sn", "Te", "Hf", "Pt"]
    element_list3 = ["Cl", "Br", "I"]
    element_list = element_list1 + element_list2
    
    element_counts = structure.composition.as_dict()
    total_count = sum(element_counts[element] for element in element_list)

    neighbor_list = []
    vector_list = []
    for idx, site in enumerate(structure):
        ele = site.specie.symbol
        frac_coords1 = site.frac_coords
        #print(frac_coords1)

        if ele in element_list1:
            neighbors = structure.get_neighbors(site, r=4.2, include_index=True)            
            AX = []
            for neighbor in neighbors:
                neighbor_index = neighbor[2]
                image = neighbor[3]
                frac_coords = structure[neighbor_index].frac_coords
                unwrapped_frac_coords = frac_coords + image
                vector = unwrapped_frac_coords - frac_coords1
                cart_vector = np.dot(vector, lattice)
                vector_list.append(cart_vector)
                AX.append(neighbor_index)
            ele_iter = repeat(idx)
            AX_zip= zip(ele_iter,AX)            
            neighbor_list.extend(AX for AX in AX_zip)
        
        elif ele in element_list2:
            neighbors = structure.get_neighbors(site, 3.0, include_index=True)
            BX = []
            for neighbor in neighbors:
                neighbor_index = neighbor[2]
                image = neighbor[3]
                frac_coords = structure[neighbor_index].frac_coords
                unwrapped_frac_coords = frac_coords + image
                vector = unwrapped_frac_coords - frac_coords1
                cart_vector = np.dot(vector, lattice)
                vector_list.append(cart_vector)
                BX.append(neighbor_index)
            ele_iter = repeat(idx)
            BX_zip = zip(ele_iter,BX)
            neighbor_list.extend(BX for BX in BX_zip)
        
        elif ele in element_list3:
            neighbors = structure.get_neighbors(site, r=4.2, include_index = True)
            XAB = []
            for neighbor in neighbors:
                neighbor_site = neighbor[0]
                neighbor_index = neighbor[2]
                image = neighbor[3]
                frac_coords = structure[neighbor_index].frac_coords
                element = neighbor_site.specie.symbol
                if element not in element_list3:
                    unwrapped_frac_coords = frac_coords + image
                    vector = unwrapped_frac_coords - frac_coords1
                    cart_vector = np.dot(vector, lattice)
                    vector_list.append(cart_vector)
                    XAB.append(neighbor_index)
            ele_iter = repeat(idx)
            XAB_zip = zip(ele_iter,XAB)
            neighbor_list.extend(XAB for XAB in XAB_zip)
    mask_list.append(neighbor_list)
    coord_list.append(vector_list)

            
    vectors = []
    for site in structure:
        ele = site.specie
        electronic_structure = element_electron_structures.get(ele.symbol)
        electronegativity = element_electronegativity.get(ele.symbol)
        ionic_radius = element_ionic_radius.get(ele.symbol)        
        
        group_encoding = [0, 0, 0]
        if ele.symbol in first_group_elements:
            group_encoding[0] = 1
        elif ele.symbol in second_group_elements:
            group_encoding[1] = 1
        elif ele.symbol in third_group_elements:
            group_encoding[2] = 1
        
        vector = electronic_structure+electronegativity + ionic_radius + group_encoding
        vectors.append(vector)
    input_AMX.append(vectors)
tensor_input = [torch.tensor(sublist) for sublist in input_AMX]

tensor_mask = [torch.tensor(sublist) for sublist in mask_list]
#print(tensor_mask)
tensor_coord = [torch.tensor(sublist) for sublist in coord_list]
#print(tensor_coord)
#print(lattice_list)
tensor_lattice = [torch.tensor(sublist) for sublist in lattice_list]

#band_gaps=[]
#with open('./new_perb-all.txt','r') as file:
#    band_gap= [float(line.split()[1]) for line in file]
#    band_gaps.append(band_gap)
#band_gaps_tensor = torch.tensor(band_gaps).view(-1,1)

    
#formations = []
#with open('./new_perb-all.txt','r') as file:
#    formation= [float(line.split()[2]) for line in file]
#    formations.append(formation)
#formations_tensor = torch.tensor(formations).view(-1,1)


torch.save(tensor_input, 'input_AMX.pt')
#torch.save(X_test, 'X_test.pt')
#torch.save(band_gaps_tensor, 'gap.pt')
#torch.save(gap_test, 'gap_test.pt')
#torch.save(formations_tensor, 'formation.pt')
#torch.save(formation_test, 'formation_test.pt')
torch.save(tensor_mask, 'mask_edge.pt')
torch.save(tensor_coord,'mask_coord.pt')
torch.save(tensor_lattice,'lattice.pt')
#torch.save(mask_test,'mask_test.pt')
