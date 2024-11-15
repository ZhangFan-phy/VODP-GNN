import numpy as np
import sys

def read_poscar(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines

def write_poscar(file_path, lines):
    with open(file_path, 'w') as f:
        f.writelines(lines)

def generate_new_poscar(poscar_lines, a1, m1, m2, m3, m4, m5, m6, x1, x2):
    element_line_idx = 5  
    count_line_idx = 6    
    pos_line_idx = 8      
    
    elements = poscar_lines[element_line_idx].split()
    counts = list(map(int, poscar_lines[count_line_idx].split()))
    
    A_index = elements.index("A")
    M_index = elements.index("M")
    X_index = elements.index("X")
    A_count = counts[A_index]
    M_count = counts[M_index]
    X_count = counts[X_index]
    #print(A_index, M_count, X_count) 
    Cs_count, Rb_count = a1, A_count-a1
    Ti_count, Zr_count, Hf_count, Pd_count, Pt_count, Sn_count, Te_count = m1, m2, m3, m4, m5, m6, M_count-m1-m2-m3-m4-m5-m6
    Cl_count, Br_count, I_count = x1, x2, X_count-x1-x2
    #print(Cs_count, Rb_count) 
    elements[A_index] = f"Cs Rb"
    elements[M_index] = f"Ti Zr Hf Pd Pt Sn Te"
    elements[X_index] = f"Cl Br I"
    counts[0:1] = [Cs_count, Rb_count]
    counts[2:8] = [Ti_count, Zr_count, Hf_count, Pd_count, Pt_count, Sn_count, Te_count]
    counts[9:11] = [Cl_count, Br_count, I_count]

    poscar_lines[element_line_idx] = ' '.join(elements) + '\n'
    poscar_lines[count_line_idx] = ' '.join(map(str, counts)) + '\n'
    A_positions = poscar_lines[pos_line_idx:pos_line_idx+A_count]
    M_positions = poscar_lines[pos_line_idx+32:pos_line_idx+32+M_count]
    X_positions = poscar_lines[pos_line_idx+48:pos_line_idx+48+X_count]
    np.random.shuffle(A_positions)
    np.random.shuffle(M_positions)
    np.random.shuffle(X_positions)
    poscar_lines[pos_line_idx:pos_line_idx+A_count] = A_positions
    poscar_lines[pos_line_idx+32:pos_line_idx+32+M_count] = M_positions
    poscar_lines[pos_line_idx+48:pos_line_idx+48+X_count] = X_positions
    return poscar_lines

#a1 = 3
#m1, m2, m3, m4, m5, m6 = 0, 4, 0, 0, 0, 0
#x1, x2 = 5, 6
#poscar_lines = read_poscar("./POSCAR")
#
#for i in range(1, 21):
#    new_poscar_lines = generate_new_poscar(poscar_lines.copy(), a1, m1, m2, m3, m4, m5, m6, x1, x2)  
#    output_path = f"./new_POSCAR_{i}"
#    write_poscar(output_path, new_poscar_lines)

def main():
    if len(sys.argv) != 10:
        print("Usage: python run_prediction.py a1 m1 m2 m3 m4 m5 m6 x1 x2")
        sys.exit(1)

    a1, m1, m2, m3, m4, m5, m6, x1, x2 = map(int, sys.argv[1:])
    poscar_lines = read_poscar("./POSCAR")

    # 生成20个新结构
    for i in range(1, 21):
        new_poscar_lines = generate_new_poscar(
            poscar_lines.copy(), a1, m1, m2, m3, m4, m5, m6, x1, x2
        )
        output_path = f"./new_POSCAR_{i}"
        write_poscar(output_path, new_poscar_lines)

if __name__ == "__main__":
    main()
