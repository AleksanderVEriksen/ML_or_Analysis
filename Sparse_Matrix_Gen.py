# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 00:34:07 2024

@author: eriks
"""

from scipy.sparse import rand
import numpy as np
import os
import unittest
# Params to update
mat_sizes = [100, 800, 1000, 2000]
is_sparse = 1
# Low means more zeroes
density = 0.1

def generate_matrices(mat_sizes, is_sparse, density):

    if is_sparse == 1:
        sparse = True 
    else:
        sparse = False
    
    
    created_matrixes = []
    for x in mat_sizes:
        name = "Matrix"
        
        if sparse:
            name = "SparseMatrix"
        
        print(" ")
        print(f'Creating {name} with {x} rows and cols')
        
        if sparse:
            matrix = rand(x, x, density=density, format="csr", random_state=42)
            # Endre verdier direkte i sparse format
        else:
            matrix = np.random.normal(0,1,(x,x))
        matrix.data = np.where(matrix.data < 0, np.abs(matrix.data), matrix.data)
        matrix.data = np.where((0 < matrix.data) & (matrix.data < 1), matrix.data * 100, matrix.data)
        matrix = matrix.astype(int, copy=False)  # Endre type uten å kopiere data

        created_matrixes.append(matrix)
        print("Done")
        
    return created_matrixes

"""    
    #direct = "./lab5/matrix-matrix-mul/Matrixes/"  
    #file = f'{name}{x}.txt'
    #filepath = os.path.join(direct, file)
    g = f'{x+1}\n'   # Kilde: Tobias
    print(g)
    print("Writing to file")
    print(" ")
    for x in range (0, len(matrix)):
        s = ""
        for j in range (0, len(matrix[x])):
            s += ''.join(str(matrix[x][j]))
            s += ' '
        s +="\n"
        g += ''.join(str(x) for x in s)
     
    with open(filepath, "w") as f: 
        f.write(g)"""      

class Testcase(unittest.TestCase):

    def test_mat_dens(self):
        true_dens = 0.1
        mats = generate_matrices([10], 1, true_dens)
        zeroes = 0
        nums = 0
        for l in mats:
            for mat in l:
                for num in mat:
                    if num == 0:
                        zeroes+=1
                    else:
                        nums+=1
        dens = nums/(zeroes+nums)
        self.assertAlmostEqual(dens, true_dens)
    
    def test_mat_rows_cols(self):
        true_row_col = 10
        mats = generate_matrices([true_row_col], 1, 0.1)
        col = len(mats[0][0])
        row = len(mats[0])
        self.assertEqual(col, true_row_col)
        self.assertEqual(row, true_row_col)

    def test_mat_noSparse(self):
        mats = generate_matrices([8], 0, 0.1)
        nums = 0
        zeroes=0
        for l in mats:
            for mat in l:
                for num in mat:
                    nums+=1
                    if num == 0:
                        zeroes+=1
                    else:
                        nums+=1
        dens = nums/(zeroes+nums)
        self.assertTrue(0.5<dens)

if __name__ == "__main__":
    #unittest.main()
    generate_matrices([10], 1, 0.1)
    # Test reasoning used
    