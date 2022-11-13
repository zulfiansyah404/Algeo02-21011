from sympy import *
import numpy as np

lamda = symbols('λ')

def merge_matrix(matrix1, matrix2):
    return np.concatenate((matrix1, matrix2), axis=1)

def creatIdentiti(n):
    mat = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        mat[i][i] = lamda
    return mat

def displayMat(mat):
    for i in range(len(mat)):
        print(mat[i])
        
def subtract_matrix(matrix1, matrix2):
    return (np.subtract(matrix1, matrix2))

def determinant(matrix):
    return matrix.det()

def createxxx(n, listt):
    for i in range(n):
        listt.append(symbols('x'+str(i+1)))
    return listt
    
def createzero(n):
    mat = [[0 for i in range(n)] for j in range(n)]
    return mat

def multiply_matrix(matrix1, matrix2):
    return np.dot(matrix1, matrix2)

def solveEigenVector(matrix, listt, xxx):
    #kali matrix dengan x
    kali = multiply_matrix(matrix, xxx)
    count = 0
    for i in range(len(matrix)):
        # kalo hasilnya 0 gausah dimasukin ke list
        if(kali[i]==0):
            count += 1
            continue
        else:
            listt.append(kali[i])
            count+=1
    return listt

def transpose_matrix(matrix):
    return np.transpose(matrix)

def isCollZero(listt):
    count = 0
    for i in range(len(listt)):
        if(listt[i]==0):
            count+=1
    if(count == len(listt)):
        return True
    else:
        return False
    
def convertToMatrix(matrix,row,col):
    new_mat = []
    count = 0
    for i in range(row):
        temp = []
        for j in range(col):
            temp.append(matrix[count])
            count+=1
        new_mat.append(temp)
    return new_mat
            
# mat = Matrix([[3,0],[8,-1]])
# mat = [[3,0],[8,-1]]
# mat = [[1,3],[3,1]]
mat = [[3,-2,0],[-2,3,0],[0,0,5]]
# mat = [[3,1],[0,1]]
# mat = [[1,4],[-4,-7]]
# mat = [[-6,3],[4,5]]
        
#bikin matrix identitas kali lamda
lam = creatIdentiti(len(mat))
#bikin matrix isinya 0 semua
zero = createzero(len(mat))
displayMat(lam)
print("")
#ngurangin matrix identitas kali lamda sama matrix yang diinput
hasil = subtract_matrix(lam,mat)

displayMat(hasil)

hasil1 = Matrix(hasil)

#itung determinan
det = determinant(hasil1)

print("\nPersamaan :\n"+ str(det) +" = 0")


print("")
#nyrai lamda
factor = solve(det)
#ngurutin lamda
factor.sort(reverse=True)
#ketemu hasil lamda
print("Eigen value =",factor)
print("")

#list buat nyimpen hasil matrix yang lamdanya udah diganti pake eigen value
udah_dikurang_lamda = []
for m in range(len(factor)):
    #bikin matrix kosong buat copy matrix yang udah diinput
    mattttt = createzero(len(mat))
    count=0
    #bikin matrix constant biar nilainya gak berubah
    const_matrix = Matrix(hasil)
    for i in range(len(mat)):
        for j in range(len(mat)):
            #copy matrix yang udah diinput ke matrix kosongan
            mattttt[i][j] = (hasil[i][j])
            #yang ada lamdanya pasti diagonal utama
            if(i==j):
                #subtitusi lamda sama eigen value
                temp = const_matrix[count]
                udah_diganti_lamdanya = temp.subs(lamda,factor[m])
                mattttt[i][j] = udah_diganti_lamdanya
            count+=1
    udah_dikurang_lamda.append(mattttt)    
        
for i in range(len(udah_dikurang_lamda)):
    print("dimasukkan lamda =",factor[i])
    displayMat(udah_dikurang_lamda[i])
    print("")

tt = createxxx(len(udah_dikurang_lamda[0]),[])

hampir_eigen_vector = []
for i in range(len(udah_dikurang_lamda)):
    xx = solveEigenVector(udah_dikurang_lamda[i],[],tt)
    hampir_eigen_vector.append(xx)

displayMat(hampir_eigen_vector)

# print("")
# for i in range(len(hampir_eigen_vector)):
#     for j in range(len(hampir_eigen_vector[i])):
#         print(hampir_eigen_vector[i][j],"= 0")
#         # print(eigen_vector[i][j].subs(tt[j],1))
#         # print(eigen_vector[i][j].coeff(tt[j]))
#         print(solve(hampir_eigen_vector[i][j]))

# print("")
# opop = Matrix(udah_dikurang_lamda[1]).eigenvects(simplify=True)
# print(opop)
# print("")
# # print(udah_dikurang_lamda[0])

# A = np.array([[2, 2, 0], [2, 2, 0], [0, 0, 0]])
# # A = np.array([[4, 3, 2], [-2, 2, 3], [3, -5, 2]])
# B = np.array([0, 0, 0])
# print(B)
# X2 = np.linalg.solve(A, B)
# print(X2)
print("")
# x,y,z = symbols('x y z')
C = (np.float64(udah_dikurang_lamda[0]))
B = [[0 for i in range(1)] for j in range(len(C))]
ref = []
for i in range(len(udah_dikurang_lamda)):
    temp = merge_matrix((udah_dikurang_lamda[i]),B)
    
    ref.append(temp)
    
C = (np.float64(udah_dikurang_lamda[0]))
B = [[0 for i in range(1)] for j in range(len(C))]
X6 = np.linalg.svd(C, B)[2]


# print(X6)
print("")

eigen_vector =[]
for i in range(len(X6)):
    temp = []
    for j in range(len(X6[i])):
        if(X6[i][j] != 0):
            if(X6[i][j] < 0):
                temp.append((X6[i][j]/-X6[i][j]))
            else:
                temp.append((X6[i][j]/X6[i][j]))
        else:
            temp.append((X6[i][j]))
    eigen_vector.append(temp)


eigen_vector = transpose_matrix(eigen_vector)
print("vector eigen 1:")
displayMat(eigen_vector)
print("")

eigen_vector2 = []
for i in range(len(ref)):
    temp1 = Matrix(ref[i])
    temppp = (temp1.rref()[0])
    # print(temppp)
    temp2 = convertToMatrix(temppp,len(ref[i]),len(ref[i][0]))
    # displayMat(temp2)
    for j in range(len(temp2)):
        if(isCollZero(temp2[j])==False):
            l =  temp2[j][:-1]
            eigen_vector2.append(l)

    
print("vector eigen 2 :")
# displayMat(eigen_vector2)
op = transpose_matrix(eigen_vector2)
displayMat(op)
