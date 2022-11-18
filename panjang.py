import cv2
import os
import numpy as np

def add_matrix(matrix1, matrix2):
    return np.add(matrix1, matrix2)

def divide_matrix(matrix, divisor):
    return np.divide(matrix, divisor)

def subtract_matrix(matrix1, matrix2):
    return np.subtract(matrix1, matrix2)

def merge_matrix(matrix1, matrix2):
    return np.concatenate((matrix1, matrix2), axis=1)

def transpose_matrix(matrix):
    return np.transpose(matrix)

def multiply_matrix(matrix1, matrix2):
    return np.matmul(matrix1, matrix2)

def displayMat(mat):
    for i in range(len(mat)):
        print(mat[i])

def createzero(n):
    mat = [[0 for i in range(n)] for j in range(n)]
    return mat

def avg_matrix(dataset):
    matrix = createzero(len(dataset[0][0]))
    for i in range(len(dataset[0])): 
        matrix = add_matrix(matrix, dataset[0][i])
    
    matrix = divide_matrix(matrix, len(dataset[0]))
    return matrix.round()

def nilai_tengah(matrix, listt):
    for i in range(len(dataset[0])):
        listt.append((subtract_matrix(matrix,dataset[0][i])))
    return listt

def get_covarian(list_selisih):
    covarian = [[1 for j in range(256)] for i in range(256)]
    temp = []
    temp = createzero(256)
    for i in range(len(list_selisih)):
        temp = merge_matrix(temp, list_selisih[i])
    covarian = multiply_matrix((temp), transpose_matrix(temp))
    return covarian
def getQR(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((m, n))
    for j in range(n):
        v = A[:, j]
        norm = np.linalg.norm(v)
        for i in range(j):
            temp = A[:, j] 
            q = Q[:, i]
            qt = np.transpose(q)
            temp1 = np.dot(qt, temp)
            temp2 = np.multiply(temp1, q)
            R[i][j] = temp1
            v = np.subtract(v, temp2)
        if j == 0:
            R[j][j] = norm
            Q[:, j] = np.divide(v, norm)
        else :
            norm = np.linalg.norm(v)
            R[j][j] = round(norm, 6)
            v = np.divide(v, norm)
            Q[:, j] = v

    return Q, R

def getEigenValues(A):
    Ak = np.copy(A)
    n = A.shape[0]
    QQ = np.eye(n)
    for k in range(10):
        s = Ak.item(n - 1, n - 1)
        smult = s * np.eye(n)
        Q, R = getQR(np.subtract(Ak, smult))
        Ak = np.add(R @ Q, smult)
        QQ = QQ @ Q

    return Ak, QQ

def getEigenVal(matrix,listt):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if(i==j):
                listt.append(matrix[i][j])
                break
    return listt


count = 0
dataset = []
list_subfolders_with_paths = [f.path for f in os.scandir('dataset') if f.is_dir()]
for j in (list_subfolders_with_paths):
    listt = []
    print(j)
    for i in os.listdir(j):
        image = cv2.imread(j+'/'+str(i), 0)
        resized = cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)
        flat = resized.flatten('F')
        listt.append(flat)
    count += 1
    dataset.append(listt)
    if count == 3:
        break
    
# mat = [[1,-2,1,-3],[1,3,-1,2],[2,1,-2,3],[1,2,2,1]]
mat = dataset[0]

print(mat[0])
print("\n")

total = np.sum(mat, axis=0)
mean = total/len(mat)
print(mean)


for i in range(len(mat)):
    mat[i] = mat[i] - mean
# print(mat[0])
print(len(mat))
print(len(mat[0]))
# cv2.imshow('mean', mat[2])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# mat = np.transpose(mat)
print(len(mat))

cov = multiply_matrix(mat,np.transpose(mat))/len(mat)
print(len(cov[0]))
print(len(cov))
print(cov)

eigenVector = getEigenValues(cov)[1]
print(eigenVector, "=========================")
eigenVector = eigenVector.resize(65536, 213)
# eigenVector = np.linalg.eig(cov)[1]
print("\n")
print(eigenVector)
# cv2.imshow('eigen', eigenVector)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
        

listt = []
normm = []
for i in range(len(eigenVector[0])):
    temp2 = eigenVector[:,i].multiply(mat[:,i])
    # norm = np.linalg.norm(eigenVector[:,i])
    # print(norm)
    # temp = (1/norm)*eigenVector[:,i]
    listt.append(temp2)
    # normm.append(norm)

print(listt[0])    
# weight = []
# for i in range(len(listt)):
#     temp = (1/normm[i])*eigenVector[:,i]



# print(len(eigenVector[0]))
cv2.imshow('eigen', listt[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
        