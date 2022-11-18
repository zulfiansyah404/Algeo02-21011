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
    return matrix

def nilai_tengah(matrix, listt,dataset):
    for i in range(len(dataset[0])):
        listt.append((subtract_matrix(dataset[0][i],matrix)))
    return listt

def get_covarian(list_selisih):
    # covarian = [[1 for j in range(256)] for i in range(256)]
    temp = list_selisih[0]
    for i in range(len(list_selisih) - 1):
        temp = merge_matrix(temp, list_selisih[i+1])
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
    for k in range(100):
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


# m1 = [[2,0,1],[1,2,0],[0,2,4]]
# m2 = [[1,1,1],[0,1,0],[1,2,2]]
# data = [[m1,m2]]
# print(len(data))
# print(len(data[0]))
# print(len(data[0][0]))


# mean = avg_matrix(data)
# displayMat(mean)

# hasil_selisih = []
# hasil_selisih = nilai_tengah(mean,hasil_selisih,data)

# for i in range(len(hasil_selisih)):
#     print("\n")
#     displayMat(hasil_selisih[i])
    
# temp = hasil_selisih[0]
# for i in range(len(hasil_selisih) - 1):
#     temp = merge_matrix(temp, hasil_selisih[i+1])

# print(temp)
# print("\n")

# cov = np.cov(temp)
# displayMat(cov)


# mat = [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
# matt = [[0, 1, 0] ,[0, 0, 0], [ 1, 0, 1]]
# mattt = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# x = merge_matrix(m,mat)
# # x = merge_matrix(mat, matt)
# # x = merge_matrix(x, mattt)
# print((x))

# path = 'dataset/pins_Zendaya'
count = 0
dataset = []
list_subfolders_with_paths = [f.path for f in os.scandir('dataset') if f.is_dir()]
for j in (list_subfolders_with_paths):
    listt = []
    print(j)
    for i in os.listdir(j):
        image = cv2.imread(j+'/'+str(i), 0)
        resized = cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)
        listt.append(resized)
    count += 1
    if count == 3:
        break
    dataset.append(listt)
    
print(len(dataset[0]))

print(dataset[0][0])
#rata-rata       
matrix = avg_matrix(dataset)
# displayMat(matrix)


#cari nilai tengah
hasil_selisih = []
hasil_selisih = nilai_tengah(matrix,hasil_selisih,dataset)
# cv2.imshow("dddd",hasil_selisih[67])
cv2.imshow('Original image',hasil_selisih[200])
cv2.waitKey(0)
cv2.destroyAllWindows()

# for i in range(len(hasil_selisih)):
#     print(hasil_selisih[i])

print("====================================")
temp = hasil_selisih[0]
# temp = createzero(256)
for i in range(1, len(hasil_selisih)):
    temp = merge_matrix(temp, hasil_selisih[i])
print("==================================")
# cv2.imshow('Original image',temp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
    


# # # temp = [[1,0,0,0,1,0],[1,1,0,0,0,0],[0,0,1,1,0,1]]

# # #cari kovarian
##covarian = get_covarian(hasil_selisih)
# # # print(covarian)
# # print("==================================")
# temp = (np.cov(temp))
covarian = np.matmul(temp, np.transpose(temp))

# cv2.imshow('Original image',covarian)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(len(covarian))
# print(len(covarian[0]))


# print(len(covarian[0]))

eigen = (getEigenValues(covarian))
# # print(eigen[0])
# print("\n")
# eigenVal =[]
# eigenVal = getEigenVal(eigen[0],eigenVal)  

eigen_vector = eigen[1]
# print(eigen_vector)
print("============================================================")
#eigen_vector = np.linalg.eig(covarian)[1]
# print(eigen_vector)


# for i in range(len(eigen_vector)):
#     norm = np.linalg.norm(eigen_vector[:,i])
#     temp = (1/norm)*eigen_vector[:,i]
#     listt.append(temp)
#listt =[]
for i in range(len(hasil_selisih)):
    temp3 = multiply_matrix(eigen[1], hasil_selisih[i])
    cv2.imshow('Original image'+str(i),temp3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #listt.append(temp)

# print((hasil_selisih[0]))


# displayMat(listt)
# print(listt[:,0])
# print(len(listt))
# print(len(listt[0]))
# print(len(listt[0][0]))
for i in range(5):
    cv2.imshow('Original image'+str(i),listt[i])

cv2.waitKey(0)
cv2.destroyAllWindows()