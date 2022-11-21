import cv2
import os
import numpy as np
import time

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
    matrix = createzero(len(dataset[0]))
    for i in range(len(dataset)): 
        matrix = add_matrix(matrix, dataset[i])
    
    matrix = divide_matrix(matrix, len(dataset))
    return matrix

def nilai_tengah(matrix, listt,dataset):
    for i in range(len(dataset)):
        temp = ((subtract_matrix(dataset[i],matrix)))
        # temp = (temp.astype(np.uint8))
        listt.append(temp)
    return ((listt))

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
    for k in range(5):
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

def getNormMat(matrix):
    listt = createzero(len(matrix))
    sum = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            sum+= matrix[i][j]**2
        
    return sum**0.5
    
    


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
start = time.time()
newIMG = cv2.imread('adrianalima.jpg', 0)
newIMG = cv2.resize(newIMG, (256,256), interpolation = cv2.INTER_AREA)

count = 0
dataset = []
imagee = []
list_subfolders_with_paths = [f.path for f in os.scandir('dataset') if f.is_dir()]
for j in (list_subfolders_with_paths):
    listt = []
    temp_img = []
    print(j)
    for i in os.listdir(j):
        image = cv2.imread(j+'/'+str(i), 0)
        img2 = cv2.imread(j+'/'+str(i))
        resized = cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)
        listt.append(resized)
        temp_img.append(img2)
    count += 1
    imagee.append(temp_img)
    dataset.append(listt)
    # if j == "dataset\pins_Katherine Langford":
    #     break
    if count == 4:
        break
# dataset = np.array(dataset)
    
# print(len(dataset[0]))

# print(dataset[0][0])
#rata-rata
eigenface = []
eigen_vector_list = []
mean_list = []

for data in range(len(dataset)):
    print(data + 1)     
    matrix = avg_matrix(dataset[data])
    # matrix = matrix.astype(np.uint8)
    mean_list.append(matrix)
    # displayMat(matrix)
    # cv2.imshow('avg image',matrix)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


#cari nilai tengah
    hasil_selisih = []
    hasil_selisih = nilai_tengah(matrix,hasil_selisih,dataset[data])
# cv2.imshow("dddd",hasil_selisih[67])
    # cv2.imshow('selisih',hasil_selisih[5])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# for i in range(len(hasil_selisih)):
#     print(hasil_selisih[i])

    # print("====================================")
    temp = hasil_selisih[0]
    # temp = createzero(256)
    for i in range(1, len(hasil_selisih)):
        temp = merge_matrix(temp, hasil_selisih[i])
    # print("==================================")
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
    # print("============================================================")
    #eigen_vector = np.linalg.eig(covarian)[1]
    # print(eigen_vector)


    # for i in range(len(eigen_vector)):
    #     norm = np.linalg.norm(eigen_vector[:,i])
    #     temp = (1/norm)*eigen_vector[:,i]
    #     listt.append(temp)
    listt =[]
    for i in range(len(hasil_selisih)):
        # temp = []
        temp3 = multiply_matrix(eigen_vector, hasil_selisih[i])
        # cv2.imshow('Original image'+str(i),temp3)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        listt.append(temp3)
        
    eigenface.append(listt)
    eigen_vector_list.append(eigen_vector)

# print((hasil_selisih[0]))




# displayMat(listt)
# print(listt[:,0])
# print(len(listt))
# print(len(listt[0]))
# print(len(listt[0][0]))
# for i in range(5):
#     cv2.imshow('Original image'+str(i),listt[i])

# cv2.waitKey(0)    
# cv2.destroyAllWindows()


# newIMG = cv2.imread('gal-gadot.jpeg', 0)
# newIMG = cv2.resize(newIMG, (256,256), interpolation = cv2.INTER_AREA)




hasil_euclidian = []
list_min = []
for vector in range(len(eigen_vector_list)):
    print(vector+1)
    testIMG = subtract_matrix(newIMG, mean_list[vector])
    newtst = multiply_matrix(eigen_vector_list[vector], testIMG)
    min = 0
    idx_min = 0
    # for j in range(len(eigenface)):
    euiclidian = []
    # temp_min = []
    for i in range(len(eigenface[vector])):
        temp = subtract_matrix(newtst, eigenface[vector][i])
        temp = np.linalg.norm(temp)
        euiclidian.append(temp)
        # temp_min.append(temp)
        if i == 0:
            min = temp
            idx_min = 0
        elif temp <= min:
            min = temp
            idx_min = i
        # if(temp<=100):
        #     print("ada")
        #     print(i)
        #     break
        # minnnnnnnn = min(temp_min)
        # print(temp_min)
        # print("====================================")
    list_min.append((min, idx_min))
    hasil_euclidian.append(euiclidian)
 
# hasil_euclidian = np.array(hasil_euclidian)   
# minn = np.unravel_index(np.argmin(hasil_euclidian, axis=None), hasil_euclidian.shape)
print(list_min)
idx_min = 0
final_min = list_min[0]
for i in range(len(list_min)):
    if list_min[i][0] <= final_min[0]:
        final_min = list_min[i]
        idx_min = i
       
# index = hasil_euclidian.index(minn)
img = imagee[idx_min][final_min[1]]


print(time.time()-start)

cv2.imshow('Original image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()