import cv2
import os
import numpy as np
import time

def getNormMat(matrix):
    sum = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            sum+= matrix[i][j]**2
    return sum**0.5

def getNormVec(array):
    sum = 0
    for i in range(len(array)):
        sum += array[i] ** 2

    return sum ** 0.5

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
        listt.append(temp)
    return ((listt))

def get_covarian(list_selisih):
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
        norm = getNormVec(v)
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
            norm = getNormVec(v)
            R[j][j] = round(norm, 6)
            v = np.divide(v, norm)
            Q[:, j] = v

    return Q, R

def getEigenValues(A):
    Ak = np.copy(A)
    n = A.shape[0]
    QQ = np.eye(n)
    for k in range(1):
        s = Ak.item(n - 1, n - 1)
        smult = s * np.eye(n)
        Q, R = getQR(np.subtract(Ak, smult))
        Ak = np.add(R @ Q, smult)
        QQ = QQ @ Q

    return Ak, QQ

def answer(datasets,filename):
    start = time.time()
    newIMG = cv2.imread(filename, 0)
    newIMG = cv2.resize(newIMG, (256,256), interpolation = cv2.INTER_AREA)

    count = 0
    dataset = []
    imagee = []
    nama_folder = []
    list_subfolders_with_paths = [f.path for f in os.scandir(datasets) if f.is_dir()]
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
        nama_folder.append(j)
    eigenface = []
    eigen_vector_list = []
    mean_list = []

    # mencari nilai rata rata matriks

    for data in range(len(dataset)):
        print(data + 1)     
        matrix = avg_matrix(dataset[data])
        mean_list.append(matrix)

    # mencari nilai tengah

        hasil_selisih = []
        hasil_selisih = nilai_tengah(matrix,hasil_selisih,dataset[data])

    # menggabungkan matriks

        temp = hasil_selisih[0]
        for i in range(1, len(hasil_selisih)):
            temp = merge_matrix(temp, hasil_selisih[i])
        
        
    # mencari matriks covarian
        covarian = np.matmul(temp, np.transpose(temp))

    # mencari vector eigen
        eigen = (getEigenValues(covarian))
        eigen_vector = eigen[1]

    # mencari nilai eigenface
        listt =[]
        for i in range(len(hasil_selisih)):
            temp3 = multiply_matrix(eigen_vector, hasil_selisih[i])
            listt.append(temp3)
            
        eigenface.append(listt)
        eigen_vector_list.append(eigen_vector)


    # mencari nilai selisih euclidean dan indeks gambar
    hasil_euclidian = []
    list_min = []
    for vector in range(len(eigen_vector_list)):
        print(vector+1)
        testIMG = subtract_matrix(newIMG, mean_list[vector])
        newtst = multiply_matrix(eigen_vector_list[vector], testIMG)
        min = 0
        idx_min = 0
        euiclidian = []
        for i in range(len(eigenface[vector])):
            temp = subtract_matrix(newtst, eigenface[vector][i])
            temp = getNormMat(temp)
            euiclidian.append(temp)
            if i == 0:
                min = temp
                idx_min = 0
            elif temp <= min:
                min = temp
                idx_min = i
        list_min.append((min, idx_min))
        hasil_euclidian.append(euiclidian)
    
    # mencari selisih euclidian terkecil
    idx_min = 0
    final_min = list_min[0]
    for i in range(len(list_min)):
        if list_min[i][0] <= final_min[0]:
            final_min = list_min[i]
            idx_min = i

    # mencari gambar dengan nilai euclidian terkecil
    img = imagee[idx_min][final_min[1]]

    # return gambar, nama folder, serta kemiripan
    return [img, nama_folder[idx_min], list_min[idx_min][0]]
