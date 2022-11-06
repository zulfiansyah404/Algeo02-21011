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
    return np.dot(matrix1, matrix2)

# mat = [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
# matt = [[0, 1, 0] ,[0, 0, 0], [ 1, 0, 1]]

# x = merge_matrix(mat, matt)
# print((x))

path = 'dataset/pins_Zendaya'
count = 0
dataset = []
list_subfolders_with_paths = [f.path for f in os.scandir('dataset') if f.is_dir()]
for j in (list_subfolders_with_paths):
    listt = []
    print(j)
    for i in os.listdir(j):
        # print(i)
        image = cv2.imread(j+'/'+str(i), 0)
        # count += 1
        # print(image)
        # # print(i)
        # listt.append(image)
    # print(image)
        resized = cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)
        # cv2.imshow(str(count),resized)
        listt.append(resized)
    count += 1
    if count == 3:
        break
        
        #cv2.waitKey(0)
    dataset.append(listt)
    
print(len(dataset[0]))
matrix = [[0 for j in range(256)] for i in range(256)]

# print(len(matrix))
# print(len(dataset[0][212]))
for i in range(len(dataset[0])): 
    matrix = add_matrix(matrix, dataset[0][i])
        
matrix = divide_matrix(matrix, len(dataset[0]))
# print(matrix)

hasil_selisih = []

for i in range(len(dataset[0])):
    hasil_selisih.append(subtract_matrix(matrix,dataset[0][i]))
    
print(len(hasil_selisih[0]))

for i in range(len(hasil_selisih)-1):
    hasil_selisih[0] = merge_matrix(hasil_selisih[0], hasil_selisih[i+1])
    
print(len(hasil_selisih[0]))
#print(len(transpose_matrix(hasil_selisih[0])))

covarian = [[1 for j in range(256)] for i in range(256)]

covarian = multiply_matrix((hasil_selisih[0]), transpose_matrix(hasil_selisih[0]))

print(len(covarian[0]))
    
    
    
cv2.destroyAllWindows()
cv2.waitKey(0)
cv2.destroyAllWindows()