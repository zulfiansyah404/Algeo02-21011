from sympy import *
import numpy as np
import cv2
import time


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
    for k in range(20):
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


start = time.time()
# temp = np.array([[1,2,3,4,5], [4,5,6,7,8], [7,8,9,10,11], [10, 11, 12,13,14], [15,16,17,18,19]], dtype='float')
# temp = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]], dtype='float')
# temp = np.array([[-2,-4,2],[-2,1,2],[4,2,5]],dtype='float')
temp = cv2.imread('dataset/pins_Cristiano Ronaldo/Cristiano Ronaldo75_1363.jpg',0)
temp = cv2.resize(temp, (256,256))

print(getQR(temp)[0])
print("\n")
eigen = (getEigenValues(temp))
print(eigen[0])
print("\n")
eigenVal =[]
eigenVal = getEigenVal(eigen[0],eigenVal)
print(eigenVal)
print("\n")
print((eigen[1]))



print(time.time()-start)

cv2.imshow('Original image',eigen[1])

cv2.waitKey(0)
cv2.destroyAllWindows()