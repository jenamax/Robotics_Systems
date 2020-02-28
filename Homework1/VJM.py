import numpy as np
from numpy import cos, sin, arctan2, arccos, pi, block, zeros
from funcs import *
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

L = 1

k0 = 10**6

E_1 = 350 * 10**9
G_1 = 100 * 10**9
d_1 = 0.5
L_1 = L
L_2 = L

S_1 = pi * d_1**2 / 4
Iy_1 = pi * d_1**4 / 64
Iz_1 = pi * d_1**4 / 64
J_1 = Iy_1 + Iz_1

k1 = np.matrix([[E_1*S_1/L_1, 0, 0, 0, 0, 0], \
[0, 12*E_1*Iz_1/L_1**3, 0, 0, 0, 6*E_1*Iy_1/L_1**2], \
[0, 0, 12*E_1*Iy_1/L_1**3, 0, -6*E_1*Iy_1/L_1**2, 0], \
[0, 0, 0, G_1*J_1/L_1, 0, 0], \
[0, 0, -6*E_1*Iy_1/L_1**2, 0, 4*E_1*Iy_1/L_1, 0], \
[0, 6*E_1*Iy_1/L_1**2, 0, 0, 0, 4*E_1*Iz_1/L_1]])

k2 = np.matrix([[E_1*S_1/L_1, 0, 0, 0, 0, 0], \
[0, 12*E_1*Iz_1/L_1**3, 0, 0, 0, 6*E_1*Iy_1/L_1**2], \
[0, 0, 12*E_1*Iy_1/L_1**3, 0, -6*E_1*Iy_1/L_1**2, 0], \
[0, 0, 0, G_1*J_1/L_1, 0, 0], \
[0, 0, -6*E_1*Iy_1/L_1**2, 0, 4*E_1*Iy_1/L_1, 0], \
[0, 6*E_1*Iy_1/L_1**2, 0, 0, 0, 4*E_1*Iz_1/L_1]])

k3 = np.matrix([[E_1*S_1/L_1, 0, 0, 0, 0, 0], \
[0, 12*E_1*Iz_1/L_1**3, 0, 0, 0, 6*E_1*Iy_1/L_1**2], \
[0, 0, 12*E_1*Iy_1/L_1**3, 0, -6*E_1*Iy_1/L_1**2, 0], \
[0, 0, 0, G_1*J_1/L_1, 0, 0], \
[0, 0, -6*E_1*Iy_1/L_1**2, 0, 4*E_1*Iy_1/L_1, 0], \
[0, 6*E_1*Iy_1/L_1**2, 0, 0, 0, 4*E_1*Iz_1/L_1]])

Kt = inv(block([[k0, zeros([1, 6]), zeros([1, 6]), zeros([1, 6])], [zeros([6, 1]), k1, zeros([6, 6]), zeros([6, 6])], [zeros([6, 1]), zeros([6, 6]), k2, zeros([6, 6])], [zeros([6, 1]), zeros([6, 6]), zeros([6, 6]), k3]])) * 10**16


def ik(x, y, z):
    q1 = z
    q3 = arccos((x**2 + y**2 - L_1**2 - L_2**2) / (2 * L_1 * L_2))
    q2 = arctan2(y, x) - arctan2(L_2 * sin(q3), L_1 + L_2 * cos(q3))
    q4 = -q2 - q3
    return q1, q2, q3, q4

def dk1(q1, q2, q3, q4, th1, th2, th3, th4):
    return Tb * Tz(q1) * Tz(th1) * T6d(th2) * Rz(q2) * T6d(th3) * Tx(L_1) \
            * Rz(q3) * T6d(th4) * Tx(L_2) * Rz(q4)

def dk2(q1, q2, q3, q4, th1, th2, th3, th4):
    return Tb * Tx(q1) * Tx(th1) * T6d(th2) * Rx(q2) * T6d(th3) * Tz(L_1) \
            * Rx(q3) * T6d(th4) * Tz(L_2) * Rx(q4)

def dk3(q1, q2, q3, q4, th1, th2, th3, th4):
    return Tb * Ty(q1) * Ty(th1) * T6d(th2) * Ry(q2) * T6d(th3) * Tx(L_1) \
            * Ry(q3) * T6d(th4) * Tx(L_2) * Ry(q4)

def J_theta1(q1, q2, q3, q4):

    J2 = Tb * Tz(q1) * Tz(th1) * dTz() * T6d(th2) * Rz(q2) * Tx(L_1) * T6d(th3) *  \
        Rz(q3) * Tx(L_2) * T6d(th4) * Rz(q4)

    J3 = Tb * Tz(q1) * Tz(th1) * Tx(th2[0]) * dTx() * Ty(th2[1]) * Tz(th2[2]) \
        * Rx(th2[3]) * Ry(th2[4]) * Rz(th2[5]) * Rz(q2) * Tx(L_1) * T6d(th3) * \
         Rz(q3) * Tx(L_2) * T6d(th4) * Rz(q4)

    J4 = Tb * Tz(q1) * Tz(th1) * Tx(th2[0]) * Ty(th2[1]) * dTy() * Tz(th2[2]) \
        * Rx(th2[3]) * Ry(th2[4]) * Rz(th2[5]) * Rz(q2) * Tx(L_1) * T6d(th3) *\
         Rz(q3) * Tx(L_2) * T6d(th4) * Rz(q4)

    J5 = Tb * Tz(q1) * Tz(th1) * Tx(th2[0]) * Ty(th2[1]) * Tz(th2[2]) * dTz() \
        * Rx(th2[3]) * Ry(th2[4]) * Rz(th2[5]) * Rz(q2) * Tx(L_1) * T6d(th3) * \
        Rz(q3) * Tx(L_2) * T6d(th4) * Rz(q4)

    J6 = Tb * Tz(q1) * Tz(th1) * Tx(th2[0]) * Ty(th2[1]) * Tz(th2[2]) * Rx(th2[3]) \
         * dRx() * Ry(th2[4]) * Rz(th2[5]) * Rz(q2) * Tx(L_1) * T6d(th3) * Rz(q3) \
          * Tx(L_2) * T6d(th4) * Rz(q4)

    J7 = Tb * Tz(q1) * Tz(th1) * Tx(th2[0]) * Ty(th2[1]) * Tz(th2[2]) * Rx(th2[3]) \
         * Ry(th2[4]) * dRy() * Rz(th2[5]) * Rz(q2) * Tx(L_1) * T6d(th3) * Rz(q3) \
         * Tx(L_2) * T6d(th4) * Rz(q4)

    J8 = Tb * Tz(q1) * Tz(th1) * Tx(th2[0]) * Ty(th2[1]) * Tz(th2[2]) * Rx(th2[3]) \
         * Ry(th2[4]) * Rz(th2[5]) * dRz() * Rz(q2) * Tx(L_1) * T6d(th3) * Rz(q3) \
         * Tx(L_2) * T6d(th4) * Rz(q4)

    J9 = Tb * Tz(q1) * Tz(th1) * T6d(th2) * Rz(q2) * Tx(L_1) * \
        Tx(th3[0]) * dTx() * Ty(th3[1]) * Tz(th3[2]) * Rx(th3[3]) * Ry(th3[4]) * \
        Rz(th3[5]) * \
        Rz(q3) * Tx(L_2) * T6d(th4) * Rz(q4)

    J10 = Tb * Tz(q1) * Tz(th1) * T6d(th2) * Rz(q2) * Tx(L_1) * \
        Tx(th3[0]) * Ty(th3[1]) * dTy() * Tz(th3[2]) * Rx(th3[3]) * Ry(th3[4]) * \
        Rz(th3[5]) * \
        Rz(q3) * Tx(L_2) * T6d(th4) * Rz(q4)

    J11 = Tb * Tz(q1) * Tz(th1) * T6d(th2) * Rz(q2) * Tx(L_1) * \
        Tx(th3[0]) * Ty(th3[1]) * Tz(th3[2]) * dTz() * Rx(th3[3]) * Ry(th3[4]) * \
        Rz(th3[5]) * \
        Rz(q3) * Tx(L_2) * T6d(th4) * Rz(q4)

    J12 = Tb * Tz(q1) * Tz(th1) * T6d(th2) * Rz(q2) * Tx(L_1) * \
        Tx(th3[0]) * Ty(th3[1]) * Tz(th3[2]) * Rx(th3[3]) * dRx() * Ry(th3[4]) * \
        Rz(th3[5]) * \
        Rz(q3) * Tx(L_2) * T6d(th4) * Rz(q4)

    J13 = Tb * Tz(q1) * Tz(th1) * T6d(th2) * Rz(q2) * Tx(L_1) * \
        Tx(th3[0]) * Ty(th3[1]) * Tz(th3[2]) * Rx(th3[3]) * Ry(th3[4]) * dRy() * \
        Rz(th3[5]) * \
        Rz(q3) * Tx(L_2) * T6d(th4) * Rz(q4)

    J14 = Tb * Tz(q1) * Tz(th1) * T6d(th2) * Rz(q2) * Tx(L_1) * \
        Tx(th3[0]) * Ty(th3[1]) * Tz(th3[2]) * Rx(th3[3]) * Ry(th3[4]) * \
        Rz(th3[5]) * dRz() * \
        Rz(q3) * Tx(L_2) * T6d(th4) * Rz(q4)

    J15 = Tb * Tz(q1) * Tz(th1) * T6d(th2) * Rz(q2) * Tx(L_1) * T6d(th3) * Rz(q3) \
        * Tx(L_2) * Tx(th4[0]) * dTx() * Ty(th4[1]) * Tz(th4[2]) * Rx(th4[3]) * Ry(th4[4]) * \
        Rz(th4[5]) * Rz(q4)

    J16 = Tb * Tz(q1) * Tz(th1) * T6d(th2) * Rz(q2) * Tx(L_1) * T6d(th3) * Rz(q3) \
        * Tx(L_2) * Tx(th4[0]) * Ty(th4[1]) * dTy() * Tz(th4[2]) * Rx(th4[3]) * Ry(th4[4]) * \
        Rz(th4[5]) * Rz(q4)

    J17 = Tb * Tz(q1) * Tz(th1) * T6d(th2) * Rz(q2) * Tx(L_1) * T6d(th3) * Rz(q3) \
        * Tx(L_2) * Tx(th4[0]) * Ty(th4[1]) * Tz(th4[2]) * dTz() * Rx(th4[3]) * Ry(th4[4]) * \
        Rz(th4[5]) * Rz(q4)

    J18 = Tb * Tz(q1) * Tz(th1) * T6d(th2) * Rz(q2) * Tx(L_1) * T6d(th3) * Rz(q3) \
         * Tx(L_2) * Tx(th4[0]) * Ty(th4[1]) * Tz(th4[2]) * Rx(th4[3]) * dRx() * Ry(th4[4]) * \
        Rz(th4[5]) * Rz(q4)

    J19 = Tb * Tz(q1) * Tz(th1) * T6d(th2) * Rz(q2) * Tx(L_1) * T6d(th3) * Rz(q3) \
         * Tx(L_2) * Tx(th4[0]) * Ty(th4[1]) * Tz(th4[2]) * Rx(th4[3]) * Ry(th4[4]) * dRy() * \
        Rz(th4[5]) * Rz(q4)

    J20 = Tb * Tz(q1) * Tz(th1) * T6d(th2) * Rz(q2) * Tx(L_1) * T6d(th3) * Rz(q3) \
        * Tx(L_2) * Tx(th4[0]) * Ty(th4[1]) * Tz(th4[2]) * Rx(th4[3]) * Ry(th4[4]) * \
        Rz(th4[5]) * dRz() * Rz(q4)

    T = [J2, J3, J4, J5, J6, J7, J8, J9, J10, J11, J12, J13, J14, J15, \
        J16, J17, J18, J19, J20]

    J = np.matrix(zeros((6, len(T))))
    DK = dk1(q1, q2, q3, q4, th1, th2, th3, th4)
    if np.linalg.det(DK) == 0:
        print("Singularity error")
    D_inv = inv(np.matrix(DK))
    for i in range(0, len(T)):
        J[0, i] = T[i][0, 3]
        J[1, i] = T[i][1, 3]
        J[2, i] = T[i][2, 3]
        J[3, i] = (T[i] * D_inv)[2, 1]
        J[4, i] = (T[i] * D_inv)[0, 2]
        J[5, i]= (T[i] * D_inv)[1, 0]
    return J

def J_theta2(q1, q2, q3, q4):

    J2 = Tb * Tx(q1) * Tx(th1) * dTx() * T6d(th2) * Rx(q2) * Tz(L_1) * T6d(th3) *  \
        Rx(q3) * Tz(L_2) * T6d(th4) * Rx(q4)

    J3 = Tb * Tx(q1) * Tx(th1) * Tx(th2[0]) * dTx() * Ty(th2[1]) * Tz(th2[2]) \
        * Rx(th2[3]) * Ry(th2[4]) * Rz(th2[5]) * Rx(q2) * Tz(L_1) * T6d(th3) * \
         Rx(q3) * Tz(L_2) * T6d(th4) * Rx(q4)

    J4 = Tb * Tx(q1) * Tx(th1) * Tx(th2[0]) * Ty(th2[1]) * dTy() * Tz(th2[2]) \
        * Rx(th2[3]) * Ry(th2[4]) * Rz(th2[5]) * Rx(q2) * Tz(L_1) * T6d(th3) *\
         Rx(q3) * Tz(L_2) * T6d(th4) * Rx(q4)

    J5 = Tb * Tx(q1) * Tx(th1) * Tx(th2[0]) * Ty(th2[1]) * Tz(th2[2]) * dTz() \
        * Rx(th2[3]) * Ry(th2[4]) * Rz(th2[5]) * Rx(q2) * Tz(L_1) * T6d(th3) * \
        Rx(q3) * Tz(L_2) * T6d(th4) * Rx(q4)

    J6 = Tb * Tx(q1) * Tx(th1) * Tx(th2[0]) * Ty(th2[1]) * Tz(th2[2]) * Rx(th2[3]) \
         * dRx() * Ry(th2[4]) * Rz(th2[5]) * Rx(q2) * Tz(L_1) * T6d(th3) * Rx(q3) \
          * Tz(L_2) * T6d(th4) * Rx(q4)

    J7 = Tb * Tx(q1) * Tx(th1) * Tx(th2[0]) * Ty(th2[1]) * Tz(th2[2]) * Rx(th2[3]) \
         * Ry(th2[4]) * dRy() * Rz(th2[5]) * Rx(q2) * Tz(L_1) * T6d(th3) * Rx(q3) \
         * Tz(L_2) * T6d(th4) * Rx(q4)

    J8 = Tb * Tx(q1) * Tx(th1) * Tx(th2[0]) * Ty(th2[1]) * Tz(th2[2]) * Rx(th2[3]) \
         * Ry(th2[4]) * Rz(th2[5]) * dRz() * Rx(q2) * Tz(L_1) * T6d(th3) * Rx(q3) \
         * Tz(L_2) * T6d(th4) * Rx(q4)

    J9 = Tb * Tx(q1) * Tx(th1) * T6d(th2) * Rx(q2) * Tz(L_1) * \
        Tx(th3[0]) * dTx() * Ty(th3[1]) * Tz(th3[2]) * Rx(th3[3]) * Ry(th3[4]) * \
        Rz(th3[5]) * \
        Rx(q3) * Tz(L_2) * T6d(th4) * Rx(q4)

    J10 = Tb * Tx(q1) * Tx(th1) * T6d(th2) * Rx(q2) * Tz(L_1) * \
        Tx(th3[0]) * Ty(th3[1]) * dTy() * Tz(th3[2]) * Rx(th3[3]) * Ry(th3[4]) * \
        Rz(th3[5]) * \
        Rx(q3) * Tz(L_2) * T6d(th4) * Rx(q4)

    J11 = Tb * Tx(q1) * Tx(th1) * T6d(th2) * Rx(q2) * Tz(L_1) * \
        Tx(th3[0]) * Ty(th3[1]) * Tz(th3[2]) * dTz() * Rx(th3[3]) * Ry(th3[4]) * \
        Rz(th3[5]) * \
        Rx(q3) * Tz(L_2) * T6d(th4) * Rx(q4)

    J12 = Tb * Tx(q1) * Tx(th1) * T6d(th2) * Rx(q2) * Tz(L_1) * \
        Tx(th3[0]) * Ty(th3[1]) * Tz(th3[2]) * Rx(th3[3]) * dRx() * Ry(th3[4]) * \
        Rz(th3[5]) * \
        Rx(q3) * Tz(L_2) * T6d(th4) * Rx(q4)

    J13 = Tb * Tx(q1) * Tx(th1) * T6d(th2) * Rx(q2) * Tz(L_1) * \
        Tx(th3[0]) * Ty(th3[1]) * Tz(th3[2]) * Rx(th3[3]) * Ry(th3[4]) * dRy() * \
        Rz(th3[5]) * \
        Rx(q3) * Tz(L_2) * T6d(th4) * Rx(q4)

    J14 = Tb * Tx(q1) * Tx(th1) * T6d(th2) * Rx(q2) * Tz(L_1) * \
        Tx(th3[0]) * Ty(th3[1]) * Tz(th3[2]) * Rx(th3[3]) * Ry(th3[4]) * \
        Rz(th3[5]) * dRz() * \
        Rx(q3) * Tz(L_2) * T6d(th4) * Rx(q4)

    J15 = Tb * Tx(q1) * Tx(th1) * T6d(th2) * Rx(q2) * Tz(L_1) * T6d(th3) * Rx(q3) \
        * Tz(L_2) * Tx(th4[0]) * dTx() * Ty(th4[1]) * Tz(th4[2]) * Rx(th4[3]) * Ry(th4[4]) * \
        Rz(th4[5]) * Rx(q4)

    J16 = Tb * Tx(q1) * Tx(th1) * T6d(th2) * Rx(q2) * Tz(L_1) * T6d(th3) * Rx(q3) \
        * Tz(L_2) * Tx(th4[0]) * Ty(th4[1]) * dTy() * Tz(th4[2]) * Rx(th4[3]) * Ry(th4[4]) * \
        Rz(th4[5]) * Rx(q4)

    J17 = Tb * Tx(q1) * Tx(th1) * T6d(th2) * Rx(q2) * Tz(L_1) * T6d(th3) * Rx(q3) \
        * Tz(L_2) * Tx(th4[0]) * Ty(th4[1]) * Tz(th4[2]) * dTz() * Rx(th4[3]) * Ry(th4[4]) * \
        Rz(th4[5]) * Rx(q4)

    J18 = Tb * Tx(q1) * Tx(th1) * T6d(th2) * Rx(q2) * Tz(L_1) * T6d(th3) * Rx(q3) \
         * Tz(L_2) * Tx(th4[0]) * Ty(th4[1]) * Tz(th4[2]) * Rx(th4[3]) * dRx() * Ry(th4[4]) * \
        Rz(th4[5]) * Rx(q4)

    J19 = Tb * Tx(q1) * Tx(th1) * T6d(th2) * Rx(q2) * Tz(L_1) * T6d(th3) * Rx(q3) \
         * Tz(L_2) * Tx(th4[0]) * Ty(th4[1]) * Tz(th4[2]) * Rx(th4[3]) * Ry(th4[4]) * dRy() * \
        Rz(th4[5]) * Rx(q4)

    J20 = Tb * Tx(q1) * Tx(th1) * T6d(th2) * Rx(q2) * Tz(L_1) * T6d(th3) * Rx(q3) \
        * Tz(L_2) * Tx(th4[0]) * Ty(th4[1]) * Tz(th4[2]) * Rx(th4[3]) * Ry(th4[4]) * \
        Rz(th4[5]) * dRz() * Rx(q4)

    T = [J2, J3, J4, J5, J6, J7, J8, J9, J10, J11, J12, J13, J14, J15, \
        J16, J17, J18, J19, J20]

    J = np.matrix(zeros((6, len(T))))
    DK = dk2(q1, q2, q3, q4, th1, th2, th3, th4)
    if np.linalg.det(DK) == 0:
        print("Singularity error")
    D_inv = inv(np.matrix(DK))
    for i in range(0, len(T)):
        J[0, i] = T[i][0, 3]
        J[1, i] = T[i][1, 3]
        J[2, i] = T[i][2, 3]
        J[3, i] = (T[i] * D_inv)[2, 1]
        J[4, i] = (T[i] * D_inv)[0, 2]
        J[5, i]= (T[i] * D_inv)[1, 0]
    return J

def J_theta3(q1, q2, q3, q4):

    J2 = Tb * Ty(q1) * Ty(th1) * dTx() * T6d(th2) * Ry(q2) * Tx(L_1) * T6d(th3) *  \
        Rx(q3) * Tx(L_2) * T6d(th4) * Ry(q4)

    J3 = Tb * Ty(q1) * Ty(th1) * Tx(th2[0]) * dTx() * Ty(th2[1]) * Tz(th2[2]) \
        * Rx(th2[3]) * Ry(th2[4]) * Rz(th2[5]) * Ry(q2) * Tx(L_1) * T6d(th3) * \
         Ry(q3) * Tx(L_2) * T6d(th4) * Ry(q4)

    J4 = Tb * Ty(q1) * Ty(th1) * Tx(th2[0]) * Ty(th2[1]) * dTy() * Tz(th2[2]) \
        * Rx(th2[3]) * Ry(th2[4]) * Rz(th2[5]) * Ry(q2) * Tx(L_1) * T6d(th3) *\
         Ry(q3) * Tx(L_2) * T6d(th4) * Ry(q4)

    J5 = Tb * Ty(q1) * Ty(th1) * Tx(th2[0]) * Ty(th2[1]) * Tz(th2[2]) * dTz() \
        * Rx(th2[3]) * Ry(th2[4]) * Rz(th2[5]) * Ry(q2) * Tx(L_1) * T6d(th3) * \
        Ry(q3) * Tx(L_2) * T6d(th4) * Ry(q4)

    J6 = Tb * Ty(q1) * Ty(th1) * Tx(th2[0]) * Ty(th2[1]) * Tz(th2[2]) * Rx(th2[3]) \
         * dRx() * Ry(th2[4]) * Rz(th2[5]) * Ry(q2) * Tx(L_1) * T6d(th3) * Ry(q3) \
          * Tx(L_2) * T6d(th4) * Ry(q4)

    J7 = Tb * Ty(q1) * Ty(th1) * Tx(th2[0]) * Ty(th2[1]) * Tz(th2[2]) * Rx(th2[3]) \
         * Ry(th2[4]) * dRy() * Rz(th2[5]) * Ry(q2) * Tx(L_1) * T6d(th3) * Ry(q3) \
         * Tx(L_2) * T6d(th4) * Ry(q4)

    J8 = Tb * Ty(q1) * Ty(th1) * Tx(th2[0]) * Ty(th2[1]) * Tz(th2[2]) * Rx(th2[3]) \
         * Ry(th2[4]) * Rz(th2[5]) * dRz() * Ry(q2) * Tx(L_1) * T6d(th3) * Ry(q3) \
         * Tx(L_2) * T6d(th4) * Ry(q4)

    J9 = Tb * Ty(q1) * Ty(th1) * T6d(th2) * Ry(q2) * Tx(L_1) * \
        Tx(th3[0]) * dTx() * Ty(th3[1]) * Tz(th3[2]) * Rx(th3[3]) * Ry(th3[4]) * \
        Rz(th3[5]) * \
        Ry(q3) * Tx(L_2) * T6d(th4) * Ry(q4)

    J10 = Tb * Ty(q1) * Ty(th1) * T6d(th2) * Ry(q2) * Tx(L_1) * \
        Tx(th3[0]) * Ty(th3[1]) * dTy() * Tz(th3[2]) * Rx(th3[3]) * Ry(th3[4]) * \
        Rz(th3[5]) * \
        Ry(q3) * Tx(L_2) * T6d(th4) * Ry(q4)

    J11 = Tb * Ty(q1) * Ty(th1) * T6d(th2) * Ry(q2) * Tx(L_1) * \
        Tx(th3[0]) * Ty(th3[1]) * Tz(th3[2]) * dTz() * Rx(th3[3]) * Ry(th3[4]) * \
        Rz(th3[5]) * \
        Ry(q3) * Tx(L_2) * T6d(th4) * Ry(q4)

    J12 = Tb * Ty(q1) * Ty(th1) * T6d(th2) * Ry(q2) * Tx(L_1) * \
        Tx(th3[0]) * Ty(th3[1]) * Tz(th3[2]) * Rx(th3[3]) * dRx() * Ry(th3[4]) * \
        Rz(th3[5]) * \
        Ry(q3) * Tx(L_2) * T6d(th4) * Ry(q4)

    J13 = Tb * Ty(q1) * Tz(th1) * T6d(th2) * Ry(q2) * Tx(L_1) * \
        Tx(th3[0]) * Ty(th3[1]) * Tz(th3[2]) * Rx(th3[3]) * Ry(th3[4]) * dRy() * \
        Rz(th3[5]) * \
        Ry(q3) * Tx(L_2) * T6d(th4) * Ry(q4)

    J14 = Tb * Ty(q1) * Tz(th1) * T6d(th2) * Ry(q2) * Tx(L_1) * \
        Tx(th3[0]) * Ty(th3[1]) * Tz(th3[2]) * Rx(th3[3]) * Ry(th3[4]) * \
        Rz(th3[5]) * dRz() * \
        Ry(q3) * Tx(L_2) * T6d(th4) * Ry(q4)

    J15 = Tb * Ty(q1) * Tz(th1) * T6d(th2) * Ry(q2) * Tx(L_1) * T6d(th3) * Ry(q3) \
        * Tx(L_2) * Tx(th4[0]) * dTx() * Ty(th4[1]) * Tz(th4[2]) * Rx(th4[3]) * Ry(th4[4]) * \
        Rz(th4[5]) * Ry(q4)

    J16 = Tb * Ty(q1) * Tz(th1) * T6d(th2) * Ry(q2) * Tx(L_1) * T6d(th3) * Ry(q3) \
        * Tx(L_2) * Tx(th4[0]) * Ty(th4[1]) * dTy() * Tz(th4[2]) * Rx(th4[3]) * Ry(th4[4]) * \
        Rz(th4[5]) * Ry(q4)

    J17 = Tb * Ty(q1) * Tz(th1) * T6d(th2) * Ry(q2) * Tx(L_1) * T6d(th3) * Ry(q3) \
        * Tx(L_2) * Tx(th4[0]) * Ty(th4[1]) * Tz(th4[2]) * dTz() * Rx(th4[3]) * Ry(th4[4]) * \
        Rz(th4[5]) * Ry(q4)

    J18 = Tb * Ty(q1) * Tz(th1) * T6d(th2) * Ry(q2) * Tx(L_1) * T6d(th3) * Ry(q3) \
         * Tx(L_2) * Tx(th4[0]) * Ty(th4[1]) * Tz(th4[2]) * Rx(th4[3]) * dRx() * Ry(th4[4]) * \
        Rz(th4[5]) * Ry(q4)

    J19 = Tb * Ty(q1) * Tz(th1) * T6d(th2) * Ry(q2) * Tx(L_1) * T6d(th3) * Ry(q3) \
         * Tx(L_2) * Tx(th4[0]) * Ty(th4[1]) * Tz(th4[2]) * Rx(th4[3]) * Ry(th4[4]) * dRy() * \
        Rz(th4[5]) * Ry(q4)

    J20 = Tb * Ty(q1) * Tz(th1) * T6d(th2) * Ry(q2) * Tx(L_1) * T6d(th3) * Ry(q3) \
        * Tx(L_2) * Tx(th4[0]) * Ty(th4[1]) * Tz(th4[2]) * Rx(th4[3]) * Ry(th4[4]) * \
        Rz(th4[5]) * dRz() * Ry(q4)

    T = [J2, J3, J4, J5, J6, J7, J8, J9, J10, J11, J12, J13, J14, J15, \
        J16, J17, J18, J19, J20]

    J = np.matrix(zeros((6, len(T))))
    DK = dk3(q1, q2, q3, q4, th1, th2, th3, th4)
    if np.linalg.det(DK) == 0:
        print("Singularity error")
    D_inv = inv(np.matrix(DK))
    for i in range(0, len(T)):
        J[0, i] = T[i][0, 3]
        J[1, i] = T[i][1, 3]
        J[2, i] = T[i][2, 3]
        J[3, i] = (T[i] * D_inv)[2, 1]
        J[4, i] = (T[i] * D_inv)[0, 2]
        J[5, i]= (T[i] * D_inv)[1, 0]
    return J

def Kc(J_theta, J_q):
    return inv(np.block([[zeros((6, 6)), J_theta, J_q], [np.transpose(J_theta), Kt, zeros([19, 3])], [np.transpose(J_q), zeros([3, 22])]]))[0:6, 0:6]

def J_q1(q1, q2, q3, q4):
    J2 = Tb * Tz(q1) * Tz(th1) * T6d(th2) * Rz(q2) * dRz() * Tx(L_1) * T6d(th3) * Rz(q3) * Tx(L_2) * T6d(th4) * Rz(q4)
    J3 = Tb * Tz(q1) * Tz(th1) * T6d(th2) * Rz(q2) * Tx(L_1) * T6d(th3) * Rz(q3) * dRz() * Tx(L_2) * T6d(th4) * Rz(q4)
    J4 = Tb * Tz(q1) * Tz(th1) * T6d(th2) * Rz(q2) * Tx(L_1) * T6d(th3) * Rz(q3) * Tx(L_2) * T6d(th4) * Rz(q4) * dRz()

    T = [J2, J3, J4]
    J = np.matrix(zeros((6, len(T))))
    DK = dk1(q1, q2, q3, q4, th1, th2, th3, th4)
    if np.linalg.det(DK) == 0:
        print("Singularity error")
    D_inv = inv(np.matrix(DK))
    for i in range(0, len(T)):
        J[0, i] = T[i][0, 3]
        J[1, i] = T[i][1, 3]
        J[2, i] = T[i][2, 3]
        J[3, i] = (T[i] * D_inv)[2, 1]
        J[4, i] = (T[i] * D_inv)[0, 2]
        J[5, i]= (T[i] * D_inv)[1, 0]
    return J

def J_q2(q1, q2, q3, q4):
    J2 = Tb * Tx(q1) * Tx(th1) * T6d(th2) * Rx(q2) * dRx() * Tz(L_1) * T6d(th3) * Rx(q3) * Tz(L_2) * T6d(th4) * Rx(q4)
    J3 = Tb * Tx(q1) * Tx(th1) * T6d(th2) * Rx(q2) * Tz(L_1) * T6d(th3) * Rx(q3) * dRx() * Tz(L_2) * T6d(th4) * Rx(q4)
    J4 = Tb * Tx(q1) * Tx(th1) * T6d(th2) * Rx(q2) * Tz(L_1) * T6d(th3) * Rx(q3) * Tz(L_2) * T6d(th4) * Rx(q4) * dRx()

    T = [J2, J3, J4]
    J = np.matrix(zeros((6, len(T))))
    DK = dk2(q1, q2, q3, q4, th1, th2, th3, th4)
    if np.linalg.det(DK) == 0:
        print("Singularity error")
    D_inv = inv(np.matrix(DK))
    for i in range(0, len(T)):
        J[0, i] = T[i][0, 3]
        J[1, i] = T[i][1, 3]
        J[2, i] = T[i][2, 3]
        J[3, i] = (T[i] * D_inv)[2, 1]
        J[4, i] = (T[i] * D_inv)[0, 2]
        J[5, i]= (T[i] * D_inv)[1, 0]
    return J

def J_q3(q1, q2, q3, q4):
    J2 = Tb * Ty(q1) * Ty(th1) * T6d(th2) * Ry(q2) * dRx() * Tx(L_1) * T6d(th3) * Ry(q3) * Tx(L_2) * T6d(th4) * Ry(q4)
    J3 = Tb * Ty(q1) * Ty(th1) * T6d(th2) * Ry(q2) * Tx(L_1) * T6d(th3) * Ry(q3) * dRx() * Tx(L_2) * T6d(th4) * Ry(q4)
    J4 = Tb * Ty(q1) * Ty(th1) * T6d(th2) * Ry(q2) * Tx(L_1) * T6d(th3) * Ry(q3) * Tx(L_2) * T6d(th4) * Ry(q4) * dRy()

    T = [J2, J3, J4]
    J = np.matrix(zeros((6, len(T))))
    DK = dk3(q1, q2, q3, q4, th1, th2, th3, th4)
    if np.linalg.det(DK) == 0:
        print("Singularity error")
    D_inv = inv(np.matrix(DK))
    for i in range(0, len(T)):
        J[0, i] = T[i][0, 3]
        J[1, i] = T[i][1, 3]
        J[2, i] = T[i][2, 3]
        J[3, i] = (T[i] * D_inv)[2, 1]
        J[4, i] = (T[i] * D_inv)[0, 2]
        J[5, i]= (T[i] * D_inv)[1, 0]
    return J

z = 1

Tb = Tz(0)

th1  = 0
th2 = zeros(6)
th3 = zeros(6)
th4 = zeros(6)

x = 0.5
y = 0.5

# q1, q2, q3, q4 = ik(x, y, z)
# print(dk1(q1, q2, q3, q4, th1, th2, th3, th4))
#
# q1, q2, q3, q4 = ik(z, -y, x)
# print(dk2(q1, q2, q3, q4, th1, th2, th3, th4))
#
# q1, q2, q3, q4 = ik(x, -z, y)
# print(dk3(q1, q2, q3, q4, th1, th2, th3, th4))
# exit()
dx_fx = []
dx_fy = []
dx_fz = []

dy_fx = []
dy_fy = []
dy_fz = []

dz_fx = []
dz_fy = []
dz_fz = []

X = np.arange(0.1, 1, 0.01)
Y = np.arange(0.1, 1, 0.01)
x_pl = []
y_pl = []
for x in X:
    for y in Y:
        q1, q2, q3, q4 = ik(x, y, z)
        #print("Joints pos: ", q1, q2, q3, q4)

        Jq = J_q1(q1, q2, q3, q4)
        Jt = J_theta1(q1, q2, q3, q4)
        K1 = Kc(Jt, Jq)

        q1, q2, q3, q4 = ik(z, -y, x)
        Jq = J_q2(q1, q2, q3, q4)
        Jt = J_theta2(q1, q2, q3, q4)
        K2 = Kc(Jt, Jq)

        q1, q2, q3, q4 = ik(x, -z, y)
        Jq = J_q3(q1, q2, q3, q4)
        Jt = J_theta3(q1, q2, q3, q4)
        K3 = Kc(Jt, Jq)

        K = K1 + K2 + K3

        W = np.transpose(np.matrix([100, 0, 0, 0, 0, 0]))
        dt = inv(K) * W
        #print(dt[0, 0])
        dx_fx.append(dt[0, 0])
        dy_fx.append(dt[1, 0])
        dz_fx.append(dt[2, 0])
        W = np.transpose(np.matrix([0, 100, 0, 0, 0, 0]))
        dt = inv(K) * W
        dx_fy.append(dt[0, 0])
        dy_fy.append(dt[1, 0])
        dz_fy.append(dt[2, 0])

        W = np.transpose(np.matrix([0, 0, 100, 0, 0, 0]))
        dt = inv(K) * W
        dx_fz.append(dt[0, 0])
        dy_fz.append(dt[1, 0])
        dz_fz.append(dt[2, 0])

        x_pl.append(x)
        y_pl.append(y)

dy_fy = np.asarray(dy_fy)
dx_fx = np.asarray(dx_fx)
dz_fz = np.asarray(dz_fz)
x_pl = np.asarray(x_pl)
y_pl = np.asarray(y_pl)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x_pl, y_pl, dz_fz, color='red')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x_pl, y_pl, dy_fy, color='red')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x_pl, y_pl, dx_fx, color='red')
plt.show()
