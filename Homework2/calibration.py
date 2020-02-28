import numpy as np
from funcs import *
from numpy import arctan2, transpose, zeros
from numpy.linalg import inv, det
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

K_theta = np.matrix(zeros((3, 3)))
K_theta[0, 0] = 10**6
K_theta[1, 1] = 2 * 10**6
K_theta[2, 2] = 0.5 * 10**6

K_theta = inv(K_theta)

noise_var = 5 * 10**(-4)

param_num = 3

th1 = 0
th2 = 0
th3 = 0

#J_q = np.zeros(6, 1)

def ik(x, y, z, th1, th2, th3):
     q1 = z - th1
     q2 = arctan2(y, x) - th2
     q3 = (x**2 + y**2)**.5 - th3

     return q1, q2, q3

def T_robot(q1, q2, q3, th1, th2, th3):
    return Tz(q1) * Tz(th1) * Rz(q2) * Rz(th2) * Tx(q3) * Tx(th3)

def J_theta(q1, q2, q3, th1, th2, th3):
    J1 = Tz(q1) * Tz(th1) * dTz() * Rz(q2) * Rz(th2) * Tx(q3) * Tx(th3)
    J2 = Tz(q1) * Tz(th1) * Rz(q2) * Rz(th2) * dRz() * Tx(q3) * Tx(th3)
    J3 = Tz(q1) * Tz(th1) * Rz(q2) * Rz(th2) * Tx(q3) * Tx(th3) * dTx()

    T = [J1, J2, J3]
    J = np.matrix(zeros((6, len(T))))
    DK = T_robot(q1, q2, q3, th1, th2, th3)
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

K_real = np.matrix([[10**-6], [.5 * 10**-6], [2 * 10**-6]])
A = []
dt = []
sum1 = np.matrix(zeros((3, 3)))
measure_num = 30
while det(sum1) == 0:
    for i in range(0, measure_num):
        q1 = np.random.uniform(0, 2)
        q2 = np.random.uniform(0, 2 * np.pi)
        q3 = np.random.uniform(0, 2)
        w = np.matrix(np.random.uniform(0, 100, 6))
        J = J_theta(q1, q2, q3, 0, 0, 0)
        A_i = np.matrix(zeros((6, 3)))
        for j in range(0, param_num):
            A_i[:, j] = J[:, j] * transpose(J[:, j]) * transpose(w)
        A.append(A_i)
        noise = np.random.normal(0, noise_var)
        dt.append(A_i * K_real + noise)

    sum1 = np.matrix(zeros((3, 3)))
    for j in range(0, measure_num):
        sum1 += transpose(A[j]) * A[j]

sum2 = np.matrix(zeros((3, 1)))
for i in range(0, measure_num):
    sum2 += transpose(A[i]) * dt[i]

x_est = inv(sum1) * sum2
k = np.array([1 / x for x in x_est])
print("Estimated stiffness, MN / m: ", k / 10**6)

K_theta_est = np.matrix(np.zeros((3, 3)))
K_theta_est[0, 0] = k[0]
K_theta_est[1, 1] = k[1]
K_theta_est[2, 2] = k[2]

K_theta_est = inv(K_theta_est)

points_num = 100
x = np.linspace(-1, 1, points_num)
y = (1 - x**2)**.5
z = np.linspace(0.999, 1, points_num)

x0 = np.zeros(points_num)
y0 = np.zeros(points_num)
z0 = np.zeros(points_num)

x1 = np.zeros(points_num)
y1 = np.zeros(points_num)
z1 = np.zeros(points_num)

W = np.matrix(np.random.uniform(10, 10000, 6))
W = np.matrix([1000, 1000, 1000, 1000, 1000, 1000])
print("Applied force: ", W)
q = []
q_calib = []
for i in range(0, len(x)):
    q1, q2, q3 = ik(x[i], y[i], z[i], 0, 0, 0)

    J = J_theta(q1, q2, q3, 0, 0, 0)
    dt_real = np.asarray(J * K_theta * transpose(J) * transpose(W)).reshape(6,)
    dt_est = np.asarray(J * K_theta_est * transpose(J) * transpose(W)).reshape(6,)

    x0[i] = x[i] + dt_real[0]
    y0[i] = y[i] + dt_real[1]
    z0[i] = z[i] + dt_real[2]

    my_ik = ik(x[i] - dt_est[0], y[i] - dt_est[1], z[i] - dt_est[2], 0, 0, 0)
    xyz = T_robot(my_ik[0], my_ik[1], my_ik[2], 0, 0, 0)[0:3, 3]
    x1[i] = xyz[0] + dt_real[0]
    y1[i] = xyz[1] + dt_real[1]
    z1[i] = xyz[2] + dt_real[2]

print("Errors for not calibrated robot (on x, y and z) :", np.mean(x - x0), np.mean(y - y0), np.mean(z - z0))
x_err = "{0:.6f}".format(np.mean(x - x1))
y_err = "{0:.6f}".format(np.mean(y - y1))
z_err = "{0:.6f}".format(np.mean(z - z1))
print("Errors for calibrated robot (on x, y and z) : ", x_err, y_err, z_err )

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(x, y, z, color='red', label='desired')
ax.plot(x0, y0, z0, color='green', label='non-calibrated')
ax.plot(x1, y1, z1, color='blue', label='calibrated')
ax.legend()
plt.show()
