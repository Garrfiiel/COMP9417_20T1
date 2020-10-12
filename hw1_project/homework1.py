import pandas
import numpy
import math
from matplotlib import pyplot as plt


def normalization(df):
    return (df-df.min())/(df.max()-df.min())

def SGD(x_training, y_training):
    theta_0 = -1
    theta_1 = -0.5
    loss = []
    y_pred = numpy.zeros((300, 1))

    for i in range(50):
        for j in range(300):
            y_pred[j][0] = x_training[j] * theta_1 + theta_0
            theta_0 = theta_0 + 0.01 * (y_training[j] - y_pred[j][0])
            theta_1 = theta_1 + 0.01 * (y_training[j] - y_pred[j][0]) * x_training[j]

        for k in range(300):
            y_pred[k][0] = x_training[k] * theta_1 + theta_0

        s = (y_training - y_pred) * (y_training - y_pred)
        s_list = s.tolist()
        sl = []
        for n in s_list:
            for m in n:
                sl.append(m)
        loss.append(sum(sl) / 300)
    return theta_0, theta_1, loss

def RMSE(x_test, y_test, theta_0, theta_1):
    y_pred = numpy.zeros((100, 1))
    for i in range(100):
        y_pred[i][0] = x_test[i] * theta_1 + theta_0
    s = (y_test - y_pred) * (y_test - y_pred)
    s = s.tolist()
    sl = []
    for i in s:
        for j in i:
            sl.append(j)
    result = math.sqrt(sum(sl) / 100)
    return result

file = pandas.read_csv('house_prices.csv')
df = pandas.DataFrame(file)
df_curr = df.drop('No', axis = 1)
y = df_curr[['house price of unit area']]
x = df_curr[['house age', 'distance to the nearest MRT station', 'number of convenience stores']]

x_preprocessed = normalization(x)

x_training = x_preprocessed[0: 300]
x_test = x_preprocessed[300: 400]
y_training = (y[0: 300]).values
y_test = (y[300: 400]).values

x_hatrn = x_training[['house age']].values
theta_0_ha, theta_1_ha, loss_ha = SGD(x_hatrn, y_training)
plt.title('Cost Functions Using House Age Feature')
plt.xlabel('iteration')
plt.ylabel('cost function')
plt.plot(loss_ha)
plt.show()
RMSE_hatrn = math.sqrt(loss_ha[49])
print('Theta_0,1 using house age feature:', theta_0_ha, theta_1_ha)
print('RMSE for training set using house age feature:', RMSE_hatrn)

x_hatst = x_test[['house age']].values
RMSE_hatst = RMSE(x_hatst, y_test, theta_0_ha, theta_1_ha)
print('RMSE for test set using house age feature:', RMSE_hatst)

x_dtstrn = x_training[['distance to the nearest MRT station']].values
theta_0_dts, theta_1_dts, loss_dts = SGD(x_dtstrn, y_training)
plt.title('Cost Functions Using Distance to The Station Feature')
plt.xlabel('iteration')
plt.ylabel('cost function')
plt.plot(loss_dts)
plt.show()
RMSE_dtstrn = math.sqrt(loss_dts[49])
print('Theta_0,1 using distance to the station feature:', theta_0_dts, theta_1_dts)
print('RMSE for training set using distance to the station feature:', RMSE_dtstrn)

x_dtstst = x_test[['distance to the nearest MRT station']].values
RMSE_dtstst = RMSE(x_dtstst, y_test, theta_0_dts, theta_1_dts)
print('RMSE for test set using distance to the station feature:', RMSE_dtstst)

x_ncstrn = x_training[['number of convenience stores']].values
theta_0_ncs, theta_1_ncs, loss_ncs = SGD(x_ncstrn, y_training)
plt.title('Cost Functions Using Number of Stores Feature')
plt.xlabel('iteration')
plt.ylabel('cost function')
plt.plot(loss_ncs)
plt.show()
RMSE_ncstrn = math.sqrt(loss_ncs[49])
print('Theta_0,1 using number of stores feature:', theta_0_ncs, theta_1_ncs)
print('RMSE for training set using number of stores feature:', RMSE_ncstrn)

x_ncstst = x_test[['number of convenience stores']].values
RMSE_ncstst = RMSE(x_ncstst, y_test, theta_0_ncs, theta_1_ncs)
print('RMSE for test set using number of stores feature:', RMSE_ncstst)