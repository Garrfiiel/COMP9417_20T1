with open('house_prices.csv') as file:
    data = csv.reader(file)
    for i in data:
        print(i)

theta_0 = -1
theta_1 = -0.5
y_pred = numpy.zeros(300, 1)
for i in range(50):
    for j in range(300):
        y_pred[j][0] = x_training_ha[j] * theta_1 + theta_0
        theta_0 = theta_0 + 0.01 * (y_training_ha[j] - y_pred[j][0])
        theta_1 = theta_1 + 0.01 * (y_training_ha[j] - y_pred[j][0]) * x_training_ha[j]
#print(y_pred)
y_hatrn_pred = numpy.zeros((300, 1))
for i in range(300):
    y_hatrn_pred[i][0] = x_hatrn[i] * theta_1 + theta_0
sel_hatrn = squared_error(y_training, y_hatrn_pred)
RMSE_hatrn = math.sqrt(sum(sel_hatrn) / 300)
print(RMSE_hatrn)

for i in range(50):
    for j in range(300):
        y_pred[j][0] = x_hatrn[j] * theta_1 + theta_0
        theta_0 = theta_0 + 0.01 * (y_training[j] - y_pred[j][0])
        theta_1 = theta_1 + 0.01 * (y_training[j] - y_pred[j][0]) * x_hatrn[j]

    for j in range(300):
        y_pred[j][0] = x_hatrn[j] * theta_1 + theta_0

    d = y_training - y_pred
    s = d * d
    s_list = s.tolist()
    sl = []
    for n in s_list:
        for m in n:
            sl.append(m)
    loss.append(sum(sl) / 300)
print(loss)
plt.plot(loss)
plt.show()
print(math.sqrt(loss[49]))