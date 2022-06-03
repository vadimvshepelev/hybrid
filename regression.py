import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([0., 120., 240., 360., 480., 600., 720., 840., 960., 1080.])
y = np.array([52683.98, 52582.59, 52748.68, 52717.39, 52603.43, 52517.86, 52536.28, 52484.11, 52472.27, 52388.01])

x_train = x.reshape((-1, 1))

print(x_train)
print(y)

model = LinearRegression()

model.fit(x_train, y)
print(model.coef_, model.intercept_)
