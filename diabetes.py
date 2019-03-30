#!/usr/bin/env python

import os
import os.path as op
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.linear_model import LassoLarsIC, LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
import statsmodels.api as sm


diabetes = load_diabetes()
x = pd.DataFrame(data=diabetes["data"],columns=diabetes["feature_names"])
y = diabetes["target"].reshape(-1,1)

model_1 = sm.OLS(y, x)
result_1 = model_1.fit()
print(result_1.summary())

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=0)

#############################################################################
# Create linear regression object

regr = LinearRegression()
# Train the model using the training sets
regr.fit(X_train, Y_train)
# Make predictions using the testing set
Y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, Y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, Y_pred))


#############################################################################
# Create Ridge regression object

clf = Ridge(alpha=1.0)
# Train the model using the training sets
clf.fit(X_train, Y_train)
# Make predictions using the testing set
Y_predclf = clf.predict(X_test)
# The coefficients
print('Coefficients: \n', clf.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, Y_predclf))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, Y_predclf))


#############################################################################
# Create Lasso object

regrLasso = Lasso(alpha=0.1)
# Train the model using the training sets
regrLasso.fit(X_train, Y_train)
# Make predictions using the testing set
Y_predLasso = regrLasso.predict(X_test)
# The coefficients
print('Coefficients: \n', regrLasso.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, Y_predLasso))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, Y_predLasso))

# #############################################################################
# LassoLarsIC: least angle regression with BIC criterion

model_bic = LassoLarsIC(criterion='bic')
model_bic.fit(X_train, Y_train)
alpha_bic_ = model_bic.alpha_


def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
             linewidth=3, label='%s criterion' % name)
    plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                label='alpha: %s estimate' % name)
    plt.xlabel('-log(alpha)')
    plt.ylabel('criterion')

plt.figure()
plot_ic_criterion(model_bic, 'BIC', 'r')
plt.legend()
plt.title("Least angle regression with BIC criterion")
plt.savefig("files_figures/BIC_criterion.png")
plt.clf()

# plt.show()
