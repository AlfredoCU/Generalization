#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 8 18:33:44 2020

@author: alfredocu
"""

# Modelo lineal, caracteristicas polinomicas. sobreentrenamiento.

# Bibliotecas.
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

# Algoritmos.
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Semilla.
np.random.seed(42)

# Datos
m = 100

# Vector aleatorio.
x = 6 * np.random.rand(m, 1) - 3

# Funsión
y = 0.5  * x**2 + x + 0.2 * np.random.randn(m, 1)

# Graficar.
# plt.plot(x, y, ".b")
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", fontsize=18)
# plt.axis([-3, 3, 0, 10])
# plt.show() 

# Es un preprocesamiento que se realiza a los datos y ayuda a enriquecer 
# las entradas. 
# from sklearn.preprocessing import PolynomialFeatures

# Expancion polinomicas. Aproximacion polimonial.
# De grado 2, y no incluyes sesgo o bias
# poly_features = PolynomialFeatures(degree=2, include_bias=False)

# Transforma nuestros datos.
# xploy = poly_features.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y)

d = 300

# Juntar varios modelos a la vez.
model = Pipeline([("poly", PolynomialFeatures(degree=d, include_bias=False)),
                ("scaler", StandardScaler()),
                ("lin_reg", LinearRegression())])

# Entrenamos.
model.fit(xtrain, ytrain)

print("Train score: ", model.score(xtrain, ytrain))
print("Test score: ", model.score(xtest, ytest))

# print(model.score(x, y))

# Dibujar.
xnew = np.linspace(-3, 3, 1000).reshape(1000, 1)
ynew = model.predict(xnew)

plt.plot(xtrain, ytrain, ".b")
plt.plot(xtest, ytest, ".r")
plt.plot(xnew, ynew, "-k")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", fontsize=18)
plt.axis([-3, 3, -5, 15])
plt.show() 

# Intercepción.
# print(model.intercept_)

# Coeficientes.
# print(model.coef_)

