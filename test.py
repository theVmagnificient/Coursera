import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import scipy.optimize
from scipy.optimize import minimize_scalar


data = pd.read_csv('weights_heights.csv', index_col='Index')

#data.plot(y='Height', kind='hist',
#           color='red',  title='Height (inch.) distribution')


print(data.head(5))

#data.plot(y='Weight', kind='hist',
#           color='green',  title='Weight distribution')


def make_bmi(height_inch, weight_pound):
    METER_TO_INCH, KILO_TO_POUND = 39.37, 2.20462
    return (weight_pound / KILO_TO_POUND) / (height_inch / METER_TO_INCH) ** 2

data['BMI'] = data.apply(lambda row: make_bmi(row['Height'],
                                              row['Weight']), axis=1)


#sns.pairplot(data)

def weight_category(weight):
    if weight < 120:
        return 1
    elif weight >= 150:
        return 3
    else:
        return 2


data['weight_cat'] = data['Weight'].apply(weight_category)

#sns.boxplot(data['weight_cat'], data['Height'])

#data.plot(y='Height', x='Weight', kind='scatter',
#           color='red',  title='Рост от веса')

def error(w0, w1, data):
    sum = 0.0
    for _, row in data.iterrows():
        sum += (row['Height'] - (w0 + w1 * row['Weight'])) ** 2
    return sum


def y1(x):
    return 60 + 0.05 * x

def y2(x):
    return 50 + 0.16 * x

x = np.linspace(60, 170)

#data.plot(x='Weight', y='Height', kind='scatter', title='Correlation between weight and height')
#plt.plot(x, y1(x), color='green')
#plt.plot(x, y2(x), color='red')
#plt.legend( ('line (60, 0.05)', 'line (50, 0.16)') );

#w0 = 50.0
#error_graph = []
#for w1 in np.arange(-5.0, 5.0, 1):
#    error_graph.append(error(w0, w1, data))
#plt.plot(error_graph)


fig = plt.figure()
ax = fig.gca(projection='3d')

w0 = np.arange(-5, 5, 0.25)
w1 = np.arange(-5, 5, 0.25)
w0, w1 = np.meshgrid(w0, w1)
Z = error(w0, w1, data)

surf = ax.plot_surface(w0, w1, Z)
ax.set_xlabel('Intercept')
ax.set_ylabel('Slope')
ax.set_zlabel('Error')
ax.set_title('Measurement Error in 3D')
plt.show()

a = 0