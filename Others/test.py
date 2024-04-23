import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
points = digits.data[100:200]
print(points.shape[0])
