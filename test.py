import numpy as np
from sklearn.decomposition import PCA
s = set()
s.add((1,9))
s.add((2,9))
s.add((3,9))

s2 = set()
s2.add((1,9))
s2.add((2,9))

print(s.intersection(s2))