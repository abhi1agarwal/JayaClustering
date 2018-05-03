import pandas as pd
from constants import URL


def get_file_contents(url):
    dataset = pd.read_csv(url)
    #print(dataset)


    return dataset


def output(fit):
    return 1*fit

delta_x = {"./dataset/random2_2.data.txt":1.0,
			"./dataset/glass.data.txt":4.0,
			"./dataset/ionosphere.data.txt":4.0,
			"./dataset/iris.data.txt":0.5,
			"./dataset/random3_2.data.txt":1,
			"./dataset/segmentation.data.txt":2,
			"./dataset/parkinsons.data.txt":0,
			"./dataset/wine.data.txt":10,
			"./dataset/sonar.data.txt":0	
			}