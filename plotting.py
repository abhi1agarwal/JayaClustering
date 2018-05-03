import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from constants import K
def plot2D(cen,pop_distri):
    #print(cen)


    for i in range(0,K) :
        x,y = pop_distri[0][i].T
        plt.scatter(x,y)

    #x, y = pop_distri[0][0].T
    #plt.scatter(x, y)

    #x1, y1 = pop_distri[0][1].T
    #plt.scatter(x1, y1)

    #x2,y2 = pop_distri[0][2].T
    #plt.scatter(x2,y2)
    xc, yc = cen.T
    plt.scatter(xc, yc)
    plt.show()


def plot3D(cen,pop_distri):
    #print(cen)

    fig = plt.figure()
    ax = Axes3D(fig)


    for i in range(0,K) :
        x,y,z = pop_distri[0][i].T
        ax.scatter(x,y,z)

    xc, yc ,zc = cen.T
    ax.scatter(xc, yc,zc)
    plt.show()