from generate_data_gaussian import load_data
from configs import *
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from copy import deepcopy

def extract(cs):
    acs=[]
    ms=[]
    ss=[]
    for item in cs:
        acs.append(item['acc'])
        ms.append(item['m'])
        ss.append(item['sigma'])
    return ss,ms,acs


aic_accuracy = load_data(CRI_FILES[0])
bic_accuracy = load_data(CRI_FILES[0])

aic_10 = deepcopy(aic_accuracy[0:9])
aic_100 = deepcopy(aic_accuracy[9:18])
bic_10 = deepcopy(bic_accuracy[0:9])
bic_100 = deepcopy(bic_accuracy[9:18])


fig = plt.figure(figsize=(18,12))
plt.subplots_adjust(left=.02, right=.98, bottom=.01, top=.96, wspace=.05,
                    hspace=.01)
ax = fig.add_subplot(2,2,1,projection="3d")
plt.title("AIC m=10")
ax.set_xlabel("sigma")
ax.set_ylabel("m")
ax.set_zlabel("accuracy")
ax.scatter(*extract(aic_10), s=POINTSIZE)

ax = fig.add_subplot(2,2,2,projection="3d")
plt.title("BIC m=10")
ax.set_xlabel("sigma")
ax.set_ylabel("m")
ax.set_zlabel("accuracy")
ax.scatter(*extract(bic_10), s=POINTSIZE)

ax = fig.add_subplot(2,2,3,projection="3d")
plt.title("AIC m=100")
ax.set_xlabel("sigma")
ax.set_ylabel("m")
ax.set_zlabel("accuracy")
ax.scatter(*extract(aic_100), s=POINTSIZE)

ax = fig.add_subplot(2,2,4,projection="3d")
plt.title("BIC m=100")
ax.set_xlabel("sigma")
ax.set_ylabel("m")
ax.set_zlabel("accuracy")
ax.scatter(*extract(bic_100), s=POINTSIZE)
plt.show()