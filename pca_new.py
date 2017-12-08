import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import split
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn.manifold import TSNE
import os

def create_dict(df, method):
    y_dict = {}
    for i, y in enumerate(df['label']):
        if y in y_dict:
            y_dict[y][0].append(df[method+'-one'].iloc[i])
            y_dict[y][1].append(df[method+'-two'].iloc[i])
            y_dict[y][2].append(df[method+'-three'].iloc[i])
        else:
            y_dict[y] = [[df[method+'-one'].iloc[i]], [df[method+'-two'].iloc[i]], [df[method+'-three'].iloc[i]]]
    import operator
    for y in y_dict:
        if y == 2:
            continue
        for _ in xrange(50):
            for i in xrange(3):
                index, value = max(enumerate(y_dict[y][i]), key=operator.itemgetter(1))
                y_dict[y][0].pop(index)
                y_dict[y][1].pop(index)
                y_dict[y][2].pop(index)
                index, value = min(enumerate(y_dict[y][i]), key=operator.itemgetter(1))
                y_dict[y][0].pop(index)
                y_dict[y][1].pop(index)
                y_dict[y][2].pop(index)

    return y_dict


def just_plot_it_man(y_dict, method):
    for y_val in y_dict:
        print y_val, len(y_dict[y_val][0])
    for angle in range(0, 361, 75):
        print angle
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors_and_markers = [('r', 'o'), ('b', 'o'), ('c', 'o')]
        label_list=['','Charged Off','Default','Fully Paid']
        ax.view_init(30, angle)
        i = 0
        for y_val in y_dict:
            y_val = int(y_val)
            print "Y_VAL:"+str(y_val)
            ax.scatter(y_dict[y_val][0], y_dict[y_val][1], y_dict[y_val][2], c=colors_and_markers[i][0], marker=colors_and_markers[i][1],s=0.2,alpha=0.7,label=label_list[y_val])
            i += 1
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel(method + "1")
        ax.set_ylabel(method + "2")
        ax.set_zlabel(method + "3")

        plt.legend()
        if not os.path.exists(method+"_pics"):
            os.makedirs(method+"_pics")
        plt.savefig(method+"_pics/"+method + "plot_" + str(angle) + ".png", dpi=200)



'''mnist = fetch_mldata("MNIST original")
X = mnist.data / 255.0
y = mnist.target'''

X_train, X_test, y_train, y_test = split.split(split.PD)

print X_train.shape, y_train.shape

#feat_cols = [ 'pixel'+str(i) for i in range(X_train.shape[1]) ]

df = pd.DataFrame()
df_tsne = pd.DataFrame()


print len(y_train)
df['label'] = y_train.iloc[:,0]
df_tsne['label'] = y_train.iloc[:,0]
#y_train = y_train.apply(lambda i: str(i))
s = set([])


print 'Size of the dataframe: {}'.format(X_train.shape)

pca = PCA(n_components=3)
pca_result = pca.fit_transform(X_train)

tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X_train)

df_tsne['tsne-one'] = tsne_results[:,0]
df_tsne['tsne-two'] = tsne_results[:,1]
df_tsne['tsne-three'] = tsne_results[:,2]


df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]


just_plot_it_man(create_dict(df,"pca"),"pca")

just_plot_it_man(create_dict(df_tsne,"tsne"),"tsne")

