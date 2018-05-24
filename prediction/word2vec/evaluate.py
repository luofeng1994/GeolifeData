import cPickle as pickle
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


embedding_path = './variable/embedding.pkl'
embedding = pickle.load(open(embedding_path, 'r'))
place_index = [i for i in range(1, 15)]
visualizeVecs = [embedding[str(i)] for i in place_index]
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(place_index) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2])

for i in xrange(len(place_index)):
    plt.text(coord[i,0], coord[i,1], place_index[i],
        bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.savefig('./variable/q3_word_vectors.png')