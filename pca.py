# Example
# Principal Component Analysis in an image with python, scikit-learn and scikit-image

from sklearn.decomposition import PCA
from pylab import *
from skimage import data, io, color
import matplotlib.pyplot as plt
from matplotlib import gridspec

file = "Lenna.png"
lenna = io.imread(file, as_grey=True)
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])

fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

plt.subplot(gs[0])
io.imshow(lenna)
xlabel('Original Image')

for i in range(1, 4):
    n_comp = 5 ** i
    pca = PCA(n_components=n_comp)
    pca.fit(lenna)
    lenna_pca = pca.fit_transform(lenna)
    lenna_restored = pca.inverse_transform(lenna_pca)
    plt.subplot(gs[i])
    io.imshow(lenna_restored)
    xlabel('Restored image n_components = %s' % n_comp)
    print('Variance retained %s %%' % (
                (1 - sum(pca.explained_variance_ratio_) / size(pca.explained_variance_ratio_)) * 100))
    print('Compression Ratio %s %%' % (float(size(lenna_pca)) / size(lenna) * 100))

show()
