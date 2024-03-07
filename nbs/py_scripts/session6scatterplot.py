from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def scatterplot_vis_word_list(words, model):

    X = []

    for word in words:
        X.append(model[word])

    pca = PCA(n_components=2)
    result = pca.fit_transform(X)

    plot = plt.scatter(result[:, 0], result[:, 1])

    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    plt.title("Scatter plot of given word list in 2D using PCA", fontsize = 12)
    plt.xlabel('PCA component 1')
    plt.ylabel('PCA component 2')

    return plot