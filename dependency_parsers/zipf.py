from data_process import DataProcessor
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import numpy as np

UD_ENGLISH_GUM = 'dependency-parsers/en_gum-ud-train.conllu'

def zipf(x, k, a):
    return k / (x**a)

def plot_zipf(file):
    data_processor = DataProcessor(file)
    vocab = data_processor.vocab
    words = data_processor.words
    tuples = data_processor.index_to_token_tuples

    ys = [words[value] for _, value in tuples if value in words]
    xs = [key for key, value in tuples if value in words]

    popt, _ = curve_fit(zipf, xs, ys)

    k, a = popt
    curve = zipf(xs, k, a)
    plt.plot(np.log2(xs), np.log2(ys), 'ro')
    plt.plot(np.log2(xs), np.log2(curve))
    plt.show()

plot_zipf(UD_ENGLISH_GUM)
