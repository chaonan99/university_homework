using DataFrames

df_train = readtable("dataset3.txt", separator = '\t')
df_test = readtable("dataset4.txt", separator = '\t')

# Bayes: ? BayesNet http://nbviewer.jupyter.org/github/sisl/BayesNets.jl/blob/master/doc/BayesNets.ipynb
#           http://distributionsjl.readthedocs.io/en/latest/
# Fisher: http://multivariatestatsjl.readthedocs.io/en/latest/lda.html
# SVM: https://github.com/JuliaStats/SVM.jl, https://github.com/simonster/LIBSVM.jl
# MLP: http://serhanaya.github.io/neural-networks-julia-implementation/
