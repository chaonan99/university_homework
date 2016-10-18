using DataFrames

df_train = readtable("dataset3.txt", separator = '\t')
df_test = readtable("dataset4.txt", separator = '\t')

# close(df_train)
# close(df_test)