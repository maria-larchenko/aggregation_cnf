import numpy as np

S = 10


dataset_name = f"dataset_S{S}_alpha"
alpha = [0, 9, 18, 27, 36, 45, 54, 63, 72, 90] #[(9 * i) for i in range(0, 11)]
x_s = [(np.around(i * 0.1, decimals=3)) for i in range(1, 10)]
y_s = x_s
size = [0, 9,]


dataset_name = f"dataset_S{S}_shape"
alpha = [0, ]
x_s = [0.05, ]
y_s = [0.5, ]
size = [i for i in range(0, int(S))]
I = [1, 10, 50, 100]
V = [0.5, 1.0, 10.0]

cond_lst = []
for x in x_s:
    for y in y_s:
        for a in alpha:
            for v in V:
                for i in I:
                    for s in size:
                        # cond_lst.append((x, y, a, s))     # dataset_S{S}_alpha
                        cond_lst.append((v, i, s))     # dataset_S{S}_shape

np.savetxt(f'./datasets/{dataset_name}/__conditions.txt', cond_lst, header="v, I, size")
print("FINISHED")