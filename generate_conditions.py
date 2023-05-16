import numpy as np


S = 10

alpha = [0, 9, 18, 27, 36, 45, 54, 63, 72, 90] #[(9 * i) for i in range(0, 11)]
x_s = [(np.around(i * 0.1, decimals=3)) for i in range(1, 10)]
y_s = x_s
size = [0, 9,]
dataset_name = f"dataset_S{S}_alpha"

cond_lst = []
for x in x_s:
    for y in x_s:
        for a in alpha:
            for s in size:
                cond_lst.append((x, y, a, s))

np.savetxt(f'./datasets/{dataset_name}/__conditions.txt', cond_lst, header="x_s, y_s, alpha, size")
print("FINISHED")

