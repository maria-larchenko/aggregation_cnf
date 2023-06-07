import subprocess
import numpy as np

cmd_str = 'cd ./coagulation-diffusion-2d/run_folder/ \n' \
          'ls \n' \
          'make \n' \
          'wait \n' \
          '\n'
subprocess.run(cmd_str, shell=True)

D = "e0"
S = "10"
coag = "1"

alpha = [0, ]   # [0, 9, 18, 27, 36, 45, 54, 63, 72, 90] #[(9 * i) for i in range(0, 11)]
x_s = [0.05, ]  # [i * 0.1 for i in range(1, 10)]
y_s = [0.5, ]  # [i * 0.1 for i in range(1, 10)]
I = [1, 10, 50, 100]
V = [0.5, 1.0, 10.0]

dataset_name = f"dataset_S{S}_shape"
info_str = f"coag={coag} S={S} D={D} angle=0 x_s=(.05, .5) h=6 10x10km"


subprocess.run(f'echo "{info_str}" >> info.txt', shell=True)
for x in x_s:
    for y in y_s:
        for a in alpha:
            for v in V:
                for i in I:
                    x = str(np.around(x, decimals=3))
                    y = str(np.around(y, decimals=3))
                    a = str(int(a))
                    v = str(np.around(x, decimals=3))
                    i = str(int(i))
                    mv_str = ""
                    for s in range(0, int(S)):
                        s = str(s)
                        mv_str += f'mv ./imgs/{s}_res_100.ppm ./{dataset_name}/"v={v} I={i} {s}_res_100.ppm" \n'
                    mv_str += f'mv ./concentration.txt ./{dataset_name}/"v={v} I={i} concentration.txt" \n'

                    cmd_str = f'cd ./coagulation-diffusion-2d/run_folder/ \n' \
                              f'echo "`date +%Y_%m_%d_%H:%M:%S` Start tet.exe x={x} y={y} coag={coag} v={v} angle={a} I={i}" \n' \
                              f'./tet.exe {x} {y} {coag} {v} {a} {i}\n' \
                              f'wait \n' \
                              f'if [ ! -d ./{dataset_name} ]; then \n' \
                              f'    mkdir -p ./{dataset_name}; \n' \
                              f'fi \n' \
                              f'{mv_str}' \
                              f'if [ ! -d ./imgs ]; then \n' \
                              f'    mkdir -p ./imgs; \n' \
                              f'fi \n'
                    subprocess.run(cmd_str, shell=True)
print("FINISHED")

