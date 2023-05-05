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
I = "10"
v = "1.0"

alpha = [str(9 * i) for i in range(0, 11)]
x_s = [str(np.around(i * 0.1, decimals=3)) for i in range(1, 10)]

info_str = f"S={S} D={D} I={I} h=6 v={v} 10x10km"
dataset_name = f"dataset_S{S}_alpha"

for x in (0.5, 0.6, 0.7, 0.8, 0.9):
    for y in x_s:
        for a in alpha:
            mv_str = ""
            for s in range(0, int(S)):
                s = str(s)
                mv_str += f'mv ./imgs/{s}_res_100.ppm ./{dataset_name}/"x={x}_y={y}_angle={a}  {s}_res_100.ppm" \n'
            mv_str += f'mv ./concentration.txt ./{dataset_name}/"{info_str} angle={a} concentration.txt" \n'

            cmd_str = f'cd ./coagulation-diffusion-2d/run_folder/ \n' \
                      f'echo "`date +%Y_%m_%d_%H:%M:%S` Start tet.exe x={x} y={y} coag={coag} v={v} angle={a} I={I}" \n' \
                      f'./tet.exe {x} {y} {coag} {v} {a} {I}\n' \
                      f'wait \n' \
                      f'if [ ! -d ./{dataset_name} ]; then \n' \
                      f'    mkdir -p ./{dataset_name}; \n' \
                      f'fi \n' \
                      f'{mv_str}' \
                      f'if [ ! -d ./imgs ]; then \n' \
                      f'    mkdir -p ./imgs; \n' \
                      f'fi \n'
            subprocess.run(cmd_str, shell=True)

np.savetxt(f'./{dataset_name}/_x_conditions.txt', x_s)
np.savetxt(f'./{dataset_name}/_a_conditions.txt', alpha)
print("FINISHED")

