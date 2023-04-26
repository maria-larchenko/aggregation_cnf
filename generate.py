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

dataset_n = f"dataset_S{S}_intensity"
for i in (20, 100):
    x = str(0.5)
    y = str(0.5)
    I = str(i)
    mv_str = ""
    for s in range(0, int(S)):
        s = str(s)
        mv_str += f'mv ./imgs/{s}_res_100.ppm ./{dataset_n}/"I={I}_h=6_D={D}_v=1.0_angle=30  {s}_res_100.ppm" \n'
    mv_str += f'mv ./concentration.txt ./{dataset_n}/"I={I}_h=6_D={D}_v=1.0_angle=30 concentration.txt" \n'

    cmd_str = f'cd ./coagulation-diffusion-2d/run_folder/ \n' \
	      f'echo "`date +%Y_%m_%d_%H:%M:%S` Start tet.exe x={x} y={y} coag={coag} v=1.0 angle=30 I={I}" \n' \
	      f'./tet.exe {x} {y} {coag} 1.0 30 {I}\n' \
	      f'wait \n' \
	      f'if [ ! -d ./{dataset_n} ]; then \n' \
	      f'    mkdir -p ./{dataset_n}; \n' \
	      f'fi \n' \
	      f'{mv_str}' \
	      f'if [ ! -d ./imgs ]; then \n' \
	      f'    mkdir -p ./imgs; \n' \
	      f'fi \n'
    subprocess.run(cmd_str, shell=True)
