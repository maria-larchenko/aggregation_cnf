import subprocess
import numpy as np

cmd_str = 'cd ./coagulation-diffusion-2d/run_folder/ \n' \
          'ls \n' \
          'make \n' \
          'wait \n' \
          '\n'
subprocess.run(cmd_str, shell=True)

step = 0.2
for i in range(1, 5):
    for j in range(1, 5):
        x = str(np.around(i*step, decimals=3))
        y = str(np.around(j*step, decimals=3))
        cmd_str = f'cd ./coagulation-diffusion-2d/run_folder/ \n' \
                  f'echo "`date +%Y_%m_%d_%H:%M:%S` Start tet.exe x={x} y={y} coag=0 v=1.0  angle=30" \n' \
                  f'./tet.exe {x} {y} 0 1.0 30 \n' \
                  f'wait \n' \
                  f'mv imgs "x={x} y={y} D=-0 v=1.0  angle=30" \n' \
                  f'if [ ! -d ./imgs ]; then \n' \
                  f'    mkdir -p ./imgs; \n' \
                  f'fi \n'
        subprocess.run(cmd_str, shell=True)
