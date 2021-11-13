from subprocess import Popen, PIPE, STDOUT
import matplotlib.ticker as mticker  

import matplotlib.pyplot as plt

import pandas as pd
import time
import os

input_sizes=list(range(1,6))
input_sizes=[int(element*1e6) for element in input_sizes]
print(input_sizes)
# exit()
deltas=[0.5,1,10,50,100]


zero_compressed_df = pd.DataFrame(columns = ['delta'+str(a) for a  in deltas])
cub_df = pd.DataFrame(columns = ['delta'+str(a) for a  in deltas])

zero_compressed_df.index.name='Input Size'
cub_df.index.name='Input Size'
for input_size in input_sizes:
    zero_compressed_arr=[]
    cub_arr=[]
    for delta in deltas:
        p = Popen([os.path.join('x64','Debug','0-Compressed-HP')], stdout=PIPE, stdin=PIPE, stderr=PIPE)
        strinput=str(input_size)+' '+str(int(input_size/delta))

        stdout_data = p.communicate(input=strinput.encode('utf-8'))
        
        timings=[x for x in stdout_data[0].decode().split('\n') if x.startswith('Took')]
        time_zero_compressed=float(timings[0].split()[1])
        time_cub=float(timings[1].split()[1])

        zero_compressed_arr.append(time_zero_compressed)        
        cub_arr.append(time_cub)
        print(time_cub/time_zero_compressed)
    zero_compressed_df.loc[input_size] = zero_compressed_arr  
    cub_df.loc[input_size] = cub_arr  



print("ONE HP\n",zero_compressed_df.head())
print("CubSort\n",cub_df.head())
timestr_safe = time.strftime("%Y-%m-%d-%H-%M-%S")
if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists(os.path.join('plots',timestr_safe)):
    os.makedirs(os.path.join('plots',timestr_safe))

for column in cub_df.columns:
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1fms'))
    scale_x = 1e3
    ticks_x = mticker.FuncFormatter(lambda x, pos: '{0:g}k'.format(x/scale_x))
    ax.xaxis.set_major_formatter(ticks_x)
    zero_compressed_df.plot(y=column,ax=ax,label='0-Compressed HP')
    delta_val=column.split('delta')[1]
    cub_df.plot(y=column, color='red', ax=ax,label='CUB-Sort')
    ax.set(title = "Runtime comparison with δ="+delta_val,
       xlabel = "Input Size",
       ylabel = "Runtime(ms)")

    plt.savefig(os.path.join('plots',timestr_safe,column+".png"), bbox_inches='tight',dpi=199)
    plt.clf()





zero_compressed_df.to_csv(
    os.path.join('plots',timestr_safe,"1-HP.csv")
    )
cub_df.to_csv(
    os.path.join('plots',timestr_safe,"Cubsort.csv")

)

