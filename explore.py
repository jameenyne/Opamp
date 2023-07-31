#!/usr/bin/env python3

import sys, os, numpy as np, pylab as plt, pandas as pnd, seaborn as sns, pyDOE
from sklearn import linear_model
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tabulate import tabulate
import pickle

#Loading the samples
pickle_file_path = "sample_LHS.out"
# Read the pickle file
with open(pickle_file_path, "rb") as f:
    SAMPLES = pickle.load(f).to_numpy()

# Check if an output file name is provided as a command-line argument
if len(sys.argv) < 2 :
    raise Exception("Need name of an .out files to store the outputs")
else :
    outfile=sys.argv[1]

VSTEP = 0.005

# Function to read waveform data from a SPICE file
def read_spice_file(wavefile) :
    f = open(wavefile, "r")
    lines = f.readlines()
    f.close()
    NROW = len(lines) - 2
    DATA = np.zeros(NROW)
    for line in lines[1:-1] :
        n = line.split()
        idx = int(n[0])
        DATA[idx] = float(n[1])
    return DATA

# Function to write circuit parameters to a SPICE file
def write_xyce_circuit(D) :
    f = open("x.ckt", "w")
    f.write(f"""* spice file for simple amplifier
.include opamp.lib
VDD VDD 0 0.9
VSS VSS 0 -0.9
VS IN 0 0.0
R1 IN NEG 1K
R2 OUT NEG 10K
X1 0 NEG OUT VDD VSS opamp PARAMS: D0={D[0]} D1={D[1]} D2={D[2]} D3={D[3]}
+    D4={D[4]} D5={D[5]} D6={D[6]} D7={D[7]} D8={D[8]}
.dc VS -0.9 0.9 {VSTEP}
.print dc v(OUT)
.end
""")
    f.close()

# Function to get performance metrics for a given XYCE circuit configuration
def get_perf(W) :
    write_xyce_circuit(W)
    res = os.system("C:\\\"Program Files\"\\\"Xyce 7.6 NORAD\"\\bin\\xyce  x.ckt > nul 2>&1")
    DATA = read_spice_file("x.ckt.prn")
    R = np.ptp(DATA)
    G = -np.diff(DATA)/VSTEP
    VM = np.argmax(G)*VSTEP-0.9
    return R, np.amax(G), VM

# Define constants and parameter names
NUM_PARAM = 9
NUM_SAMPLES = 1000
NUM_PERF = 3
PARNAME=["D0","D1","D2","D3","D4","D5","D6","D7","D8"]
PERFNAMES = ["Range", "Gain", "Center"]

# Set initial values for variables
RES = np.zeros((NUM_SAMPLES,NUM_PERF))

# Perform simulations for each sample
for i in range(NUM_SAMPLES) :
    RES[i,0],RES[i,1],RES[i,2] = get_perf(SAMPLES[i,:])

# Create a DataFrame with the sample configurations and performance metrics
DF = pnd.DataFrame(np.hstack((SAMPLES,RES)),columns=(PARNAME+PERFNAMES))

# Save the DataFrame to a pickle file
DF.to_pickle(outfile)

# Create a pair plot to visualize the relationships between input parameters and performance metrics
DF2 = DF.iloc[:, :9]
DF3 = DF.iloc[:, 9:12]
pairplot = sns.pairplot(DF,x_vars=DF2,y_vars=DF3)
pairplot.fig.suptitle("Latin hypercube sampling (LHS)")
pairplot.fig.tight_layout()

# Perform linear regression analysis for each performance metric
fig,axes = plt.subplots(nrows=1, ncols=3, figsize=(18,12))
raw_data = []
regr = linear_model.LinearRegression()
for i in range(NUM_PERF) :
    AX = axes[i%3]
    Y = RES[:,i]
    regr.fit(SAMPLES, Y)
    Y_HAT = regr.predict(SAMPLES)
    L = AX.plot(Y, Y_HAT, '.')
    ymin,ymax = np.amin(Y),np.amax(Y)
    AX.plot((ymin,ymax), (ymin,ymax), '--', color=L[-1].get_color(), lw=0.5)
    AX.set_title(PERFNAMES[i])
    BX = inset_axes(AX, width="40%", height="30%",  bbox_to_anchor=(-0.5, 0, 1, 1), bbox_transform=axes[i].transAxes)
    coefs = regr.coef_ / np.sum(np.abs(regr.coef_))
    raw_data.append(coefs.tolist())
    BX.bar(range(NUM_PARAM), coefs)

# Create a table with the regression coefficients
range_data = ['Range']+raw_data[0]
gain_data = ['Gain']+raw_data[1]
center_data = ['Center']+raw_data[2]
data = [range_data,gain_data,center_data]
col_names = ['Parameters',"D0","D1","D2","D3","D4","D5","D6","D7","D8"]
print(tabulate(data,headers=col_names,tablefmt='fancy_grid',showindex='always'))

# Adjust the layout and display the plots
plt.tight_layout()
plt.show()
