#!/usr/bin/env python3

import sys, os, pandas as pnd, pyDOE, numpy as np


if len(sys.argv) < 2 :
    raise Exception("Need name of an .out files to store the outputs")
else :
    outfile=sys.argv[1]

NUM_PARAM = 9
NUM_SAMPLES = 1000
SMAX=3

# Perform Latin hypercube sampling (LHS) or simple Monte Carlo (MC) sampling
do_lhs = False
PARNAME=["D0","D1","D2","D3","D4","D5","D6","D7","D8"]
if do_lhs :
    SAMPLES = (2*pyDOE.lhs(NUM_PARAM, NUM_SAMPLES)-1)*0.015*SMAX
else :
    SAMPLES = np.random.uniform(size=(NUM_SAMPLES, NUM_PARAM)) * 0.015

# Create a DataFrame from the simulation results
DF = pnd.DataFrame((SAMPLES),columns=(PARNAME))
DF.to_pickle(outfile)
