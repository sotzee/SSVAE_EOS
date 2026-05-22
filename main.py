#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 13:19:03 2025

@author: alr
"""

import sys, os
import time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from VAE_EoS import VAE_EoS
import pandas as pd
from Parallel_process import main_parallel_unsave
import numpy as np
import itertools

single_run = False # change to false if you want mutliple runs
parameters=[2,0.01,2]

parallel_run = True # change to false if you don't want parallel
latent_dim_list = [1,2,3]
eta_list = [0.0001,0.001,0.01,0.1]
kappa_list = [1,2,5,10]

# eta_list = [0.001,0.01]
# kappa_list = [2,5]f

def main_single_run(parameters, other_args):
    latent_dim_variational, eta, kappa = parameters
    dirname, data_path = other_args

    user_name = os.popen("git config user.name").read().strip() or "user"
    outdir = f"latent{int(latent_dim_variational)}_eta{eta:.2g}_kappa{kappa:.2g}"
    run_path = f"{dirname}/{user_name}/{outdir}"

    pipeline = (
        VAE_EoS()
            .inputs(data_path)
            .hypers()
            .build_encoder()
            .build_decoder()
            .build_auto()
    )

    pipeline.eta = eta
    pipeline.latent_dim_variational = int(latent_dim_variational)
    pipeline.kappa = kappa

    start = time.time()
    pipeline.loop(outdir=run_path)
    end = time.time()

    runtime = end - start

    print("Finished training and evaluation.")
    mins, secs = divmod(runtime, 60)
    print(f"Total runtime: {int(mins)} min {secs:.2f} sec")

    return runtime

def main_multiple_runs(dirname,data_path):

    for latent_dim_variational in latent_dim_list:
        print(f"\n Latent dim {latent_dim_variational}: starting {len(eta_list) * len(kappa_list)} runs")

        for eta in eta_list:
            for kappa in kappa_list:
                print(f"\nTraining with latent_dim={latent_dim_variational}, eta={eta}, kappa={kappa}")

                runtime = main_single_run((latent_dim_variational, eta, kappa),[dirname, data_path])

                # Optionally log runtime summary here
                mins, secs = divmod(runtime, 60)
                print(f"[Summary] Run took {int(mins)} min {secs:.2f} sec")

        print(f"Latent dim {latent_dim_variational}: completed all runs")

def main_multiple_parallel_runs(dirname,data_path):
    user_name = os.popen("git config user.name").read().strip() or "user"
    parameters_array = np.array(list(itertools.product(latent_dim_list, eta_list, kappa_list)))
    
    timing=main_parallel_unsave(main_single_run,parameters_array,other_args=[dirname,data_path])
    timing=np.array(timing).reshape((len(latent_dim_list),len(eta_list),len(kappa_list)))
    return timing
    
if __name__ == "__main__":
    if(len(sys.argv) == 3):
        dirname = sys.argv[1]
        data_path = sys.argv[2]
    else:
        print("Input Error!!!")
        sys.exit(1)
    
    if single_run:
        main_single_run(parameters, [dirname, data_path])
    elif parallel_run:
        main_multiple_parallel_runs(dirname, data_path)
    else:
        main_multiple_runs(dirname, data_path)
