import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# ----------------------------
# 1. Load DataFrames
# ----------------------------
def read_data(data_dir):
    df_data_obs = pd.read_csv(
        data_dir+'data_obs.txt',
        sep=r'\s+',
        comment='#',
        names=['M_max', 'R_1.4']
    )

    # df_cs2 = pd.read_csv(
    #     data_dir+'data_cs2.txt',
    #     sep=r'\s+',
    #     comment='#',
    #     header=None
    # )
    
    chunk_files = sorted(Path(data_dir).glob("data_cs2_part*.txt"))
    df_cs2 = pd.concat([pd.read_csv(
        file,
        sep=r"\s+",
        comment="#",
        header=None) for file in chunk_files],ignore_index=True)

    column_names = ['nB_cc', 'eps_cc', 'pressure_cc',
                    'nB_max', 'eps_max', 'pressure_max']
    df_boundary = pd.read_csv(
        data_dir+'data_boundary.txt',
        sep=r'\s+',
        comment='#',
        header=None,
        names=column_names
    )

    boundary_selected = df_boundary[[]]
    observed_selected = df_data_obs[['M_max', 'R_1.4']]
#    observed_selected = df_data_obs[['M_max']]
    y_latent_known = pd.concat([observed_selected, boundary_selected], axis=1)

    # df_cs2 = 1.0 / df_cs2
    # data_full = pd.concat([df_cs2.reset_index(drop=True), df_boundary.reset_index(drop=True)], axis=1)
    # data_full.iloc[:, -4] = np.log(data_full.iloc[:, -4])

    data_full = pd.concat([df_cs2.reset_index(drop=True), df_boundary.reset_index(drop=True)], axis=1)
    data_full.iloc[:] = np.log(data_full.iloc[:])

    #print(f"Input data: {data_full}")

    # filter region of data
    logic_cs2 = np.array(df_cs2)[:,:101].max(axis=1)<1
    logic_Mmax = np.logical_and(np.array(y_latent_known)[:,0]>1.95,np.array(y_latent_known)[:,0]<2.5)
    logic_R14  = np.logical_and(np.array(y_latent_known)[:,1]>3*np.array(y_latent_known)[:,0]+4,np.array(y_latent_known)[:,1]<15)
    logic = np.logical_and(logic_cs2,np.logical_and(logic_Mmax, logic_R14))
    
    data_full = data_full[logic]
    y_latent_known = y_latent_known[logic]

    X = data_full.values.astype('float32')
    y_latent_known = y_latent_known.values.astype('float32')

    # ----------------------------
    # 3. Train / Validation / Test Split
    # ----------------------------
    X_train, X_temp, y_train_obs, y_temp_obs = train_test_split(
        X, y_latent_known, test_size=0.40, random_state=42
    )
    X_val,   X_temp, y_val_obs,   y_temp_obs = train_test_split(
        X_temp, y_temp_obs, test_size=0.50, random_state=42
    )
    X_test,  X_pick, y_test_obs,  y_pick_obs = train_test_split(
        X_temp, y_temp_obs, test_size=0.50, random_state=42
    )

    return X_train, X_val, X_pick, X_test, y_train_obs, y_val_obs, y_pick_obs, y_test_obs, df_data_obs, df_cs2, df_boundary


# data_dir='./data_skyrme/'
# #data_dir='./data_rmf/'
# if __name__ == "__main__":
#     read_data(data_dir)
