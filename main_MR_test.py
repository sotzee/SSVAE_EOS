import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path
from Parallel_process import main_parallel
from Calculation_MR import Calculation_MR
from read_data import read_data

latent_variational_sampling = False
  
# dir_name_list = ["VAE_data/alexR055"]
#dir_name_list = ["VAE_data/alexR055", "VAE_data/user"]
#dir_name_list = ["data_invertedcs2_logcc/sotzee"]
#dir_name_list = ["data_log/sotzee"]
#dir_name_list = ["data_log_bignet/sotzee"]
#dir_name_list = ['data_log_MmaxOnly/sotzee']

# latent_variational_sampling = False
# True:  use variational sampled latent layer as input for decoder
# False: use mean of variational parameter as input for decoder

import sys, os
user_name = os.popen("git config user.name").read().strip()

if(len(sys.argv) == 2):
    dir_name_list = [sys.argv[1]+'/'+user_name]
    data_path = './data/'
    latent_variational_sampling = False
elif(len(sys.argv) == 3):
    dir_name_list = [sys.argv[1]+'/'+user_name]
    data_path = sys.argv[2]
elif(len(sys.argv) == 4):
    dir_name_list = [sys.argv[1]+'/'+user_name]
    data_path = sys.argv[2]
    latent_variational_sampling = bool(sys.argv[3])
else:
    print("Input Error!!!")
    sys.exit(1)

#_,_, X_test, _,_, y_test_obs,_,_,_=read_data(data_path)
_,_, X_check, X_test, _,_, y_check_obs, y_test_obs,_,_,_ = read_data(data_path)
X_test=X_check
y_test_obs=y_check_obs

if (latent_variational_sampling):
    target_file = "hadronic_MR_sampling.dat"
else:
    target_file = "hadronic_MR_mean.dat"

print("dir_name_list: ", dir_name_list)
print("target_file: ", target_file)

for dir_name in dir_name_list:
    for folder_path in Path(dir_name).iterdir():
        if folder_path.is_dir():
            file_path = folder_path / target_file
            if file_path.exists():
                print(f"{target_file} existed in {folder_path}")
            else:
                print(f"{target_file} missing in {folder_path}")
    
                # import json
                # with open(dir_name+"/"+folder_name+"/config.json", "r") as f:
                #     config = json.load(f)
                # config_dict=json.dumps(config, indent=4)
                # print(config_dict)
                
                scaler_X = joblib.load(str(folder_path/"X_scaler.joblib")) # replace with file path
                #scaler_y = joblib.load(str(folder_path/"y_scaler.joblib")) # replace with file path

                X_test_scaled  = scaler_X.transform(X_test)
                
                import keras
                from VAE_EoS import Sampling
                from tensorflow import keras
                
                encoder = keras.models.load_model(str(folder_path/"encoder.keras"))
                decoder = keras.models.load_model(str(folder_path/"decoder.keras"))
                
                #vae_eos = keras.models.load_model("./vae_eos.keras")
                # print(vae_eos.summary())

                obs_test, mu_u_test, log_var_u_test, latent_sampled_test = encoder(X_test_scaled, training=False)
                if(latent_variational_sampling):
                    reconstructed_test_scaled = decoder(latent_sampled_test, training=False).numpy()
                else:
                    latent_mean_test=tf.concat([obs_test, mu_u_test], axis=1)
                    reconstructed_test_scaled = decoder(latent_mean_test, training=False).numpy()
                #reconstructed_test_original = np.exp(scaler_X.inverse_transform(reconstructed_test_scaled))

                reconstructed_test_original = np.exp(reconstructed_test_scaled)
                
                f_MRBIT=str(folder_path/target_file)
                error_log=str(folder_path/target_file)
                try:
                    result=main_parallel(Calculation_MR,reconstructed_test_original,f_MRBIT,error_log,verbose=1)
                except:
                    np.savetxt('./reconstructed_test_original_failed.txt',reconstructed_test_original)
                
    
