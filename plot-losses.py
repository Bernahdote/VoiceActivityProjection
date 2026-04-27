
path_body = "/Users/willemberner/Desktop/Exjobb/Losses/Body-loss.csv"
path_gaze = "/Users/willemberner/Desktop/Exjobb/Losses/Gaze-loss.csv"
path_FAUV = "/Users/willemberner/Desktop/Exjobb/Losses/FAUV-loss.csv"
path_base = "/Users/willemberner/Desktop/Exjobb/Losses/Baseline-loss.csv"
path_all = "/Users/willemberner/Desktop/Exjobb/Losses/All-loss.csv"

import matplotlib.pyplot as plt
import pandas as pd

df_body = pd.read_csv(path_body)
df_gaze = pd.read_csv(path_gaze)
df_FAUV = pd.read_csv(path_FAUV)
df_base = pd.read_csv(path_base)
df_all = pd.read_csv(path_all)

plt.figure(figsize=(10, 6))
plt.plot(df_base["trainer/global_step"], df_base["VAP: Baseline - val_loss"], label="Baseline", color="red")
plt.plot(df_all["trainer/global_step"], df_all["VAP: All Features  - val_loss"], label="All features", color="purple")
plt.plot(df_body["trainer/global_step"], df_body["VAP: Body Features - val_loss"], label="Body features", color="blue")
plt.plot(df_gaze["trainer/global_step"], df_gaze["VAP: Gaze Features - val_loss"], label="Gaze features", color="orange")
plt.plot(df_FAUV["trainer/global_step"], df_FAUV["VAP: FAUV features  - val_loss"], label="FAU features", color="green")
plt.xlabel("Step")
plt.ylabel("Validation loss")
plt.title("Validation loss curves for sets of features")
plt.legend()
plt.grid()
plt.show()
