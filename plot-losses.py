
path_body = "/Users/willemberner/Desktop/Exjobb/Losses/Body-loss.csv"
path_gaze = "/Users/willemberner/Desktop/Exjobb/Losses/Gaze-loss.csv"
path_FAUV = "/Users/willemberner/Desktop/Exjobb/Losses/FAUV-loss.csv"

import matplotlib.pyplot as plt
import pandas as pd

df_body = pd.read_csv(path_body)
df_gaze = pd.read_csv(path_gaze)
df_FAUV = pd.read_csv(path_FAUV)

plt.figure(figsize=(10, 6))
plt.plot(df_body["trainer/global_step"], df_body["VAP: Body Features - val_loss"], label="Body Loss", color="blue")
plt.plot(df_gaze["trainer/global_step"], df_gaze["VAP: Gaze Features - val_loss"], label="Gaze Loss", color="orange")
plt.plot(df_FAUV["trainer/global_step"], df_FAUV["VAP: FAUV features  - val_loss"], label="FAUV Loss", color="green")
plt.xlabel("Step")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Curves for models")
plt.legend()
plt.grid()
plt.show()

