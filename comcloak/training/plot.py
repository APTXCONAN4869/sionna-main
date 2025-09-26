import matplotlib.pyplot as plt
import pandas as pd

log = pd.read_csv("./comcloak/training/train_log_CSI3.txt", header=None, names=["iter", "ebno", "loss"])
plt.plot(log["iter"], log["loss"], label="loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("./comcloak/training/training_curve_CSI3.png")
plt.show()
