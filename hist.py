import numpy as np
import matplotlib.pyplot as plt

# surv_data = np.loadtxt("./patient_survival.txt")
# print(surv_data)
# observed = surv_data[np.nonzero(surv_data[:, 1])][:, 0]
# masked = surv_data[surv_data[:, 1] == 0][:, 0]
# print(observed)
# print(masked)
#
# # loss_observed_pred = train_loss[pred.astype(bool)]  # lower loss
# # # print(loss_observed_pred.shape)
# # loss_masked_pred = train_loss[np.logical_not(pred.astype(bool))]  # higher loss
# # # print(loss_masked_pred.shape)
# bins = np.logspace(-2, 3, 500)
# # # plt.xscale('log')
# # # plt.yscale('log')
# #
# plt.hist(observed, bins, alpha=0.5, color="blue", label="observed")
# plt.hist(masked, bins, alpha=0.5, color="red", label="masked")
# plt.show()
import pandas as pd

import seaborn as sns
sns.set_theme(style="ticks", palette="pastel")

# Load the example tips dataset
original_data = {0.0: [0.613, 0.634, 0.578, 0.561, 0.546, 0.832, 0.585, 0.744, 0.632, 0.756, 0.653, 0.622, 0.516, 0.529, 0.695],
                 0.2: [0.615, 0.641, 0.6, 0.567, 0.553, 0.83, 0.54, 0.748, 0.636, 0.689, 0.623, 0.622, 0.554, 0.532, 0.68],
                 0.4: [0.573, 0.56, 0.68, 0.571, 0.502, 0.799, 0.549, 0.767, 0.642, 0.644, 0.601, 0.599, 0.591, 0.534, 0.676],
                 0.6: [0.575, 0.54, 0.6, 0.668, 0.478, 0.782, 0.592, 0.746, 0.633, 0.77, 0.552, 0.52, 0.485, 0.556, 0.655],
                 0.8: [0.516, 0.454, 0.458, 0.781, 0.439, 0.673, 0.506, 0.53]}
original_arr = []
for key, item in original_data.items():
    print(key, item)
    for accuracy in item:
        original_arr.append([key, accuracy, 1])

proposed_data = {0.0: [0.635, 0.722, 0.492, 0.587, 0.535, 0.819, 0.57, 0.768, 0.636, 0.82, 0.631, 0.654, 0.573, 0.541, 0.647],
                 0.2: [0.63, 0.545, 0.519, 0.522, 0.52, 0.802, 0.5, 0.742, 0.619, 0.807, 0.551, 0.508, 0.574, 0.518, 0.592],
                 0.4: [0.541, 0.558, 0.531, 0.572, 0.513, 0.81, 0.514, 0.749, 0.624, 0.789, 0.552, 0.565, 0.549, 0.52, 0.561],
                 0.6: [0.555, 0.57, 0.554, 0.528, 0.528, 0.809, 0.501, 0.75, 0.65, 0.773, 0.654, 0.544, 0.495, 0.548, 0.61],
                 0.8: [0.559, 0.457, 0.466, 0.513, 0.778, 0.439, 0.637, 0.807, 0.604, 0.543, 0.532, 0.505, 0.58]}
for key, item in proposed_data.items():
    print(key, item)
    for accuracy in item:
        original_arr.append([key, accuracy, 0])
original_arr = np.array(original_arr)
print(original_arr)
tips = pd.DataFrame(original_arr, columns=["masked", "acc", "method"])
print(tips)

# Draw a nested boxplot to show bills by day and time
sns.boxplot(x="masked", y="acc",
            hue="method",
            data=tips)
sns.despine(offset=10, trim=True)
plt.show()