import numpy as np
import matplotlib.pyplot as plt

surv_data = np.loadtxt("./patient_survival.txt")
surv_label = surv_data[:, 1]
surv_length = surv_data[:, 0]
print(surv_length)
print(surv_label)
# print(surv_data)
observed = surv_data[np.nonzero(surv_data[:, 1])][:, 0]
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