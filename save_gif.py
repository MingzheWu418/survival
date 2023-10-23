import imageio

gif_fig_list = []
for i in range(1, 201):
    gif_fig_list.append("./visualization_survival/BLCA/BLCA_MLP_100_PCs_pred_survival_" + str(i) + "_iters.png")
ims = [imageio.imread(f) for f in gif_fig_list]
imageio.mimwrite("./BLCA_MLP_100_PCs_pred_survival.gif", ims, fps=5)