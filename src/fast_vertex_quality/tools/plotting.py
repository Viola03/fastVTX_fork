from fast_vertex_quality.tools.config import read_definition, rd

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def loss_plot(loss_list, reco_factor, kl_factor, filename="LOSSES.png"):

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(loss_list[:, 0], loss_list[:, 1])
    plt.ylabel("kl Loss")
    plt.subplot(2, 3, 2)
    plt.plot(loss_list[:, 0], loss_list[:, 2])
    plt.ylabel("reco Loss")
    plt.subplot(2, 3, 3)
    plt.plot(loss_list[:, 0], loss_list[:, 1])
    plt.ylabel("kl Loss")
    plt.yscale("log")
    plt.subplot(2, 3, 4)
    plt.plot(loss_list[:, 0], loss_list[:, 2])
    plt.ylabel("reco Loss")
    plt.yscale("log")
    plt.subplot(2, 3, 5)
    plt.plot(
        loss_list[:, 0], kl_factor * loss_list[:, 1] + reco_factor * loss_list[:, 2]
    )
    plt.ylabel("TOTAL Loss")
    plt.yscale("log")
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(filename, bbox_inches="tight")
    plt.close("all")


def plot_targets(filename, df_A, df_B, Nevents, df_C=None, df_D=None):

    with PdfPages(f"{filename}_targets.pdf") as pdf:

        plt.figure(figsize=(5 * 5, 5 * 2))

        for subplot_idx, var in enumerate(rd.targets):

            plt.subplot(2, 5, subplot_idx + 1)
            hist = plt.hist(
                [df_A[var][:Nevents], df_B[var][:Nevents]],
                bins=35,
                color=["tab:blue", "tab:red"],
                histtype="step",
                density=True,
            )

            plt.hist(
                df_A[var][:Nevents],
                bins=hist[1],
                alpha=0.5,
                color="tab:blue",
                density=True,
            )
            plt.hist(
                df_B[var][:Nevents],
                bins=hist[1],
                alpha=0.5,
                color="tab:red",
                density=True,
            )

            if df_C is not None and df_D is not None:
                plt.hist(
                    [df_C[var][:Nevents], df_D[var][:Nevents]],
                    bins=hist[1],
                    color=["tab:blue", "tab:red"],
                    histtype="step",
                    linestyle="dashed",
                    density=True,
                )

            plt.xlabel(f"{var}")

        pdf.savefig(bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(5 * 5, 5 * 2))

        for subplot_idx, var in enumerate(rd.targets):

            plt.subplot(2, 5, subplot_idx + 1)
            hist = plt.hist(
                [df_A[var][:Nevents], df_B[var][:Nevents]],
                bins=35,
                color=["tab:blue", "tab:red"],
                histtype="step",
                density=True,
            )
            plt.hist(
                df_A[var][:Nevents],
                bins=hist[1],
                alpha=0.5,
                color="tab:blue",
                density=True,
            )
            plt.hist(
                df_B[var][:Nevents],
                bins=hist[1],
                alpha=0.5,
                color="tab:red",
                density=True,
            )

            if df_C is not None and df_D is not None:
                plt.hist(
                    [df_C[var][:Nevents], df_D[var][:Nevents]],
                    bins=hist[1],
                    color=["tab:blue", "tab:red"],
                    histtype="step",
                    linestyle="dashed",
                    density=True,
                )

            plt.xlabel(f"{var}")

            plt.yscale("log")

        pdf.savefig(bbox_inches="tight")
        plt.close()


def plot(data, gen_data, filename, Nevents=10000):

    print(f"Plotting {filename}.pdf....")

    data_all = data.get_branches(rd.targets + rd.conditions + ["q2"], processed=False)
    data_all_pp = data.get_branches(rd.targets + rd.conditions + ["q2"], processed=True)
    # data_physics = data.get_physics_variables()
    data_all_pp["q2"] = data_all["q2"]

    gen_data_all = gen_data.get_branches(
        rd.targets + rd.conditions + ["q2"], processed=False
    )
    gen_data_all_pp = gen_data.get_branches(
        rd.targets + rd.conditions + ["q2"], processed=True
    )
    gen_data_all_pp["q2"] = gen_data_all["q2"]

    # gen_data_all, gen_data_targets, gen_data_condtions = gen_data.get_physical_data()
    # gen_data_all_pp, gen_data_targets_pp, gen_data_condtions_pp = (
    #     gen_data.get_processed_data()
    # )
    # gen_data_physics = gen_data.get_physics_variables()

    # data_all_highq2 = data_all.query("q2>15")
    # gen_data_all_highq2 = gen_data_all.query("q2>15")

    # data_all_lowq2 = data_all.query("q2<6")
    # gen_data_all_lowq2 = gen_data_all.query("q2<6")

    # data_all_pp_highq2 = data_all_pp.query("q2>15")
    # gen_data_all_pp_highq2 = gen_data_all_pp.query("q2>15")

    # data_all_pp_lowq2 = data_all_pp.query("q2<6")
    # gen_data_all_pp_lowq2 = gen_data_all_pp.query("q2<6")

    columns = list(data_all.keys())
    N = len(columns)

    plot_targets(filename, data_all, gen_data_all, Nevents)
    # plot_targets(
    #     filename + "_q2",
    #     data_all_lowq2,
    #     data_all_highq2,
    #     Nevents,
    #     gen_data_all_lowq2,
    #     gen_data_all_highq2,
    # )
    # plot_targets(
    #     filename + "_q2_pp",
    #     data_all_pp_lowq2,
    #     data_all_pp_highq2,
    #     Nevents,
    #     gen_data_all_pp_lowq2,
    #     gen_data_all_pp_highq2,
    # )

    with PdfPages(f"{filename}.pdf") as pdf:

        for column in rd.targets:

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.hist(
                [
                    data_all[column][:Nevents],
                    np.asarray(gen_data_all[column])[:Nevents],
                ],
                bins=75,
                histtype="step",
                label=["truth", "gen"],
            )
            plt.xlabel(column)
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.hist(
                [
                    data_all_pp[column][:Nevents],
                    np.asarray(gen_data_all_pp[column])[:Nevents],
                ],
                bins=75,
                histtype="step",
                range=[-1, 1],
            )
            plt.xlabel(column)
            pdf.savefig(bbox_inches="tight")
            plt.close()

        # for particle in ["K_Kst", "e_minus", "e_plus"]:

        #     plt.figure(figsize=(12, 16))
        #     idx = 0
        #     for i in range(0, N):

        #         if (
        #             "B_plus" in columns[i]
        #             or particle in columns[i]
        #             and columns[i] in rd.targets
        #         ):

        #             idx += 1

        #             plt.subplot(4, 3, idx)
        #             hist = plt.hist2d(
        #                 np.log10(data_physics[f"{particle}_P"][:Nevents]),
        #                 data_all_pp[columns[i]][:Nevents],
        #                 norm=LogNorm(),
        #                 bins=35,
        #                 cmap="Reds",
        #             )
        #             plt.xlabel(f"log {particle} P")
        #             plt.ylabel(columns[i])

        #             plt.subplot(4, 3, idx + 3)
        #             plt.hist2d(
        #                 np.log10(gen_data_physics[f"{particle}_P"][:Nevents]),
        #                 gen_data_all_pp[columns[i]][:Nevents],
        #                 norm=LogNorm(),
        #                 bins=[hist[1], hist[2]],
        #                 cmap="Blues",
        #             )
        #             plt.xlabel(f"log {particle} P")
        #             plt.ylabel(columns[i])

        #             if idx % 3 == 0 and idx > 0:
        #                 idx += 3

        #     pdf.savefig(bbox_inches="tight")
        #     plt.close()
