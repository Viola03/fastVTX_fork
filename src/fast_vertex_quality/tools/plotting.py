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


def plot(data, gen_data, filename, Nevents=10000):

    print(f"Plotting {filename}.pdf....")

    data_all = data.get_branches(rd.targets + rd.conditions, processed=False)
    data_all_pp = data.get_branches(rd.targets + rd.conditions, processed=True)
    # data_physics = data.get_physics_variables()

    gen_data_all = gen_data.get_branches(rd.targets + rd.conditions, processed=False)
    gen_data_all_pp = gen_data.get_branches(rd.targets + rd.conditions, processed=True)
    # gen_data_all, gen_data_targets, gen_data_condtions = gen_data.get_physical_data()
    # gen_data_all_pp, gen_data_targets_pp, gen_data_condtions_pp = (
    #     gen_data.get_processed_data()
    # )
    # gen_data_physics = gen_data.get_physics_variables()

    columns = list(data_all.keys())
    N = len(columns)

    with PdfPages(f"{filename}.pdf") as pdf:

        for i in range(0, N):

            # print(data_all[columns[i]][:Nevents])
            # print(gen_data_all[columns[i]][:Nevents])
            # quit()
            if columns[i] in rd.targets:

                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.hist(
                    [
                        data_all[columns[i]][:Nevents],
                        np.asarray(gen_data_all[columns[i]])[:Nevents],
                    ],
                    bins=75,
                    histtype="step",
                    label=["truth", "gen"],
                )
                plt.xlabel(columns[i])
                plt.legend()
                plt.subplot(1, 2, 2)
                plt.hist(
                    [
                        data_all_pp[columns[i]][:Nevents],
                        np.asarray(gen_data_all_pp[columns[i]])[:Nevents],
                    ],
                    bins=75,
                    histtype="step",
                )
                plt.xlabel(columns[i])
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

    with PdfPages(f"{filename}_targets.pdf") as pdf:

        plt.figure(figsize=(5 * 5, 5 * 2))

        for subplot_idx, var in enumerate(rd.targets):

            plt.subplot(2, 5, subplot_idx + 1)
            hist = plt.hist(
                [data_all[var][:Nevents], gen_data_all[var][:Nevents]],
                bins=35,
                color=["tab:blue", "tab:red"],
                histtype="step",
            )

            plt.hist(data_all[var][:Nevents], bins=hist[1], alpha=0.5, color="tab:blue")
            plt.hist(
                gen_data_all[var][:Nevents],
                bins=hist[1],
                alpha=0.5,
                color="tab:red",
            )
            plt.xlabel(f"{var}")

        pdf.savefig(bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(5 * 5, 5 * 2))

        for subplot_idx, var in enumerate(rd.targets):

            plt.subplot(2, 5, subplot_idx + 1)
            hist = plt.hist(
                [data_all[var][:Nevents], gen_data_all[var][:Nevents]],
                bins=35,
                color=["tab:blue", "tab:red"],
                histtype="step",
            )
            plt.hist(data_all[var][:Nevents], bins=hist[1], alpha=0.5, color="tab:blue")
            plt.hist(
                gen_data_all[var][:Nevents],
                bins=hist[1],
                alpha=0.5,
                color="tab:red",
            )
            plt.xlabel(f"{var}")

            plt.yscale("log")

        pdf.savefig(bbox_inches="tight")
        plt.close()
