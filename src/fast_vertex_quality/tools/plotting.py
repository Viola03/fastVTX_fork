from fast_vertex_quality.tools.config import read_definition, rd

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import alexPlot

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


def plot(data, gen_data, filename, targets, Nevents=10000, offline=False):

    print(f"Plotting {filename}.pdf....")

    data_all = data.get_branches(targets, processed=False)
    data_all_pp = data.get_branches(targets, processed=True)

    gen_data_all = gen_data.get_branches(targets, processed=False)
    gen_data_all_pp = gen_data.get_branches(targets, processed=True)

    columns = list(data_all.keys())
    N = len(columns)

    # plot_targets(filename, data_all, gen_data_all, Nevents)

    with PdfPages(f"{filename}.pdf") as pdf:

        for column in targets:
            
            if offline:
                plt.figure(figsize=(8, 7))

                ax = plt.subplot(1, 1, 1)
                colours = ['tab:blue', 'tab:red']
                values = [
                        data_all[column][:Nevents],
                        np.asarray(gen_data_all[column])[:Nevents],
                    ]
                hist = plt.hist(
                    values,
                    bins=75,
                    alpha=0.25,
                    color=colours,
                    # label=["Training sample", "Generated sample"],
                    density=True,
                    histtype="stepfilled",
                )
                plt.yscale('log')
                ymin, ymax = ax.get_ylim()
                xmin, xmax = hist[1][0], hist[1][-1]
                alexPlot.plot_data(values, density=True, also_plot_hist=True, bins=75, color=colours, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, only_canvas=True, label=["Training sample", "Generated sample"])
                plt.xlim(xmin, xmax)
                plt.xlabel(column)
                plt.legend()

                pdf.savefig(bbox_inches="tight")
                plt.close()


            plt.figure(figsize=(8, 7))
            
            ax = plt.subplot(1, 1, 1)
            colours = ['tab:blue', 'tab:red']
            values = [
                    data_all_pp[column][:Nevents],
                    np.asarray(gen_data_all_pp[column])[:Nevents],
                ]
            hist = plt.hist(
                values,
                bins=75,
                alpha=0.25,
                color=colours,
                # label=["Training sample", "Generated sample"],
                density=True,
                histtype="stepfilled",
                range=[-1, 1],
            )
            # plt.yscale('log')
            ymin, ymax = ax.get_ylim()
            alexPlot.plot_data(values, density=True, also_plot_hist=True, bins=75, color=colours, xmin=-1, xmax=1, ymin=ymin, ymax=ymax, only_canvas=True, label=["Training sample", "Generated sample"])
            plt.xlim(-1, 1)
            plt.xlabel(f'Processed({column})')
            plt.legend()

            pdf.savefig(bbox_inches="tight")
            plt.close()
