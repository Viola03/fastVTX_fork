from fast_vertex_quality.tools.config import read_definition, rd

from fast_vertex_quality.models.conditional_VAE import VAE_builder
import tensorflow as tf
import numpy as np
from fast_vertex_quality.tools.training import train_step
import fast_vertex_quality.tools.plotting as plotting
import pickle
import fast_vertex_quality.tools.data_loader as data_loader
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class BDT_tester:

    def __init__(
        self,
        transformers,
        tag,
        train=True,
        signal="datasets/Kee_2018_truthed_more_vars.csv",
        background="datasets/B2Kee_2018_CommonPresel.csv",
        signal_label="Train - sig",
        background_label="Train - comb",
    ):

        self.signal_label = signal_label
        self.background_label = background_label

        self.BDT_vars = [
            "B_plus_ENDVERTEX_CHI2",
            "B_plus_IPCHI2_OWNPV",
            "B_plus_FDCHI2_OWNPV",
            "B_plus_DIRA_OWNPV",
            "K_Kst_IPCHI2_OWNPV",
            "K_Kst_TRACK_CHI2NDOF",
            "e_minus_IPCHI2_OWNPV",
            "e_minus_TRACK_CHI2NDOF",
            "e_plus_IPCHI2_OWNPV",
            "e_plus_TRACK_CHI2NDOF",
        ]

        self.BDT_vars_gen = [
            x.replace("CHI2NDOF", "CHI2NDOF_gen") for x in self.BDT_vars
        ]

        self.transformers = transformers

        if train:
            event_loader_MC = data_loader.load_data(
                [
                    signal,
                ],
                transformers=self.transformers,
            )
            event_loader_MC.select_randomly(Nevents=50000)

            events_MC = event_loader_MC.get_branches(
                self.BDT_vars + ["kFold"], processed=False
            )

            event_loader_data = data_loader.load_data(
                [
                    background,
                ],
                transformers=self.transformers,
            )
            event_loader_data.select_randomly(Nevents=50000)

            events_data = event_loader_data.get_branches(
                self.BDT_vars + ["kFold"], processed=False
            )

            self.BDTs = {}

            for kFold in range(10):

                print(f"Training kFold {kFold}...")

                clf = GradientBoostingClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=4
                )

                events_data_i = events_data.query(f"kFold!={kFold}")
                events_MC_i = events_MC.query(f"kFold!={kFold}")

                events_data_i = events_data_i.drop("kFold", axis=1)
                events_MC_i = events_MC_i.drop("kFold", axis=1)

                real_training_data = np.squeeze(np.asarray(events_MC_i[self.BDT_vars]))

                fake_training_data = np.squeeze(
                    np.asarray(events_data_i[self.BDT_vars])
                )

                size = 25000
                real_training_data = real_training_data[:size]
                fake_training_data = fake_training_data[:size]

                real_training_labels = np.ones(size)

                fake_training_labels = np.zeros(size)

                total_training_data = np.concatenate(
                    (real_training_data, fake_training_data)
                )

                total_training_labels = np.concatenate(
                    (real_training_labels, fake_training_labels)
                )

                clf.fit(total_training_data, total_training_labels)

                self.BDTs[kFold] = {}
                self.BDTs[kFold]["BDT"] = clf

                break

            for kFold in range(10):

                clf = self.BDTs[kFold]["BDT"]

                events_data_i = events_data.query(f"kFold=={kFold}")
                events_MC_i = events_MC.query(f"kFold=={kFold}")

                events_data_i = events_data_i.drop("kFold", axis=1)
                events_MC_i = events_MC_i.drop("kFold", axis=1)

                real_testing_data = np.squeeze(np.asarray(events_MC_i[self.BDT_vars]))

                fake_testing_data = np.squeeze(np.asarray(events_data_i[self.BDT_vars]))

                size = 25000
                real_testing_data = real_testing_data[:size]
                fake_testing_data = fake_testing_data[:size]

                self.BDTs[kFold]["values_sig"] = clf.predict_proba(real_testing_data)[
                    :, 1
                ]

                self.BDTs[kFold]["values_bkg"] = clf.predict_proba(fake_testing_data)[
                    :, 1
                ]

                break

            pickle.dump(
                self.BDTs,
                open(
                    f"{tag}.pkl",
                    "wb",
                ),
            )

        else:

            self.BDTs = pickle.load(open(f"{tag}.pkl", "rb"))

    def get_BDT(self):

        return self.BDTs[0]["BDT"]

    def get_sample(
        self,
        sample_loc,
        vertex_quality_trainer_obj,
        generate,
        N=10000,
    ):

        event_loader = data_loader.load_data(
            [
                sample_loc,
            ],
            transformers=self.transformers,
        )

        event_loader.select_randomly(Nevents=N)
        if generate:

            event_loader = vertex_quality_trainer_obj.predict_from_data_loader(
                event_loader
            )

            query = event_loader.get_branches(self.BDT_vars_gen, processed=False)

            query = np.squeeze(np.asarray(query[self.BDT_vars_gen]))

        else:
            query = event_loader.get_branches(self.BDT_vars, processed=False)

            query = np.squeeze(np.asarray(query[self.BDT_vars]))

        return query

    def make_BDT_plot(
        self, vertex_quality_trainer_obj, filename, include_combinatorial=False
    ):
        signal_gen = self.get_sample(
            "datasets/Kee_2018_truthed_more_vars.csv",
            vertex_quality_trainer_obj,
            generate=True,
            N=10000,
        )

        prc_MC = self.get_sample(
            "datasets/Kstee_2018_truthed_more_vars.csv",
            None,
            generate=False,
            N=10000,
        )

        prc_gen = self.get_sample(
            "datasets/Kstee_2018_truthed_more_vars.csv",
            vertex_quality_trainer_obj,
            generate=True,
            N=10000,
        )

        if include_combinatorial:

            combi_gen = self.get_sample(
                "datasets/B2Kee_2018_CommonPresel_more_vars.csv",
                vertex_quality_trainer_obj,
                generate=True,
                N=10000,
            )

            self.query_and_plot_samples(
                [signal_gen, prc_MC, prc_gen, combi_gen],
                ["sig - gen", "prc - MC", "prc - gen", "combi - gen"],
                colours=[
                    "tab:blue",
                    "tab:red",
                    "tab:green",
                    "tab:purple",
                    "k",
                    "tab:orange",
                ],
                filename=filename,
                include_combinatorial=include_combinatorial,
            )

        else:

            self.query_and_plot_samples(
                [signal_gen, prc_MC, prc_gen],
                ["sig - gen", "prc - MC", "prc - gen"],
                colours=["tab:blue", "tab:red", "tab:green", "tab:purple", "k"],
                filename=filename,
                include_combinatorial=include_combinatorial,
            )

    def query_and_plot_samples(
        self,
        samples,
        labels,
        colours=["tab:blue", "tab:red", "tab:green", "tab:purple", "k"],
        filename="BDT.pdf",
        kFold=0,
        include_combinatorial=False,
    ):

        sample_values = {}
        sample_values[self.signal_label] = self.BDTs[kFold]["values_sig"]
        sample_values[self.background_label] = self.BDTs[kFold]["values_bkg"]

        clf = self.BDTs[kFold]["BDT"]

        for idx, sample in enumerate(samples):
            # print(idx, sample, np.where(np.isinf(sample)), np.where(np.isnan(sample)))
            sample_values[labels[idx]] = clf.predict_proba(sample)[:, 1]

        with PdfPages(f"{filename}") as pdf:

            plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, 1)

            hist = plt.hist(
                sample_values.values(),
                bins=50,
                color=colours,
                alpha=0.25,
                label=list(sample_values.keys()),
                density=True,
                histtype="stepfilled",
                range=[0, 1],
            )
            plt.hist(
                sample_values.values(),
                bins=50,
                color=colours,
                density=True,
                histtype="step",
                range=[0, 1],
            )
            # plt.legend(loc="upper left")
            plt.xlabel(f"BDT output")
            plt.yscale("log")
            plt.subplot(2, 3, 2)
            plt.hist(
                sample_values.values(),
                bins=50,
                color=colours,
                alpha=0.25,
                label=list(sample_values.keys()),
                density=True,
                histtype="stepfilled",
                range=[0, 1],
            )
            plt.hist(
                sample_values.values(),
                bins=50,
                color=colours,
                density=True,
                histtype="step",
                range=[0, 1],
            )
            plt.legend(loc="upper left")
            plt.xlabel(f"BDT output")

            ax = plt.subplot(2, 3, 3)

            hist = plt.hist(
                sample_values.values(),
                bins=15,
                color=colours,
                alpha=0.25,
                label=list(sample_values.keys()),
                density=False,
                histtype="stepfilled",
                range=[0, 1],
            )
            plt.hist(
                sample_values.values(),
                bins=15,
                color=colours,
                density=False,
                histtype="step",
                range=[0, 1],
            )
            # plt.title("Samples may not be correctly scaled") # set_visible(False)
            # plt.legend(loc="upper left")
            plt.xlabel(f"BDT output")
            plt.yscale("log")
            ax.set_visible(False)

            plt.subplot(2, 3, 4)
            x = hist[1][:-1] + (hist[1][1] - hist[1][0]) / 2.0
            y = hist[0][0] / hist[0][3]
            yerr = y * np.sqrt(
                (np.sqrt(hist[0][0]) / hist[0][0]) ** 2
                + (np.sqrt(hist[0][3]) / hist[0][3]) ** 2
            )
            y *= np.sum(hist[0][3]) / np.sum(hist[0][0])
            plt.errorbar(
                x,
                y,
                yerr=yerr,
                label="MC",
                color="tab:blue",
                marker="o",
                fmt=" ",
                capsize=2,
                linewidth=1.75,
            )

            x = hist[1][:-1] + (hist[1][1] - hist[1][0]) / 2.0
            y = hist[0][2] / hist[0][4]
            yerr = y * np.sqrt(
                (np.sqrt(hist[0][2]) / hist[0][2]) ** 2
                + (np.sqrt(hist[0][4]) / hist[0][4]) ** 2
            )
            y *= np.sum(hist[0][4]) / np.sum(hist[0][2])
            plt.errorbar(
                x,
                y,
                yerr=yerr,
                label="gen",
                color="tab:red",
                marker="o",
                fmt=" ",
                capsize=2,
                linewidth=1.75,
            )

            plt.ylabel("Signal/prc")
            plt.xlabel(f"BDT output")
            plt.legend()
            plt.axhline(y=1, c="k")

            plt.subplot(2, 3, 5)

            effs = {}
            x = np.linspace(0, 0.99, 50)

            if include_combinatorial:
                sample_list = [
                    self.signal_label,
                    "sig - gen",
                    "prc - MC",
                    "prc - gen",
                    self.background_label,
                    "combi - gen",
                ]
            else:
                sample_list = [self.signal_label, "sig - gen", "prc - MC", "prc - gen"]

            for sample in sample_list:

                eff = np.empty(0)
                for cut in x:

                    values = sample_values[sample]

                    pass_i = np.shape(np.where(values > cut))[1]
                    eff = np.append(eff, pass_i / np.shape(values)[0])
                effs[sample] = eff

                if "sig" in sample or sample == self.signal_label:
                    color = "tab:blue"
                else:
                    color = "tab:red"
                if "gen" in sample:
                    style = "--"
                else:
                    style = "-"

                if "combi" in sample or sample == self.background_label:
                    color = "tab:orange"

                plt.plot(x, effs[sample], label=sample, color=color, linestyle=style)
            plt.legend()
            plt.ylabel(f"Selection efficiency")
            plt.xlabel(f"BDT cut")

            # quit()

            pdf.savefig(bbox_inches="tight")
            plt.close()
