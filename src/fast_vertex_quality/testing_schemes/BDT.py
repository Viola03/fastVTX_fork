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
        BDT_vars=[
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
        ],
        signal="datasets/Kee_2018_truthed_more_vars.csv",
        background="datasets/B2Kee_2018_CommonPresel.csv",
        signal_label="Train - sig",
        background_label="Train - comb",
        gen_track_chi2=True,
    ):

        self.signal_label = signal_label
        self.background_label = background_label

        self.BDT_vars = BDT_vars

        if gen_track_chi2:
            self.BDT_vars_gen = [
                x.replace("CHI2NDOF", "CHI2NDOF_gen") for x in self.BDT_vars
            ]
        else:
            self.BDT_vars_gen = self.BDT_vars

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

    def get_jpsiX_samples(
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

        lost_mass_cuts = [125, 250, 750, 1250]

        event_loader.select_randomly(Nevents=N * (len(lost_mass_cuts) + 1))
        if generate:

            event_loader = vertex_quality_trainer_obj.predict_from_data_loader(
                event_loader
            )

            query = event_loader.get_branches(
                self.BDT_vars_gen + ["lost_mass"], processed=False
            )

            queries = []
            for idx in range(len(lost_mass_cuts) + 1):
                if idx == 0:
                    query_i = query.query(f"lost_mass<{lost_mass_cuts[idx]}")
                elif idx < len(lost_mass_cuts):
                    query_i = query.query(
                        f"lost_mass<{lost_mass_cuts[idx]} and lost_mass>{lost_mass_cuts[idx-1]}"
                    )
                else:
                    query_i = query.query(f"lost_mass>{lost_mass_cuts[idx-1]}")

                query_i = np.squeeze(np.asarray(query_i[self.BDT_vars_gen]))
                queries.append(query_i)

        else:
            query = event_loader.get_branches(
                self.BDT_vars + ["lost_mass"], processed=False
            )

            queries = []
            for idx in range(len(lost_mass_cuts) + 1):
                if idx == 0:
                    query_i = query.query(f"lost_mass<{lost_mass_cuts[idx]}")
                elif idx < len(lost_mass_cuts):
                    query_i = query.query(
                        f"lost_mass<{lost_mass_cuts[idx]} and lost_mass>{lost_mass_cuts[idx-1]}"
                    )
                else:
                    query_i = query.query(f"lost_mass>{lost_mass_cuts[idx-1]}")

                query_i = np.squeeze(np.asarray(query_i[self.BDT_vars]))
                queries.append(query_i)

        return queries

    def make_BDT_plot(
        self,
        vertex_quality_trainer_obj,
        filename,
        include_combinatorial=False,
        include_jpsiX=False,
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

        samples = [signal_gen, prc_MC, prc_gen]
        labels = ["sig - gen", "prc - MC", "prc - gen"]
        colours = ["tab:blue", "tab:red", "tab:green", "tab:purple", "k"]

        if include_combinatorial:

            combi_gen = self.get_sample(
                "datasets/B2Kee_2018_CommonPresel_more_vars.csv",
                vertex_quality_trainer_obj,
                generate=True,
                N=10000,
            )

            colours.append("tab:orange")
            samples.append(combi_gen)
            labels.append("combi - gen")

        if include_jpsiX:

            jpsix_MC = self.get_jpsiX_samples(
                "datasets/JPSIX_2018_truthed_more_vars.csv",
                None,
                generate=False,
                N=50000,
            )
            jpsix_gen = self.get_jpsiX_samples(
                "datasets/JPSIX_2018_truthed_more_vars.csv",
                vertex_quality_trainer_obj,
                generate=True,
                N=50000,
            )

            scores = self.query_and_plot_samples_jpsiX(
                jpsix_MC,
                jpsix_gen,
                filename=filename.replace(".pdf", "") + "_jpsiX.pdf",
                include_combinatorial=False,
            )

            # colours.append("tab:orange")
            # samples.append(combi_gen)
            # labels.append("combi - gen")

        scores = self.query_and_plot_samples(
            samples,
            labels,
            colours=colours,
            filename=filename,
            include_combinatorial=include_combinatorial,
        )

        return scores

    def query_and_plot_samples_jpsiX(
        self,
        samples_MC,
        samples_gen,
        filename="BDT.pdf",
        kFold=0,
        include_combinatorial=False,
    ):

        sample_values_MC = {}
        sample_values_gen = {}

        clf = self.BDTs[kFold]["BDT"]

        for i in range(len(samples_MC)):
            sample_values_MC[f"MC_{i}"] = clf.predict_proba(samples_MC[i])[:, 1]
        for i in range(len(samples_gen)):
            sample_values_gen[f"gen_{i}"] = clf.predict_proba(samples_gen[i])[:, 1]
        colours = [
            "tab:blue",
            "tab:red",
            "tab:orange",
            "tab:green",
            "tab:purple",
        ]
        with PdfPages(f"{filename}") as pdf:

            plt.figure(figsize=(8, 8))

            n_points = 75

            x = np.linspace(0, 0.99, n_points)

            for ii in range(5):

                colour = colours[ii]

                eff_true = np.empty(0)
                eff_fake = np.empty(0)

                for cut in x:

                    values = sample_values_MC[f"MC_{ii}"]
                    pass_i = np.shape(np.where(values > cut))[1]
                    eff_true = np.append(eff_true, pass_i / np.shape(values)[0])

                    values = sample_values_gen[f"gen_{ii}"]
                    pass_i = np.shape(np.where(values > cut))[1]
                    eff_fake = np.append(eff_fake, pass_i / np.shape(values)[0])

                plt.plot(x, eff_true, color=colour, linestyle="-")
                plt.plot(x, eff_fake, color=colour, linestyle="--")
                plt.fill_between(x, eff_true, eff_fake, color=colour, alpha=0.1)

            pdf.savefig(bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, 1)

            hist = plt.hist(
                sample_values_MC.values(),
                bins=25,
                color=colours,
                density=True,
                label=list(sample_values_MC.keys()),
                histtype="step",
                range=[0, 1],
                alpha=1.0,
            )
            hist = plt.hist(
                sample_values_gen.values(),
                bins=25,
                color=colours,
                density=True,
                label=list(sample_values_gen.keys()),
                histtype="step",
                range=[0, 1],
                alpha=0.5,
            )
            plt.xlabel(f"BDT output")
            plt.yscale("log")
            plt.subplot(2, 3, 2)
            hist = plt.hist(
                sample_values_MC.values(),
                bins=25,
                color=colours,
                density=True,
                label=list(sample_values_MC.keys()),
                histtype="step",
                range=[0, 1],
                alpha=1.0,
            )
            hist = plt.hist(
                sample_values_gen.values(),
                bins=25,
                color=colours,
                density=True,
                label=list(sample_values_gen.keys()),
                histtype="step",
                range=[0, 1],
                alpha=0.5,
            )
            plt.legend(loc="upper left")
            plt.xlabel(f"BDT output")

            pdf.savefig(bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, 1)
            plt.title("MC")

            hist = plt.hist(
                sample_values_MC.values(),
                bins=25,
                color=colours,
                density=True,
                label=list(sample_values_MC.keys()),
                histtype="step",
                range=[0, 1],
                alpha=1.0,
            )
            plt.ylim(0.05, 16)
            plt.xlabel(f"BDT output")
            plt.yscale("log")
            plt.subplot(2, 3, 2)
            plt.title("MC")
            hist = plt.hist(
                sample_values_MC.values(),
                bins=25,
                color=colours,
                density=True,
                label=list(sample_values_MC.keys()),
                histtype="step",
                range=[0, 1],
                alpha=1.0,
            )
            plt.ylim(0.05, 16)
            plt.legend(loc="upper left")
            plt.xlabel(f"BDT output")

            pdf.savefig(bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, 1)
            plt.title("GEN")

            hist = plt.hist(
                sample_values_gen.values(),
                bins=25,
                color=colours,
                density=True,
                label=list(sample_values_gen.keys()),
                histtype="step",
                range=[0, 1],
                alpha=1.0,
            )
            plt.ylim(0.05, 16)
            plt.xlabel(f"BDT output")
            plt.yscale("log")
            plt.subplot(2, 3, 2)
            plt.title("GEN")
            hist = plt.hist(
                sample_values_gen.values(),
                bins=25,
                color=colours,
                density=True,
                label=list(sample_values_gen.keys()),
                histtype="step",
                range=[0, 1],
                alpha=1.0,
            )
            plt.ylim(0.05, 16)
            plt.legend(loc="upper left")
            plt.xlabel(f"BDT output")

            pdf.savefig(bbox_inches="tight")
            plt.close()

            for ii in range(5):

                plt.figure(figsize=(15, 10))
                plt.subplot(2, 3, 1)

                hist = plt.hist(
                    sample_values_MC[f"MC_{ii}"],
                    bins=25,
                    color=colours[ii],
                    density=True,
                    label=f"MC_{ii}",
                    histtype="step",
                    range=[0, 1],
                    alpha=1.0,
                )
                hist = plt.hist(
                    sample_values_gen[f"gen_{ii}"],
                    bins=25,
                    color=colours[ii],
                    density=True,
                    label=f"gen_{ii}",
                    histtype="step",
                    range=[0, 1],
                    alpha=0.5,
                )
                plt.xlabel(f"BDT output")
                plt.yscale("log")
                plt.subplot(2, 3, 2)
                hist = plt.hist(
                    sample_values_MC[f"MC_{ii}"],
                    bins=25,
                    color=colours[ii],
                    density=True,
                    label=f"MC_{ii}",
                    histtype="step",
                    range=[0, 1],
                    alpha=1.0,
                )
                hist = plt.hist(
                    sample_values_gen[f"gen_{ii}"],
                    bins=25,
                    color=colours[ii],
                    density=True,
                    label=f"gen_{ii}",
                    histtype="step",
                    range=[0, 1],
                    alpha=0.5,
                )
                plt.legend(loc="upper left")
                plt.xlabel(f"BDT output")

                pdf.savefig(bbox_inches="tight")
                plt.close()

        scores = None
        return scores

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

            n_points = 50

            effs = {}
            x = np.linspace(0, 0.99, n_points)

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

            pairs = [[0, 1], [2, 3]]
            if include_combinatorial:
                pairs.append([4, 5])

            scores = []

            colors = ["tab:blue", "tab:red", "tab:orange"]
            for idx, pair in enumerate(pairs):
                true = effs[list(effs.keys())[pair[0]]]
                false = effs[list(effs.keys())[pair[1]]]
                plt.fill_between(x, true, false, color=colors[idx], alpha=0.1)
                scores.append(np.sum(np.abs(true - false)) / n_points)

            plt.legend()
            plt.ylabel(f"Selection efficiency")
            plt.xlabel(f"BDT cut")

            # quit()

            pdf.savefig(bbox_inches="tight")
            plt.close()

        return scores
