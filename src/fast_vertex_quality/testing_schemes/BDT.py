from fast_vertex_quality.tools.config import read_definition, rd

from fast_vertex_quality.models.conditional_VAE import VAE_builder
import tensorflow as tf
import numpy as np
from fast_vertex_quality.tools.training import train_step
import fast_vertex_quality.tools.plotting as plotting
import pickle
import fast_vertex_quality.tools.new_data_loader as data_loader
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
    ):

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

    def query_and_plot_samples(self, samples, labels, filename="BDT.pdf", kFold=0):

        sample_values = {}
        sample_values["Training - sig"] = self.BDTs[kFold]["values_sig"]
        sample_values["Training - bkg"] = self.BDTs[kFold]["values_bkg"]

        clf = self.BDTs[kFold]["BDT"]

        for idx, sample in enumerate(samples):
            sample_values[labels[idx]] = clf.predict_proba(sample)[:, 1]

        colours = ["tab:blue", "tab:red", "tab:green", "tab:purple", "k"]
        with PdfPages(f"{filename}") as pdf:

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)

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
            # plt.legend(loc="upper left")
            plt.xlabel(f"BDT output")
            plt.yscale("log")
            plt.subplot(1, 2, 2)
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
            pdf.savefig(bbox_inches="tight")
            plt.close()
