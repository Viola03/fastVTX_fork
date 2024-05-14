# branches = training_data_loader.get_branches(conditions + ["file"], processed=False)
# branches_processed = training_data_loader.get_branches(
#     conditions + ["file"], processed=True
# )

# with PdfPages(f"conditions_processed.pdf") as pdf:
#     for condition in conditions:
#         print(condition)
#         plt.figure(figsize=(8, 8))
#         plt.subplot(2, 2, 1)
#         plt.hist(
#             [
#                 branches.query("file==0")[condition],
#                 branches.query("file==1")[condition],
#             ],
#             label=["sig", "prc"],
#             bins=50,
#             histtype="step",
#         )
#         plt.xlabel(condition)
#         plt.subplot(2, 2, 3)
#         plt.hist(
#             [
#                 branches_processed.query("file==0")[condition],
#                 branches_processed.query("file==1")[condition],
#             ],
#             label=["sig", "prc"],
#             bins=50,
#             histtype="step",
#         )
#         plt.xlabel(f"{condition} processed")

#         if condition == "DIRA_B":
#             plt.subplot(2, 2, 4)
#             plt.hist(
#                 [
#                     np.log(1.0 - branches.query("file==0")[condition]),
#                     np.log(1.0 - branches.query("file==1")[condition]),
#                 ],
#                 label=["sig", "prc"],
#                 bins=50,
#                 histtype="step",
#             )
#             plt.xlabel(condition)

#         plt.subplot(2, 2, 2)
#         plt.hist(
#             [
#                 np.log(branches.query("file==0")[condition]),
#                 np.log(branches.query("file==1")[condition]),
#             ],
#             label=["sig", "prc"],
#             bins=50,
#             histtype="step",
#         )
#         plt.xlabel(condition)
#         plt.legend()
#         pdf.savefig(bbox_inches="tight")
#         plt.close()
# quit()
