import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import alexPlot
import alexPlot.funcs.useful_functions as uf

def variable_plotter(pdf, variable, analyserA, analyserB, cut, labelA, labelB, title=None, log=False):

	alexPlot.plot_data([analyserA.data.query(cut).eval(variable), analyserB.data.query(cut).eval(variable)], density=True, also_plot_hist=True, bins=75, label=[labelA, labelB], only_canvas=True, pulls=True, log=log)
	if title: plt.title(title)
	plt.legend()
	pdf.savefig(bbox_inches="tight")
	plt.close()


def getBinomialEff(pass_sum, tot_sum, pass_sumErr, tot_sumErr):
		'''
		Function for computing efficiency (and uncertainty).
		'''
		eff = pass_sum/tot_sum # Easy part

		# Compute uncertainty taken from Eqs. (13) from LHCb-PUB-2016-021
		x = (1 - 2*eff)*(pass_sumErr*pass_sumErr)
		y = (eff*eff)*(tot_sumErr*tot_sumErr)

		effErr = np.sqrt(abs(x + y)/(tot_sum**2))

		return eff, effErr

def get_efficiency_as_a_function_of_variable(event_loader, cut, variable, variable_range=[]):


	if len(variable_range)>0:
		hist_pre = np.histogram(event_loader[variable], bins=50, range=variable_range)
	else:
		hist_pre = np.histogram(event_loader[variable], bins=50)

	event_loader_cut = event_loader.query(cut)

	hist_post = np.histogram(event_loader_cut[variable], bins=hist_pre[1])

	x = hist_pre[1][:-1]+(hist_pre[1][1]-hist_pre[1][0])/2.

	pass_tot_val = hist_post[0]
	gen_tot_val = hist_pre[0]
	pass_tot_err = np.sqrt(hist_post[0])
	gen_tot_err = np.sqrt(hist_pre[0])

	eff, effErr = getBinomialEff(pass_tot_val, gen_tot_val,
									pass_tot_err, gen_tot_err)
	
	return x, eff, effErr, pass_tot_val, gen_tot_val, pass_tot_err, gen_tot_err, hist_pre, hist_post


def plot_efficiency_as_a_function_of_variable(pdf, tuple_A, tuple_B, label_A, label_B, variable, cut, range_array, title, xlabel, signal, return_values=False):
		

		x, eff_A, effErr_A, pass_tot_val, gen_tot_val, pass_tot_err, gen_tot_err, hist_pre_A, hist_post_A = get_efficiency_as_a_function_of_variable(tuple_A, cut=cut, variable=variable, variable_range=range_array)
		plt.errorbar(x, eff_A, yerr=effErr_A,marker='o',fmt=' ',capsize=2,linewidth=1.75, markersize=8,alpha=1.,label=label_A, color='tab:blue')

		x, eff_B, effErr_B, pass_tot_val, gen_tot_val, pass_tot_err, gen_tot_err, hist_pre_B, hist_post_B = get_efficiency_as_a_function_of_variable(tuple_B, cut=cut, variable=variable, variable_range=range_array)
		plt.errorbar(x, eff_B, yerr=effErr_B,marker='o',fmt=' ',capsize=2,linewidth=1.75, markersize=8,alpha=1.,label=label_B, color='tab:orange')


		# plt.xlabel(xlabel)
		# plt.ylabel("BDT cut efficiency")
		# plt.title(title)
		# plt.legend()    
		# plt.ylim(0,1)
	
		# # pdf.savefig(bbox_inches="tight")
		plt.close()

		alexPlot.plot_points(x, y=[eff_A, eff_B], yerr=[effErr_A, effErr_B], density=True, also_plot_hist=True, label=[label_A, label_B], only_canvas=True, pulls=True)
		plt.ylim(ymin=0., ymax=1.)
		plt.ylabel("Efficiency")
		plt.title(f'{title}')
		plt.legend()
		pdf.savefig(bbox_inches="tight")
		plt.close()



		# ### ### ### ### ###
		# x = hist_post_A[1][:-1]+(hist_post_A[1][1]-hist_post_A[1][0])/2.
		# ax = plt.subplot(1,1,1)
		# plt.errorbar(x, hist_pre_A[0]/np.sum(hist_pre_A[0]), yerr=np.sqrt(hist_pre_A[0])/np.sum(hist_pre_A[0]),marker='o',fmt=' ',capsize=2,linewidth=1.75, markersize=8,alpha=0.25, color='tab:blue')
		# plt.errorbar(x, hist_post_A[0]/np.sum(hist_pre_A[0]), yerr=np.sqrt(hist_post_A[0])/np.sum(hist_pre_A[0]),marker='o',capsize=2,linewidth=1.75, markersize=8,alpha=1.,label=label_A, color='tab:blue')

		# plt.errorbar(x, hist_pre_B[0]/np.sum(hist_pre_B[0]), yerr=np.sqrt(hist_pre_B[0])/np.sum(hist_pre_B[0]),marker='o',fmt=' ',capsize=2,linewidth=1.75, markersize=8,alpha=0.25, color='tab:orange')
		# plt.errorbar(x, hist_post_B[0]/np.sum(hist_pre_B[0]), yerr=np.sqrt(hist_post_B[0])/np.sum(hist_pre_B[0]),marker='o',capsize=2,linewidth=1.75, markersize=8,alpha=1.,label=label_B, color='tab:orange')

		# plt.legend()
		# plt.xlabel(xlabel)
		# plt.title(title)
		# pdf.savefig(bbox_inches="tight")
		# plt.close()

		

		alexPlot.plot_points(x, y=[hist_post_A[0]/np.sum(hist_post_A[0]), hist_post_B[0]/np.sum(hist_post_B[0])], yerr=[np.sqrt(hist_post_A[0])/np.sum(hist_post_A[0]), np.sqrt(hist_post_B[0])/np.sum(hist_post_B[0])], density=True, also_plot_hist=True, label=[label_A, label_B], only_canvas=True, pulls=True)
		plt.ylim(ymin=0.)
		plt.title(f'{title} - Normalised lineshape')
		plt.ylabel("Event Density")
		plt.legend()
		pdf.savefig(bbox_inches="tight")
		plt.close()


def make_shuffle_impact_plots(pdf, events, events_alt, alt_label, bins=10, title=None, makeplot=True):

	num_bins = bins
	percentiles = np.percentile(events, np.linspace(0, 100, num_bins + 1))

	if makeplot:
		fig = plt.figure(figsize=(14,7))

		if title:
			fig.suptitle(title)

		plt.subplot(1, 2, 1)

	hist_events, bin_edges_events = np.histogram(events, bins=percentiles)
	hist_events_alt, _ = np.histogram(events_alt, bins=bin_edges_events)

	bin_midpoints = bin_edges_events[:-1] + np.diff(bin_edges_events) / 2.
	bin_widths = np.diff(bin_edges_events)
	
	hist_events_errors = uf.poisson_asym_errors(hist_events)
	hist_events_errors[np.where(hist_events_errors<0)] = 0.
	hist_events_alt_errors = uf.poisson_asym_errors(hist_events_alt)
	hist_events_alt_errors[np.where(hist_events_alt_errors<0)] = 0.

	if makeplot:
		plt.errorbar(bin_midpoints, (hist_events/np.sum(hist_events))*(1./bin_widths), yerr=(hist_events_errors/np.sum(hist_events))*(1./bin_widths), xerr=bin_widths / 2, marker='o',fmt=' ',capsize=2,linewidth=1.75, markersize=8, label='Default')
		plt.errorbar(bin_midpoints, (hist_events_alt/np.sum(hist_events_alt))*(1./bin_widths), yerr=(hist_events_alt_errors/np.sum(hist_events_alt))*(1./bin_widths), xerr=bin_widths / 2, marker='o',fmt=' ',capsize=2,linewidth=1.75, markersize=8, label=alt_label)
		plt.legend()
		plt.ylabel('Event Density')
		plt.xlabel('MOTHER_M')

	
		plt.subplot(1, 2, 2)

	ratio = np.divide(hist_events_alt/np.sum(hist_events_alt), hist_events/np.sum(hist_events), out=np.zeros_like(hist_events_alt, dtype=np.float64), where=hist_events != 0)
	ratio_err = ratio * np.sqrt((np.amax(hist_events_alt_errors,axis=0)/hist_events_alt)**2 + (np.amax(hist_events_errors,axis=0)/hist_events)**2)

	if makeplot:
		plt.axhline(y=1.,c='k')
		plt.errorbar(bin_midpoints, ratio, yerr=ratio_err, xerr=bin_widths / 2, fmt='o', capsize=2)
		plt.ylabel('Ratio (Shuffled/Default)')
		plt.xlabel('MOTHER_M')

		pdf.savefig(bbox_inches="tight")
		plt.close()

	# ratio[np.where(ratio<1.)] = 1./ratio[np.where(ratio<1.)]

	return bin_midpoints, ratio, ratio_err, bin_widths