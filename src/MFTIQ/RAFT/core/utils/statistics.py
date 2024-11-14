import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib import gridspec
import scipy
import seaborn as sns
import pandas as pd

def mkdir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def mkdir_from_full_file_path_if_not_exist(path):
    basename = os.path.basename(path)
    mkdir_if_not_exist(path[:-len(basename)])

def bw_bilinear_interpolate_flow_numpy(im, flow):
    def _bw_bilin_interp(im, x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, im.shape[1] - 1)
        x1 = np.clip(x1, 0, im.shape[1] - 1)
        y0 = np.clip(y0, 0, im.shape[0] - 1)
        y1 = np.clip(y1, 0, im.shape[0] - 1)

        Ia = im[y0, x0]
        Ib = im[y1, x0]
        Ic = im[y0, x1]
        Id = im[y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        return wa * Ia + wb * Ib + wc * Ic + wd * Id

    ndim = im.ndim
    if ndim == 2:
        im = np.expand_dims(im, axis=2)
    H, W, C = im.shape
    X_g, Y_g = np.meshgrid(range(W), range(H))
    x, y = flow[:, :, 0], flow[:, :, 1]
    x = x + X_g
    y = y + Y_g
    im_w = []
    for i in range(C):
        im_w.append(_bw_bilin_interp(im[:, :, i], x, y))
    im_w = np.stack(im_w, axis=2)

    if ndim == 2:
        im_w = im_w[:, :, 0]
    return im_w

class EPEStatistics():
    def __init__(self, nbins, dataset_name=None, save_root=None):
        self.nbins = nbins
        self.dataset_name = dataset_name
        self.save_root = save_root
        self.thresholds = np.array(list(range(nbins+1))+[np.inf])

        self.hist_list = []

    def save(self):
        mkdir_if_not_exist(self.save_root)
        sum_hist = np.sum(np.array(self.hist_list), axis=0)
        save_path = os.path.join(self.save_root, self.dataset_name+'.txt')
        np.savetxt(save_path, sum_hist)

    def add_epe(self, epe):
        c_hist = np.histogram(epe, self.thresholds)
        self.hist_list.append(c_hist[0])


class FlowOUStatistics():

    def __init__(self, dataset_name, save_root=None, model_name=None):
        self.dataset_name = dataset_name
        self.save_root = save_root
        self.model_name = model_name

        self.flow_gt_list = []
        self.flow_epe_list = []
        self.sigma_list = []

        self.bins = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 15, 50, float('inf')]
        self.bin_accumulator = {str(k): [] for k in self.bins[:-1]}
        self.bin_accumulator_non_occl = {str(k): [] for k in self.bins[:-1]}
        self.bin_accumulator_occl = {str(k): [] for k in self.bins[:-1]}

        self.sigma_epe_bins = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, float('inf')]
        self.sigma_epe_bin_accumulator = {str(k): [] for k in self.sigma_epe_bins[:-1]}
        self.sigma_epe_bin_accumulator_non_occl = {str(k): [] for k in self.sigma_epe_bins[:-1]}
        self.sigma_epe_bin_accumulator_occl = {str(k): [] for k in self.sigma_epe_bins[:-1]}


    def __call__(self, *args, **kwargs):
        pass


    def single_box_plot(self, data, ticks_xlabels, title, xlabel='flow epe [px]', ylabel='sigma', save_path=None, **kwargs):
        figsize = kwargs.get('figsize', (14, 14))
        debug = kwargs.get('debug', False)
        ylim = (0, 10)
        a0_data = [len(d) for d in data]

        nans = [float('nan'), float('nan')]

        # fig, axes = plt.subplots(3, 1, sharex=True, figsize=figsize, height_ratios=[1, 3, 3])

        # create a figure
        fig = plt.figure()

        # to change size of subplot's
        # set height of each subplot as 8
        fig.set_figheight(figsize[0])

        # set width of each subplot as 8
        fig.set_figwidth(figsize[1])

        # create grid for different subplots
        spec = gridspec.GridSpec(ncols=1, nrows=3,
                                 hspace=0.1, height_ratios=[1, 4, 4])
        axes = [fig.add_subplot(spec[0])]

        print(len(ticks_xlabels), len(a0_data), len(data))

        fig.suptitle(title, fontsize=16)
        axes[0].bar(range(1, len(a0_data) + 1), a0_data)
        # axes[0].bar_label(a0_data)
        axes[0].set_yscale('log')
        axes[0].set_ylabel('frequency [-]')

        axes.append(fig.add_subplot(spec[1], sharex=axes[0]))
        axes[1].boxplot(data)
        axes[1].set_ylabel(ylabel)
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylim(ylim)
        axes[1].grid(which='major', color='#CCCCCC', linewidth=0.9)
        axes[1].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
        axes[1].set_xticklabels(ticks_xlabels)
        axes[1].minorticks_on()

        if debug:
            data_small = data
        else:
            data_small = [np.random.choice(d, len(d)//25) if len(d) > 10000 else d for d in data ]
        data_small = [d if len(d) >=1 else nans for d in data_small]

        axes.append(fig.add_subplot(spec[2], sharex=axes[1]))
        axes[2].violinplot(data_small, showmeans=False, showmedians=True, showextrema=True)
        axes[2].set_ylabel(ylabel)
        axes[2].set_xlabel(xlabel)
        axes[2].set_ylim(ylim)
        axes[2].grid(which='major', color='#CCCCCC', linewidth=0.9)
        axes[2].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
        axes[2].set_xticklabels(ticks_xlabels)
        axes[2].minorticks_on()

        # plt.tight_layout()
        # fig.set_tight_layout(True)
        spec.tight_layout(fig)

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close(fig)




    def save_graphs_single_type(self, bins, bin_accumulator, bin_accumulator_non_occl, bin_accumulator_occl, title_start, save_start, xlabel, ylabel, **kwargs):
        debug = kwargs.get('debug', False)
        box_plot_data = []
        box_plot_non_occl_data = []
        box_plot_occl_data = []

        bin_labels = []

        for idx in range(len(bins)-1):
            bin_start, bin_end = bins[idx], bins[idx+1]
            bin_labels.append(f'{bin_start}-{bin_end}')

            box_plot_data.append(np.concatenate(bin_accumulator[str(bin_start)]))
            box_plot_non_occl_data.append(np.concatenate(bin_accumulator_non_occl[str(bin_start)]))
            box_plot_occl_data.append(np.concatenate(bin_accumulator_occl[str(bin_start)]))


        self.single_box_plot(box_plot_data, bin_labels, f'{title_start} for {self.dataset_name} - all flows',
                             save_path=os.path.join(self.save_root, f'{save_start}_{self.dataset_name}_all_flows.png'),
                             xlabel=xlabel, ylabel=ylabel, debug=debug)

        self.single_box_plot(box_plot_non_occl_data, bin_labels, f'{title_start} for {self.dataset_name} - non-occluded flows',
                             save_path=os.path.join(self.save_root, f'{save_start}_{self.dataset_name}_non_occl_flows.png'),
                             xlabel=xlabel, ylabel=ylabel, debug=debug)
        try:
            self.single_box_plot(box_plot_occl_data, bin_labels, f'{title_start} for {self.dataset_name} - occluded only flows',
                         save_path=os.path.join(self.save_root, f'{save_start}_{self.dataset_name}_only_occl_flows.png'),
                             xlabel=xlabel, ylabel=ylabel, debug=debug)
        except:
            print('SOMETHING WRONG WITH PLOTTING')

    def save_graphs(self, debug=False):
        model_name = f'\n model: {self.model_name}' if self.model_name is not None else ''

        title_start = f'Distribution of sigma (epe bins){model_name}'
        save_start = 'dist_sigma_epe'
        xlabel = 'epe [px]'
        ylabel = 'sigma distribution [dont know, px]'
        self.save_graphs_single_type(self.bins, self.bin_accumulator, self.bin_accumulator_non_occl, self.bin_accumulator_occl,
                                     title_start, save_start, xlabel, ylabel, debug=debug)

        title_start = f'Distribution of epe (sigma bins){model_name}'
        save_start = 'dist_epe_sigma'
        xlabel = 'sigma [dont know, px]'
        ylabel = 'epe distribution [px]'
        # self.save_graphs_single_type(self.sigma_epe_bins, self.sigma_epe_bin_accumulator, self.sigma_epe_bin_accumulator_non_occl, self.sigma_epe_bin_accumulator_occl,
        #                              title_start, save_start, xlabel, ylabel, debug=debug)

    def torch2numpy(self, data):
        return data.detach().cpu().numpy().transpose(1,2,0)

    def add_data(self, flow_est, flow_gt, flow_valid, occl_est, occl_gt, uncertainty_est):
        occl_gt_np = self.torch2numpy(occl_gt)
        occl_valid_np = np.logical_and(occl_gt_np > 1, occl_gt_np < 254) * 1
        occl_est_np = self.torch2numpy(occl_est)
        occl_est_thresh_np = (occl_est_np > 0.5) * 1

        flow_valid_np = self.torch2numpy(torch.unsqueeze(flow_valid,0))

        uncertainty_est_np = self.torch2numpy(uncertainty_est)
        flow_gt_np = self.torch2numpy(flow_gt)
        flow_est_np = self.torch2numpy(flow_est)

        flow_ssd = np.sum((flow_gt_np - flow_est_np) ** 2, axis=2, keepdims=True)
        flow_epe = np.sqrt(flow_ssd)

        sigma2_np = np.sqrt(np.exp(uncertainty_est_np))
        sigma_np = np.sqrt(sigma2_np)

        # self.flow_gt_list.append(flow_gt_np)
        # self.flow_epe_list.append(flow_epe)
        # self.sigma_list.append(sigma_np)

        for idx in range(len(self.bins)-1):
            bin_start, bin_end = self.bins[idx], self.bins[idx+1]
            flow_mask = np.logical_and(flow_valid_np, np.logical_and(flow_epe >= bin_start, flow_epe < bin_end))
            c_sigma = sigma_np[flow_mask].flatten()
            self.bin_accumulator[str(bin_start)].append(c_sigma)

            flow_mask_non_occl = np.logical_and(occl_gt_np < 0.5, flow_mask)
            c_sigma = sigma_np[flow_mask_non_occl].flatten()
            self.bin_accumulator_non_occl[str(bin_start)].append(c_sigma)

            flow_mask_occl = np.logical_and(occl_gt_np > 0.5, flow_mask)
            c_sigma = sigma_np[flow_mask_occl].flatten()
            self.bin_accumulator_occl[str(bin_start)].append(c_sigma)

        for idx in range(len(self.sigma_epe_bins) - 1):
            bin_start, bin_end = self.sigma_epe_bins[idx], self.sigma_epe_bins[idx + 1]
            sigma_mask = np.logical_and(flow_valid_np, np.logical_and(sigma_np >= bin_start, sigma_np < bin_end))
            c_epe = flow_epe[sigma_mask].flatten()
            self.sigma_epe_bin_accumulator[str(bin_start)].append(c_epe)

            sigma_mask_non_occl = np.logical_and(occl_gt_np < 0.5, sigma_mask)
            c_epe = flow_epe[sigma_mask_non_occl].flatten()
            self.sigma_epe_bin_accumulator_non_occl[str(bin_start)].append(c_epe)

            sigma_mask_occl = np.logical_and(occl_gt_np >= 0.5, sigma_mask)
            c_epe = flow_epe[sigma_mask_occl].flatten()
            self.sigma_epe_bin_accumulator_occl[str(bin_start)].append(c_epe)


class OcclEPESigmaStatistics():

    def __init__(self, dataset_name, save_root=None, nbins=10, model_name=None, ):
        self.dataset_name = dataset_name
        self.save_root = save_root
        self.model_name = model_name
        self.nbins = nbins + 1

        self.flow_gt_list = []
        self.flow_est_list = []
        self.flow_epe_list = []
        self.sigma_list = []
        self.occl_gt_list = []
        self.occl_est_list = []

        self.bins = np.linspace(0., 1., self.nbins)
        self.bins[-1] = 1.001
        self.bin_flow_epe_accumulator = {str(k):[] for k in self.bins[:-1]}
        self.bin_sigma_accumulator = {str(k): [] for k in self.bins[:-1]}
        # self.bin_accumulator_occl = {str(k): [] for k in self.bins[:-1]}


    def __call__(self, *args, **kwargs):
        return self.bin_flow_epe_accumulator, self.bin_sigma_accumulator, self.bins


    def get_data(self):
        return {'flow_gt': self.flow_gt_list,
                'flow_est': self.flow_est_list,
                'flow_epe': self.flow_epe_list,
                'sigma': self.sigma_list,
                'occl_est': self.occl_est_list,
                'occl_gt': self.occl_gt_list}

    def single_box_plot(self, data, ticks_xlabels, title, xlabel='flow epe [px]', ylabel='sigma distribution [-]', save_path=None, **kwargs):
        figsize = kwargs.get('figsize', (14, 24))
        debug = kwargs.get('debug', False)
        ylim = (0, 10)
        a0_data = [len(d) for d in data]

        nans = [float('nan'), float('nan')]

        # fig, axes = plt.subplots(3, 1, sharex=True, figsize=figsize, height_ratios=[1, 3, 3])

        # create a figure
        fig = plt.figure()

        # to change size of subplot's
        # set height of each subplot as 8
        fig.set_figheight(figsize[0])

        # set width of each subplot as 8
        fig.set_figwidth(figsize[1])

        # create grid for different subplots
        spec = gridspec.GridSpec(ncols=1, nrows=3,
                                 hspace=0.1, height_ratios=[1, 4, 4])
        axes = [fig.add_subplot(spec[0])]

        print(len(ticks_xlabels), len(a0_data), len(data))

        fig.suptitle(title, fontsize=16)
        axes[0].bar(range(1, len(a0_data) + 1), a0_data)
        # axes[0].bar_label(a0_data)
        axes[0].set_yscale('log')
        axes[0].set_ylabel('frequency [-]')

        axes.append(fig.add_subplot(spec[1], sharex=axes[0]))
        axes[1].boxplot(data)
        axes[1].set_ylabel(ylabel)
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylim(ylim)
        axes[1].grid(which='major', color='#CCCCCC', linewidth=0.9)
        axes[1].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
        axes[1].set_xticklabels(ticks_xlabels)
        axes[1].minorticks_on()

        if debug:
            data_small = data
        else:
            data_small = [np.random.choice(d, len(d)//25) if len(d) > 10000 else d for d in data ]
        data_small = [d if len(d) >=1 else nans for d in data_small]

        axes.append(fig.add_subplot(spec[2], sharex=axes[1]))
        axes[2].violinplot(data_small, showmeans=False, showmedians=True, showextrema=True)
        axes[2].set_ylabel(ylabel)
        axes[2].set_xlabel(xlabel)
        axes[2].set_ylim(ylim)
        axes[2].grid(which='major', color='#CCCCCC', linewidth=0.9)
        axes[2].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
        axes[2].set_xticklabels(ticks_xlabels)
        axes[2].minorticks_on()

        # plt.tight_layout()
        # fig.set_tight_layout(True)
        spec.tight_layout(fig)

        if save_path is not None:
            mkdir_from_full_file_path_if_not_exist(save_path)
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close(fig)

    def save_graph_single_type(self, bin_accumulator, bins, title='Distribution of epe for sintel-final-valid', xlabel='occl est [-]', ylabel='flow epe distribution [px]', save_path=None):
        box_plot_data = []
        bin_labels = []


        for idx in range(len(bins)-1):
            bin_start, bin_end = bins[idx], bins[idx+1]

            print(bin_start, bin_end)
            bin_labels.append(f'{bin_start:.2f}-{bin_end:.2f}')

            box_plot_data.append(np.concatenate(bin_accumulator[str(bin_start)]))

        self.single_box_plot(box_plot_data, bin_labels, title=title, xlabel=xlabel, ylabel=ylabel, save_path=save_path)

    def save_graphs(self):
        model_name = f'\n model: {self.model_name}' if self.model_name is not None else ''
        self.save_graph_single_type(self.bin_flow_epe_accumulator, self.bins,
                                    title=f'Distribution of epe for {self.dataset_name}{model_name}',
                         save_path=os.path.join(self.save_root, f'{self.dataset_name}_occl_bins_epe.png'))
        self.save_graph_single_type(self.bin_sigma_accumulator, self.bins, ylabel='sigma distribution [-]',
                                    title=f'Distribution of sigma for {self.dataset_name}{model_name}',
                         save_path=os.path.join(self.save_root, f'{self.dataset_name}_occl_bins_sigma.png'))


    def torch2numpy(self, data):
        return data.detach().cpu().numpy().transpose(1,2,0)

    def add_data(self, flow_est, flow_gt, flow_valid, occl_est, occl_gt, uncertainty_est):
        occl_gt_np = self.torch2numpy(occl_gt)
        occl_valid_np = np.logical_and(occl_gt_np > 1, occl_gt_np < 254) * 1
        occl_est_np = self.torch2numpy(occl_est)
        occl_est_thresh_np = (occl_est_np > 0.5) * 1

        flow_valid_np = self.torch2numpy(torch.unsqueeze(flow_valid,0))

        uncertainty_est_np = self.torch2numpy(uncertainty_est)
        flow_gt_np = self.torch2numpy(flow_gt)
        flow_est_np = self.torch2numpy(flow_est)

        flow_ssd = np.sum((flow_gt_np - flow_est_np) ** 2, axis=2, keepdims=True)
        flow_epe = np.sqrt(flow_ssd)

        sigma2_np = np.sqrt(np.exp(uncertainty_est_np))
        sigma_np = np.sqrt(sigma2_np)

        self.flow_gt_list.append(flow_gt_np)
        self.flow_est_list.append(flow_est_np)
        self.flow_epe_list.append(flow_epe)
        self.sigma_list.append(sigma_np)
        self.occl_gt_list.append(occl_gt_np > 0)
        self.occl_est_list.append(occl_est_np)


        for idx in range(len(self.bins)-1):
            bin_start, bin_end = self.bins[idx], self.bins[idx+1]
            occl_mask = np.logical_and(flow_valid_np, np.logical_and(occl_est_np >= bin_start, occl_est_np < bin_end))

            c_flow_epe_occl = flow_epe[occl_mask].flatten()
            self.bin_flow_epe_accumulator[str(bin_start)].append(c_flow_epe_occl)

            c_sigma_occl = sigma_np[occl_mask].flatten()
            self.bin_sigma_accumulator[str(bin_start)].append(c_sigma_occl)




class OcclDistStatistics():

    def __init__(self, dataset_name, save_root=None, nbins=10, model_name=None):
        self.dataset_name = dataset_name
        self.save_root = save_root
        self.model_name = model_name
        self.nbins = nbins + 1

        self.flow_gt_list = []
        self.flow_est_list = []
        self.flow_epe_list = []
        self.sigma_list = []
        self.occl_gt_list = []
        self.occl_est_list = []

        self.bins = list(range(self.nbins))
        self.bins[1] = 0.99
        self.bins[-1] = np.inf

        self.bin_epe_occl_gt_accumulator = {str(k):[] for k in self.bins[:-1]}
        self.bin_epe_occl_est_thr050_accumulator = {str(k): [] for k in self.bins[:-1]}
        self.bin_epe_occl_est_thr005_accumulator = {str(k): [] for k in self.bins[:-1]}

        self.bin_sigma_occl_gt_accumulator = {str(k):[] for k in self.bins[:-1]}
        self.bin_sigma_occl_est_thr050_accumulator = {str(k): [] for k in self.bins[:-1]}
        self.bin_sigma_occl_est_thr005_accumulator = {str(k): [] for k in self.bins[:-1]}

        # self.bin_accumulator_occl = {str(k): [] for k in self.bins[:-1]}


    def __call__(self, *args, **kwargs):
        return self.bin_epe_occl_gt_accumulator, self.bin_sigma_occl_gt_accumulator,\
               self.bin_epe_occl_est_thr050_accumulator, self.bin_sigma_occl_est_thr050_accumulator,\
               self.bin_epe_occl_est_thr005_accumulator, self.bin_sigma_occl_est_thr005_accumulator,\
               self.bins


    # def get_data(self):
    #     return {'flow_gt': self.flow_gt_list,
    #             'flow_est': self.flow_est_list,
    #             'flow_epe': self.flow_epe_list,
    #             'sigma': self.sigma_list,
    #             'occl_est': self.occl_est_list,
    #             'occl_gt': self.occl_gt_list}

    def single_box_plot(self, data, ticks_xlabels, title, xlabel='flow epe [px]', ylabel='sigma distribution [-]', save_path=None, **kwargs):
        figsize = kwargs.get('figsize', (14, 24))
        debug = kwargs.get('debug', False)
        ylim = (0, 10)
        a0_data = [len(d) for d in data]

        nans = [float('nan'), float('nan')]

        # fig, axes = plt.subplots(3, 1, sharex=True, figsize=figsize, height_ratios=[1, 3, 3])

        # create a figure
        fig = plt.figure()

        # to change size of subplot's
        # set height of each subplot as 8
        fig.set_figheight(figsize[0])

        # set width of each subplot as 8
        fig.set_figwidth(figsize[1])

        # create grid for different subplots
        spec = gridspec.GridSpec(ncols=1, nrows=3,
                                 hspace=0.1, height_ratios=[1, 4, 4])
        axes = [fig.add_subplot(spec[0])]

        print(len(ticks_xlabels), len(a0_data), len(data))

        fig.suptitle(title, fontsize=16)
        axes[0].bar(range(1, len(a0_data) + 1), a0_data)
        # axes[0].bar_label(a0_data)
        axes[0].set_yscale('log')
        axes[0].set_ylabel('frequency [-]')

        axes.append(fig.add_subplot(spec[1], sharex=axes[0]))
        axes[1].boxplot(data)
        axes[1].set_ylabel(ylabel)
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylim(ylim)
        axes[1].grid(which='major', color='#CCCCCC', linewidth=0.9)
        axes[1].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
        axes[1].set_xticklabels(ticks_xlabels)
        axes[1].minorticks_on()

        if debug:
            data_small = data
        else:
            data_small = [np.random.choice(d, len(d)//25) if len(d) > 10000 else d for d in data ]
        data_small = [d if len(d) >=1 else nans for d in data_small]

        axes.append(fig.add_subplot(spec[2], sharex=axes[1]))
        axes[2].violinplot(data_small, showmeans=False, showmedians=True, showextrema=True)
        axes[2].set_ylabel(ylabel)
        axes[2].set_xlabel(xlabel)
        axes[2].set_ylim(ylim)
        axes[2].grid(which='major', color='#CCCCCC', linewidth=0.9)
        axes[2].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
        axes[2].set_xticklabels(ticks_xlabels)
        axes[2].minorticks_on()

        # plt.tight_layout()
        # fig.set_tight_layout(True)
        spec.tight_layout(fig)

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close(fig)

    def save_graph_single_type(self, bin_accumulator, bins, title='Distribution of epe for sintel-final-valid', xlabel='occl est [-]', ylabel='flow epe distribution [px]',
                               save_path=None):
        box_plot_data = []
        bin_labels = []

        for idx in range(len(bins) - 1):
            bin_start, bin_end = bins[idx], bins[idx + 1]

            print(bin_start, bin_end)
            bin_labels.append(f'{bin_start:.2f}-{bin_end:.2f}')

            box_plot_data.append(np.concatenate(bin_accumulator[str(bin_start)]))

        self.single_box_plot(box_plot_data, bin_labels, title=title, xlabel=xlabel, ylabel=ylabel, save_path=save_path)

    def save_graphs(self):
        model_name = f'\n model: {self.model_name}' if self.model_name is not None else ''

        self.save_graph_single_type(self.bin_epe_occl_gt_accumulator, self.bins,
                    xlabel='distance from gt occl [px]',
                    title=f'Distribution of epe on distance from occlusion for sintel-final-valid{model_name}',
                    save_path=os.path.join(self.save_root, f'{self.dataset_name}_occl_gt_bins_epe.png'))
        self.save_graph_single_type(self.bin_sigma_occl_gt_accumulator, self.bins,
                    ylabel='sigma distribution [-]', xlabel='distance from gt occl [px]',
                    title=f'Distribution of sigma on distance from occlusion for sintel-final-valid{model_name}',
                    save_path=os.path.join(self.save_root, f'{self.dataset_name}_occl_gt_bins_sigma.png'))

        self.save_graph_single_type(self.bin_epe_occl_est_thr050_accumulator, self.bins,
                    xlabel='distance from est occl (thr 0.5) [px]',
                    title=f'Distribution of epe on distance from estimated occlusion (thr=0.5) for sintel-final-valid{model_name}',
                    save_path=os.path.join(self.save_root, f'{self.dataset_name}_occl_est050_bins_epe.png'))
        self.save_graph_single_type(self.bin_sigma_occl_est_thr050_accumulator, self.bins,
                    ylabel='sigma distribution [-]', xlabel='distance from est occl (thr 0.5) [px]',
                    title=f'Distribution of epe on distance from estimated occlusion (thr=0.5) for sintel-final-valid{model_name}',
                    save_path=os.path.join(self.save_root, f'{self.dataset_name}_occl_est050_bins_sigma.png'))

        self.save_graph_single_type(self.bin_epe_occl_est_thr005_accumulator, self.bins,
                    xlabel='distance from est occl (thr 0.05) [px]',
                    title=f'Distribution of epe on distance from estimated occlusion (thr=0.05) for sintel-final-valid{model_name}',
                    save_path=os.path.join(self.save_root, f'{self.dataset_name}_occl_est005_bins_epe.png'))
        self.save_graph_single_type(self.bin_sigma_occl_est_thr005_accumulator, self.bins,
                    ylabel='sigma distribution [-]', xlabel='distance from est occl (thr 0.05) [px]',
                    title=f'Distribution of epe on distance from estimated occlusion (thr=0.05) for sintel-final-valid{model_name}',
                    save_path=os.path.join(self.save_root, f'{self.dataset_name}_occl_est005bins_sigma.png'))


    def torch2numpy(self, data):
        return data.detach().cpu().numpy().transpose(1,2,0)

    def add_data(self, flow_est, flow_gt, flow_valid, occl_est, occl_gt, uncertainty_est):
        occl_gt_np = self.torch2numpy(occl_gt)
        occl_valid_np = np.logical_and(occl_gt_np > 1, occl_gt_np < 254) * 1
        occl_est_np = self.torch2numpy(occl_est)
        occl_est_thr050_np = (occl_est_np > 0.5) * 1
        occl_est_thr005_np = (occl_est_np > 0.05) * 1

        flow_valid_np = self.torch2numpy(torch.unsqueeze(flow_valid,0))

        uncertainty_est_np = self.torch2numpy(uncertainty_est)
        flow_gt_np = self.torch2numpy(flow_gt)
        flow_est_np = self.torch2numpy(flow_est)

        flow_ssd = np.sum((flow_gt_np - flow_est_np) ** 2, axis=2, keepdims=True)
        flow_epe = np.sqrt(flow_ssd)

        sigma2_np = np.sqrt(np.exp(uncertainty_est_np))
        sigma_np = np.sqrt(sigma2_np)

        # self.flow_gt_list.append(flow_gt_np)
        # self.flow_est_list.append(flow_est_np)
        # self.flow_epe_list.append(flow_epe)
        # self.sigma_list.append(sigma_np)
        # self.occl_gt_list.append(occl_gt_np > 0)
        # self.occl_est_list.append(occl_est_np)


        occl_gt_dist = np.expand_dims(scipy.ndimage.distance_transform_edt(occl_gt_np[:,:,0] == 0), axis=2)
        occl_est_thr050_dist = np.expand_dims(scipy.ndimage.distance_transform_edt(occl_est_thr050_np[:,:,0] == 0), axis=2)
        occl_est_thr005_dist = np.expand_dims(scipy.ndimage.distance_transform_edt(occl_est_thr005_np[:,:,0] == 0), axis=2)

        for idx in range(len(self.bins)-1):
            bin_start, bin_end = self.bins[idx], self.bins[idx+1]

            occl_gt_mask = np.logical_and(flow_valid_np, np.logical_and(occl_gt_dist >= bin_start, occl_gt_dist < bin_end))
            c_flow_epe_occl = flow_epe[occl_gt_mask].flatten()
            self.bin_epe_occl_gt_accumulator[str(bin_start)].append(c_flow_epe_occl)
            c_sigma_occl = sigma_np[occl_gt_mask].flatten()
            self.bin_sigma_occl_gt_accumulator[str(bin_start)].append(c_sigma_occl)

            occl_est_thr050_mask = np.logical_and(flow_valid_np, np.logical_and(occl_est_thr050_dist >= bin_start, occl_est_thr050_dist < bin_end))
            c_flow_epe_occl = flow_epe[occl_est_thr050_mask].flatten()
            self.bin_epe_occl_est_thr050_accumulator[str(bin_start)].append(c_flow_epe_occl)
            c_sigma_occl = sigma_np[occl_est_thr050_mask].flatten()
            self.bin_sigma_occl_est_thr050_accumulator[str(bin_start)].append(c_sigma_occl)

            occl_est_thr005_mask = np.logical_and(flow_valid_np, np.logical_and(occl_est_thr005_dist >= bin_start, occl_est_thr005_dist < bin_end))
            c_flow_epe_occl = flow_epe[occl_est_thr005_mask].flatten()
            self.bin_epe_occl_est_thr005_accumulator[str(bin_start)].append(c_flow_epe_occl)
            c_sigma_occl = sigma_np[occl_est_thr005_mask].flatten()
            self.bin_sigma_occl_est_thr005_accumulator[str(bin_start)].append(c_sigma_occl)


class MultiflowEPESigmaStatistics():

    def __init__(self, dataset_name, save_root=None, ndelta=23, model_name=None):
        self.dataset_name = dataset_name
        self.save_root = save_root
        self.model_name = model_name
        self.ndelta = ndelta

        self.bins = list(range(1, self.ndelta))

        self.df_list = [] #pd.DataFrame(columns = ['Epe', 'Valid', 'Occl', 'FG', 'delta'])

    def __call__(self, *args, **kwargs):
        return pd.concat(self.df_list, axis=0)

    def single_box_plot(self, df, title, xlabel='flow epe [px]', ylabel='sigma distribution [-]', save_path=None, **kwargs):
        figsize = kwargs.get('figsize', (14, 24))
        debug = kwargs.get('debug', False)
        ylim = (0, 10)

        # create a figure
        fig = plt.figure()

        # to change size of subplot's
        # set height of each subplot as 8
        fig.set_figheight(figsize[0])

        # set width of each subplot as 8
        fig.set_figwidth(figsize[1])

        # create grid for different subplots
        spec = gridspec.GridSpec(ncols=1, nrows=3,
                                 hspace=0.1, height_ratios=[1, 4, 4])
        axes = [fig.add_subplot(spec[0])]

        fig.suptitle(title, fontsize=16)
        sns.countplot(data=df, hue='FG', x='delta', ax=axes[0])
        axes[0].set_yscale('log')
        axes[0].grid(which='major', color='#CCCCCC', linewidth=0.9)
        axes[0].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
        axes[0].minorticks_on()
        axes[0].set_ylabel('frequency [-]')

        axes.append(fig.add_subplot(spec[1], sharex=axes[0]))
        n_samples = 1000000
        df_sample = df.sample(n=n_samples, replace=False, random_state=1) if len(df) > n_samples else df
        sns.boxplot(data=df_sample, ax=axes[1], y='Epe', x='delta', width=0.90, hue='FG')
        axes[1].set_ylabel(ylabel)
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylim(ylim)
        axes[1].grid(which='major', color='#CCCCCC', linewidth=0.9)
        axes[1].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
        # axes[1].set_xticklabels(ticks_xlabels)
        axes[1].minorticks_on()

        axes.append(fig.add_subplot(spec[2], sharex=axes[1]))
        n_samples = 10000
        df_sample = df.sample(n=n_samples, replace=False, random_state=1) if len(df) > n_samples else df
        sns.violinplot(data=df_sample, y='Epe', split=True, hue='FG', x='delta', ax=axes[2], inner='quartile', showmeans=False, showmedians=True, showextrema=True, cut=0,
                       width=0.95, gridsize=50)
        axes[2].set_ylabel(ylabel)
        axes[2].set_xlabel(xlabel)
        axes[2].set_ylim(ylim)
        axes[2].grid(which='major', color='#CCCCCC', linewidth=0.9)
        axes[2].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
        # axes[2].set_xticklabels(ticks_xlabels)
        axes[2].minorticks_on()

        # plt.tight_layout()
        # fig.set_tight_layout(True)
        spec.tight_layout(fig)

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close(fig)

    def save_graphs(self):
        df = pd.concat(self.df_list, axis=0)
        rslt_df = df[df['Valid']]
        non_occl_df = rslt_df[rslt_df['Occl'] == False]
        occl_df = rslt_df[rslt_df['Occl'] == True]


        model_name = f'\n model: {self.model_name}' if self.model_name is not None else ''
        save_root = os.path.join('/datagrid/personal/neoral/RAFT_occl_uncertainty_outputs/base-sintel/statistics_graphs_kubric', os.path.basename(self.model_name)[:-4])
        mkdir_if_not_exist(save_root)
        save_path = os.path.join(save_root, f'{self.dataset_name}_epe_multi_orig_val_occl.png')
        self.single_box_plot(occl_df, f'Distribution of EPE for individual frames (occl) - {self.dataset_name}{model_name}', xlabel='delta [frames]', ylabel='epe distribution [px]',
                        save_path=save_path)

        save_path = os.path.join(save_root, f'{self.dataset_name}_epe_multi_orig_val_nonoccl.png')
        self.single_box_plot(non_occl_df, f'Distribution of EPE for individual frames (non-occl) - {self.dataset_name}{model_name}', xlabel='delta [frames]', ylabel='epe distribution [px]',
                        save_path=save_path)

        save_path = os.path.join(save_root, f'{self.dataset_name}_epe_multi_orig_val_all.png')
        self.single_box_plot(rslt_df, f'Distribution of EPE for individual frames (all) - {self.dataset_name}{model_name}', xlabel='delta [frames]', ylabel='epe distribution [px]',
                        save_path=save_path)


    def torch2numpy(self, data):
        return data.detach().cpu().numpy().transpose(1,2,0)

    def add_data(self, flow_est, flow_gt, flow_valid, occl_est, occl_gt, uncertainty_est, fg_mask=None, delta=None, correct_flow=False):
        occl_gt_np = self.torch2numpy(occl_gt)
        occl_valid_np = np.logical_and(occl_gt_np > 1, occl_gt_np < 254) * 1
        occl_est_np = self.torch2numpy(occl_est)
        occl_est_thresh_np = (occl_est_np > 0.5) * 1

        flow_valid_np = self.torch2numpy(torch.unsqueeze(flow_valid,0))

        uncertainty_est_np = self.torch2numpy(uncertainty_est)
        flow_gt_np = self.torch2numpy(flow_gt)
        flow_est_np = self.torch2numpy(flow_est)

        flow_ssd = np.sum((flow_gt_np - flow_est_np) ** 2, axis=2, keepdims=True)
        flow_epe = np.sqrt(flow_ssd)

        sigma2_np = np.sqrt(np.exp(uncertainty_est_np))
        sigma_np = np.sqrt(sigma2_np)

        occl_mask = np.logical_and(flow_valid_np, occl_gt_np > 0.5)
        non_occl_mask = np.logical_and(flow_valid_np, occl_gt_np <= 0.5)


        if fg_mask is not None and delta is not None:
            fg_mask_np = self.torch2numpy(fg_mask)

            # 'Epe', 'Valid', 'Occl', 'FG', 'delta'
            c_df = pd.DataFrame({'Epe': flow_epe.flatten(),
                                 'delta': delta*np.ones_like(flow_epe, dtype=int).flatten(),
                                 'Valid': flow_valid_np.flatten() > 0.5,
                                 'Occl': occl_mask.flatten(),
                                 'FG': fg_mask_np.flatten(),
                                 })

            # self.df = pd.concat([self.df, c_df], axis=0)
            self.df_list.append(c_df)


class CombinedStatistics():

    def __init__(self, dataset_name, save_root=None, ndelta=23, model_name=None, split=None, **kwargs):
        self.dataset_name = dataset_name
        self.save_root = save_root
        self.model_name = model_name
        self.ndelta = ndelta
        self.split = split if split is not None else self.ndelta > 1
        self.bins = list(range(1, self.ndelta))
        self.epe_sigma_bins = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 15, 50, float('inf')]
        self.sigma_epe_bins = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, float('inf')]

        self.df_list = [] #pd.DataFrame(columns = ['Epe', 'Valid', 'Occl', 'FG', 'delta'])
        self.debug = kwargs.get('debug')

    def __call__(self, *args, **kwargs):
        return pd.concat(self.df_list, axis=0)

    def single_plot(self, df, title=None, x='delta', y='Epe', hue='FG', xlabel='flow epe [px]', ylabel='sigma distribution [-]', save_path=None, **kwargs):
        figsize = kwargs.get('figsize', (14, 24))
        debug = kwargs.get('debug', False)
        split = kwargs.get('split', True)
        ylim = kwargs.get('ylim', (0, 10))
        ticks_xlabels = kwargs.get('ticks_xlabels', None)
        order = list(range(1,len(ticks_xlabels)+1)) if ticks_xlabels is not None else None

        color = None if split else plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

        # create a figure
        fig = plt.figure()

        # to change size of subplot's
        # set height of each subplot as 8
        fig.set_figheight(figsize[0])

        # set width of each subplot as 8
        fig.set_figwidth(figsize[1])

        # create grid for different subplots
        spec = gridspec.GridSpec(ncols=1, nrows=3,
                                 hspace=0.1, height_ratios=[1, 4, 4])
        axes = [fig.add_subplot(spec[0])]

        fig.suptitle(title, fontsize=16)
        sns.countplot(data=df, hue=hue, x=x, ax=axes[0], order=order, color=color)
        axes[0].set_yscale('log')
        axes[0].grid(which='major', color='#CCCCCC', linewidth=0.9)
        axes[0].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
        axes[0].minorticks_on()
        axes[0].set_ylabel('frequency [-]')

        axes.append(fig.add_subplot(spec[1], sharex=axes[0]))
        n_samples = 10000000
        df_sample = df.sample(n=n_samples, replace=False, random_state=1) if len(df) > n_samples else df
        sns.boxplot(data=df_sample, ax=axes[1], y=y, x=x, width=0.90, order=order, hue=hue, color=color)
        axes[1].set_ylabel(ylabel)
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylim(ylim)
        axes[1].grid(which='major', color='#CCCCCC', linewidth=0.9)
        axes[1].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
        if ticks_xlabels is not None:
            # axes[1].set_xticks(range(1,len(ticks_xlabels)+1),ticks_xlabels)
            axes[1].set_xticklabels(ticks_xlabels)
        axes[1].minorticks_on()

        axes.append(fig.add_subplot(spec[2], sharex=axes[1]))
        n_samples = 1000000
        df_sample = df.sample(n=n_samples, replace=False, random_state=1) if len(df) > n_samples else df
        sns.violinplot(data=df_sample, y=y, split=split, hue=hue, x=x, ax=axes[2], inner='quartile',
                       showmeans=False, showmedians=True, showextrema=True, order=order,
                       cut=0, width=0.95, gridsize=50, bw=.15, scale='width', color=color)
        axes[2].set_ylabel(ylabel)
        axes[2].set_xlabel(xlabel)
        axes[2].set_ylim(ylim)
        axes[2].grid(which='major', color='#CCCCCC', linewidth=0.9)
        axes[2].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
        if ticks_xlabels is not None:
            # axes[2].set_xticks(range(len(ticks_xlabels)+1))
            axes[2].set_xticklabels(ticks_xlabels)
        axes[2].minorticks_on()

        # plt.tight_layout()
        # fig.set_tight_layout(True)
        spec.tight_layout(fig)

        if save_path is not None and not debug:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close(fig)

    def save_graph_single_type(self, df, occl_title='all', **kwargs):
        split = kwargs.get('split', self.split)
        hue = 'FG' if split else None
        epe_sigma_bins = kwargs.get('epe_sigma_bins', self.epe_sigma_bins)
        sigma_epe_bins = kwargs.get('sigma_epe_bins', self.sigma_epe_bins)
        epe_sigma_ticks = self.create_ticks_form_bins(epe_sigma_bins)
        sigma_epe_ticks = self.create_ticks_form_bins(sigma_epe_bins)
        debug = kwargs.get('debug', False)

        if not debug:
            mkdir_if_not_exist(os.path.join(self.save_root, self.model_name))
        model_name = f'\n model: {self.model_name}' if self.model_name is not None else ''
        # EPE in relation to DELTA (only for kubric)
        if self.ndelta > 1:
            save_path = os.path.join(self.save_root, self.model_name, f'{self.dataset_name}_epe_multi_{occl_title}.png')
            title = f'Distribution of EPE for individual frames ({occl_title}) - Kubric validation{model_name}'
            xlabel = 'delta [frames]'
            ylabel = 'epe distribution [px]'
            self.single_plot(df, x='delta', y='Epe', hue=hue, split=True,
                             title=title, xlabel=xlabel, ylabel=ylabel, save_path=save_path)

        for delta in range(1,self.ndelta+1):
            delta_df = df[df['delta']==delta]
            save_path = os.path.join(self.save_root, self.model_name, f'{self.dataset_name}_distribution_sigma_epe_{occl_title}_delta_{delta:02d}.png')
            title = f'Distribution of sigma (epe bins), delta {delta:02} ({occl_title}){model_name}'
            xlabel = 'epe [px]'
            ylabel = 'sigma distribution [px]'
            self.single_plot(delta_df, x='epe_sigma_bin', y='Sigma', hue=hue, split=split, ticks_xlabels=epe_sigma_ticks,
                             title=title, xlabel=xlabel, ylabel=ylabel, save_path=save_path)

            save_path = os.path.join(self.save_root, self.model_name, f'{self.dataset_name}_distribution_epe_sigma_{occl_title}_delta_{delta:02d}.png')
            title = f'Distribution of epe (sigma bins), delta {delta:02} ({occl_title}){model_name}'
            xlabel = 'sigma [px]'
            ylabel = 'epe distribution [px]'
            self.single_plot(delta_df, x='sigma_epe_bin', y='Epe', hue=hue, split=split, ticks_xlabels=sigma_epe_ticks,
                             title=title, xlabel=xlabel, ylabel=ylabel, save_path=save_path)

    def create_ticks_form_bins(self, bins):
        bin_labels = []
        for idx in range(len(bins) - 1):
            bin_start, bin_end = bins[idx], bins[idx + 1]
            bin_labels.append(f'{bin_start:.2f}-{bin_end:.2f}')
        return bin_labels

    def save_graphs(self, **kwargs):
        df = pd.concat(self.df_list, axis=0)
        epe_sigma_bins = kwargs.get('epe_sigma_bins', self.epe_sigma_bins)
        sigma_epe_bins = kwargs.get('sigma_epe_bins', self.sigma_epe_bins)
        epe_sigma_ticks = self.create_ticks_form_bins(epe_sigma_bins)
        sigma_epe_ticks = self.create_ticks_form_bins(sigma_epe_bins)
        df['epe_sigma_bin'] = np.searchsorted(epe_sigma_bins, df['Epe'])
        df['sigma_epe_bin'] = np.searchsorted(sigma_epe_bins, df['Sigma'])

        rslt_df = df[df['Valid']]
        non_occl_df = rslt_df[rslt_df['Occl_gt'] == 0]
        occl_df = rslt_df[rslt_df['Occl_gt'] == 1]

        self.save_graph_single_type(rslt_df, occl_title='all', **kwargs)
        self.save_graph_single_type(occl_df, occl_title='occl', **kwargs)
        self.save_graph_single_type(non_occl_df, occl_title='non_occl', **kwargs)

    def torch2numpy(self, data):
        return data.detach().cpu().numpy().transpose(1,2,0)

    def add_data(self, flow_est, flow_gt, flow_valid, occl_est, occl_gt, uncertainty_est, fg_mask=None, delta=None):
        occl_gt_np = self.torch2numpy(occl_gt)
        # occl_valid_np = np.logical_and(occl_gt_np > 1, occl_gt_np < 254) * 1
        occl_est_np = self.torch2numpy(occl_est)
        # occl_est_thresh_np = (occl_est_np > 0.5) * 1
        occl_est_thr050_np = (occl_est_np > 0.5) * 1
        occl_est_thr005_np = (occl_est_np > 0.05) * 1

        occl_gt_dist = np.expand_dims(scipy.ndimage.distance_transform_edt(occl_gt_np[:,:,0] == 0), axis=2)
        occl_est_thr050_dist = np.expand_dims(scipy.ndimage.distance_transform_edt(occl_est_thr050_np[:,:,0] == 0), axis=2)
        occl_est_thr005_dist = np.expand_dims(scipy.ndimage.distance_transform_edt(occl_est_thr005_np[:,:,0] == 0), axis=2)

        flow_valid_np = self.torch2numpy(flow_valid)

        uncertainty_est_np = self.torch2numpy(uncertainty_est)
        flow_gt_np = self.torch2numpy(flow_gt)
        flow_est_np = self.torch2numpy(flow_est)

        flow_ssd = np.sum((flow_gt_np - flow_est_np) ** 2, axis=2, keepdims=True)
        flow_epe = np.sqrt(flow_ssd)

        sigma2_np = np.sqrt(np.exp(uncertainty_est_np))
        sigma_np = np.sqrt(sigma2_np)

        # occl_mask = np.logical_and(flow_valid_np, occl_gt_np > 0.5)
        # non_occl_mask = np.logical_and(flow_valid_np, occl_gt_np <= 0.5)

        if fg_mask is not None and delta is not None:
            fg_mask_np = self.torch2numpy(fg_mask)
        else:
            fg_mask_np = -1 * np.ones_like(flow_epe, dtype=int)

        # 'Epe', 'Valid', 'Occl', 'FG', 'delta'
        c_df = pd.DataFrame({'Epe': flow_epe.flatten(),
                             'delta': delta*np.ones_like(flow_epe, dtype=int).flatten(),
                             'Valid': flow_valid_np.flatten() > 0.5,
                             'Occl_gt': occl_gt_np.flatten(),
                             'Occl_est': occl_est_np.flatten(),
                             'Occl_gt_dist': occl_gt_dist.flatten(),
                             'Occl_est_thr050_dist': occl_est_thr050_dist.flatten(),
                             'Occl_est_thr005_dist': occl_est_thr005_dist.flatten(),
                             'FG': fg_mask_np.flatten(),
                             'Sigma': sigma_np.flatten(),
                             })

        # self.df = pd.concat([self.df, c_df], axis=0)
        self.df_list.append(c_df)