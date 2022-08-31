import json
import os
import warnings
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import mne
import mne.preprocessing
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoMinorLocator

matplotlib.use('QT5agg')


class Plotter:
    def __init__(self, **kwargs):
        """
        Class that manages the plotting of the figures, either using mne instances or normal data (see static methods).
        Some methods will work only for [Epoch / continuous data] (it will depend on the given data) or for [Evoked]
         data only. (some methods works with both of them).
        Since we are not working with continuous data some of the plotting are no more working with continuous data.

        :param data: mne instance [Epochs / Continuous / Evoked] to plot
        :param disable_plot: if True disable all plots and figure saving. if False plots and figure saving is allowed
        :param show: True in order to show the plotted figures. Ignored if disable_plot=True
        :param save_fig: True in order to save the figures. Ignored if disable_plot=True
        :param save_path: Path where to save the figures, if no path is given put in the default folder "./figures"

        :param plot_data: plot all the data.                                        Works with: [Epochs / Evoked] data.
        :param plot_psd: plot power spectral density.                               Works with [Epochs] data.
        :param plot_sensors: sensor location plot.                                  Works with: [Epochs / Evoked] data.
        :param plot_image: plot epochs image and ERP for a specific channel.        Works with [Epochs] data.
        :param split: True to split the plot_image depending on the stimulus type.
        :param plot_psd_topomap: plot power spectral density and topomap.           Works with [Epochs] data.
        :param plot_topo_image: plot epochs image on topomap.                       Works with [Epochs] data.
        :param plot_evoked_joint: plot ERPs data and with topomap on peaks time.    Works with [evoked] data.
        :param plot_projs_topomap: plot ERPs on topomap.                            Works with [evoked] data.
        :return: /
        """

        self.disable_plot = kwargs.get('disable_plot')
        self.show = kwargs.get('show')
        self.save_fig = kwargs.get('save_fig')
        self.save_path = kwargs.get('save_path', os.path.join(os.getcwd(), 'figures'))

        self.plot_data = kwargs.get('plot_data')
        self.plot_psd = kwargs.get('plot_psd')
        self.plot_sensors = kwargs.get('plot_sensors')
        self.plot_image = kwargs.get('plot_image')
        self.split = kwargs.get('split')
        self.plot_psd_topomap = kwargs.get('plot_psd_topomap')
        self.plot_topo_image = kwargs.get('plot_topo_image')
        self.plot_topo_map = kwargs.get('plot_topo_map')
        self.plot_evoked_joint = kwargs.get('plot_evoked_joint')
        self.plot_projs_topomap = kwargs.get('plot_projs_topomap')

        # Data information
        self._data: Union[mne.BaseEpochs, mne.io.BaseRaw, mne.EvokedArray, None] = None
        self._name = None
        self._decomposition = None
        self._shape = None
        self._nb_epoch = None
        self._nb_channel = None
        self._nb_point = None
        self._stim_type = kwargs.get('stim_type') if kwargs.get('stim_type') is not None else 'all_stim'

        # Matplotlib parameters setting
        matplotlib.rcParams['figure.dpi'] = 100  # All figures dpi, default 100
        matplotlib.use('QT5agg')  # matplotlib backends : matplotlib.use('module://backend_interagg')

    @property
    def stim_type(self):
        return self._stim_type

    @stim_type.setter
    def stim_type(self, stim_type: str):
        self._stim_type: str = stim_type.lower() if stim_type is not None else 'all_stim'

    def _set_param(self, **kwargs):
        """
        Method to set the plotting parameters if a change is needed.
        """
        self.disable_plot = kwargs.get('disable_plot', self.disable_plot)
        self.show = kwargs.get('show', self.show)
        self.save_fig = kwargs.get('save_fig', self.save_fig)
        self.save_path = kwargs.get('save_path', self.save_path)

        self.plot_data = kwargs.get('plot_data')
        self.plot_psd = kwargs.get('plot_psd')
        self.plot_image = kwargs.get('plot_image')
        self.split = kwargs.get('split')
        self.plot_sensors = kwargs.get('plot_sensors')
        self.plot_psd_topomap = kwargs.get('plot_psd_topomap')
        self.plot_topo_image = kwargs.get('plot_topo_image')
        self.plot_topo_map = kwargs.get('plot_topo_map')
        self.plot_evoked_joint = kwargs.get('plot_evoked_joint')
        self.plot_projs_topomap = kwargs.get('plot_projs_topomap')

    def set_data(self, data):
        """
        Method that reads the data and extract useful information for plotting.

        :param data: data to be plotted. Has to be mne instance [Epochs / Continuous / Evoked] and can also be a
                      projected data [ICA / PCA].
        :return: /
        """
        self._data: Union[mne.EvokedArray, mne.BaseEpochs] = data
        description = json.loads(self._data.info['description'])
        self._name = description['file_name']
        self._decomposition = description['decomposition']
        if isinstance(data, mne.BaseEpochs):
            self._shape = self._data.get_data().shape
            self._nb_epoch = self._shape[0]
            self._nb_channel = self._shape[1]
            self._nb_point = self._shape[2]
        elif isinstance(data, mne.EvokedArray):
            self._shape = self._data.data.shape
            self._nb_channel = self._shape[0]
            self._nb_point = self._shape[1]

    def del_data(self):
        """
        Safely delete the data at the end. This allow to perform a new plot on new data without causing a conflict.
        :return: /
        """
        self._data = None
        self._name = None
        self._decomposition = None
        self._shape = None
        self._nb_epoch = None
        self._nb_channel = None
        self._nb_point = None
        self._stim_type = 'all_stim'

    def plot(self, data, **kwargs):
        """
        Main method that manages the plotting.
        :param data: data to be plotted.
        :param kwargs: If not provided, will take the initial plotting preferences, if provided update the plotting
                        preferences.
        :return: /
        """
        if kwargs:
            self._set_param(**kwargs)
        if not self.disable_plot:
            self.set_data(data)

            self.data_plot()
            self.psd_plot()
            self.sensors_plot()
            self.image_plot()
            self.psd_topomap_plot()
            self.topo_image_plot()
            self.topo_map_plot()
            self.evoked_plot_joint()

        self.del_data()

    def save(self, fig, save_name):
        """
        This method is in change of the figures saving and hiding them if show=False.
        /!\\ The main saving format is .svg
        :param fig: Figure to save.
        :param save_name: Saving name of the figure.
        :return: /
        """
        if self.save_fig:
            path = os.path.join(self.save_path, save_name + '.svg')
            fig.savefig(path, bbox_inches='tight')
            print("Figure saved at : {:}".format(path))
        if not self.show:
            plt.close()

    @staticmethod
    def layout(case, fig=None, axes=None, freq=None):
        """
        Method that manages the layout (some figures have repeated layout)
        :param case:
        :param fig:
        :param axes:
        :param freq:
        :return: /
        """
        if case == 1:
            axes[0].set_title('PSD', fontsize=14)
            axes[1].set_title('PSD mean', fontsize=14)
            for ax in axes:
                ax.set_xlabel('Frequency [Hz]', fontsize=11)
                ax.set_ylabel('$\mu V/\sqrt{Hz}$')
                ax.set_ylim(bottom=0)
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.xaxis.grid(True, which='minor', linestyle='dotted', linewidth=0.4)
                ax.xaxis.grid(True, which='major', linestyle='dotted', linewidth=1)
                ax.yaxis.grid(True, which='major', linestyle='dotted', linewidth=0.4)

        if case == 2:
            fig.tight_layout(pad=2)
            for mode, ax in enumerate(axes):
                if mode == len(axes) - 1:
                    ax.set_xlabel("Time [index]")
                ax.set_ylabel("U(f={:.2f} [Hz])".format(freq[mode]))
                ax.yaxis.set_label_coords(-0.08, 0.5)
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.xaxis.grid(True, which='minor', linestyle='--', linewidth=0.4)
                ax.xaxis.grid(True, which='major', linestyle='-', linewidth=0.4)
                ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.4)

    def data_plot(self):
        """
        Plot the epoched or raw data, whether they are projected in ICA or PCA domain.
            saving name: subj_ID-evoked_data(-ICA or PCA)
        :return: /
        """
        if not self.plot_data:
            return

        if isinstance(self._data, mne.BaseEpochs):
            # group by: 'type', 'original', 'position'
            if self._decomposition == 'PCA' or self._decomposition == 'ICA':
                scaling = dict(eeg=40e-6) if self._decomposition == 'ICA' else dict(eeg=60e-6)
                color_choice = ['C1', 'C3', '#FFC419', 'C4', 'C7']
                event_color = {key: color_choice[i] for i, key, in enumerate(self._data.event_id.keys())}
                fig = self._data.plot(picks=None, events=self._data.events, event_id=self._data.event_id,
                                      n_channels=self._nb_channel, n_epochs=10,
                                      title=self._decomposition + '  ' + self._name,
                                      # order=np.arange(shape[1] - 1, -1, -1),  # /!\ is this important ?????????
                                      scalings=scaling,
                                      show=False,
                                      event_color=event_color, block=False,
                                      show_scrollbars=True,
                                      epoch_colors=[['#0940FF'] * self._nb_channel] * self._nb_epoch)
            else:
                color_choice = ['C1', 'C3', '#FFC419', 'C4', 'C7']
                event_color = {key: color_choice[i] for i, key, in enumerate(self._data.event_id.keys())}
                fig = self._data.plot(picks=['eeg', 'eog'], events=self._data.events, event_id=self._data.event_id,
                                      n_channels=self._nb_channel, n_epochs=10,
                                      title=self._name,
                                      scalings=dict(eeg=15e-6, eog=15e-9),
                                      show=False,
                                      event_color=event_color, block=False,
                                      show_scrollbars=True, group_by='type',
                                      epoch_colors=[['#1960FF'] * self._nb_channel] * self._nb_epoch)

            save_name = self._name + '-data' + ('-' + self._decomposition if self._decomposition else "")

        elif isinstance(self._data, mne.EvokedArray):
            if self._decomposition == 'PCA' or self._decomposition == 'ICA':
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
                fig = self._data.plot(picks=['eeg'], exclude=[], unit=True, show=False, proj=False, hline=None,
                                      scalings=None,
                                      titles='Evoked ' + self._decomposition,
                                      axes=ax, gfp=False,
                                      window_title='Evoked ' + self._decomposition + '  ' + self._name,
                                      spatial_colors=False, zorder='unsorted',
                                      selectable=False, noise_cov=None, time_unit='ms')
                ax.set_axisbelow(True)
                ax.xaxis.grid(True, which='major', linestyle='dotted', zorder=2)
                ax.yaxis.grid(True, which='major', linestyle='dotted', zorder=2)

            else:
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
                fig = self._data.plot(picks='eeg', exclude=[], unit=True, show=False, ylim=None, xlim='tight',
                                      proj=False,
                                      hline=None, units=None,
                                      scalings=None,
                                      titles='Evoked ' + self.stim_type.capitalize(),
                                      axes=ax, gfp=False,
                                      window_title='Evoked ' + self.stim_type.capitalize() + '  ' + self._name,
                                      spatial_colors=True, zorder='unsorted',
                                      selectable=True, noise_cov=None, time_unit='s', sphere=None)
                ax.set_axisbelow(True)
                ax.xaxis.set_minor_locator(AutoMinorLocator())

                with warnings.catch_warnings(record=True) as warning:
                    fig.tight_layout(pad=1.5)
            save_name = self._name + '-evoked_data' + ('-' + self._decomposition if self._decomposition else "")
        self.save(fig=fig, save_name=save_name)

    def psd_plot(self):
        """
        Plot the power spectral density for each channel
            saving name: subj_ID-psd(-ICA or PCA)
        :return: /
        """
        if self.plot_psd and isinstance(self._data, mne.BaseEpochs):
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), num=self._name)
            with warnings.catch_warnings(record=True) as warning:
                self._data.plot_psd(fmin=0, fmax=50, tmin=None, tmax=None, proj=False, bandwidth=None, adaptive=False,
                                    low_bias=True,
                                    normalization='length', picks=['eeg'], ax=axes[0], color='black', xscale='linear',
                                    area_mode='std', area_alpha=0.33, dB=False, estimate='auto', show=False, n_jobs=-1,
                                    average=False, line_alpha=None, spatial_colors=True, sphere=None, verbose=None)
                self._data.plot_psd(fmin=0, fmax=50, tmin=None, tmax=None, proj=False, bandwidth=None, adaptive=False,
                                    low_bias=True,
                                    normalization='length', picks=['eeg'], ax=axes[1], color='black', xscale='linear',
                                    area_mode='std', area_alpha=0.33, dB=False, estimate='auto', show=False, n_jobs=-1,
                                    average=True, line_alpha=None, spatial_colors=True, sphere=None, verbose=None)
            self.layout(case=1, fig=fig, axes=axes)
            save_name = self._name + '-psd' + ('-' + self._decomposition if self._decomposition else "")
            self.save(fig=fig, save_name=save_name)

    def sensors_plot(self):
        """
        Plot the sensor location, only works if no decomposition is applied on the data.
            saving name: sensor location
        :return: /
        """
        if self.plot_sensors and self._decomposition is None:
            fig, ax = plt.subplots(figsize=(11.6 / 2.5, 11.6 / 2.5), tight_layout=True)
            fig = self._data.plot_sensors(kind='topomap',  # ‘topomap’, ‘3d’, ‘select’.
                                          ch_type='eeg', title='Electrodes location', show_names=True,
                                          pointsize=15,
                                          ch_groups=None,  # None / 'position'
                                          to_sphere=False,  # change to True / False
                                          axes=ax, block=False, show=False,
                                          sphere=None, verbose=None)
            save_name = 'sensor location'
            self.save(fig=fig, save_name=save_name)

    def image_plot(self):
        """
        Plot Event Related Potential / Fields image.
            saving name: subj_ID-channels-stim_type(-ICA or PCA)
        :return: /
        """
        if self.plot_image and isinstance(self._data, mne.BaseEpochs):
            # sigma: the variance of the gaussian smoothing window / like low-pass filtering
            # picks: should contain the desired channel to display it's ERP
            # combine: 'mean' 'median' 'std' 'gfp' (global field power) if None it's gfp
            # group_by: None, dict(tile_1=[1, 2, 3], title_2=['Fz', 'Cz'])

            # /!\ this should be defined
            if self._decomposition == 'PCA':
                picks = ['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5']
                picks = [ch_name for ch_name in self._data.info['ch_names'] if ch_name.split('|')[0] in picks]
            elif self._decomposition == 'ICA':
                picks = ['IC 1', 'IC 2', 'IC 3', 'IC 4', 'IC 5']
                picks = [ch_name for ch_name in self._data.info['ch_names'] if ch_name.split('|')[0] in picks]
                # the split is not working but I need to add the pvaf in the ICs
            else:
                picks = ['CPz']  # FCz, CPz, Cz

            if self.split:
                for stim, value in self._data.event_id.items():
                    with warnings.catch_warnings(record=True) as warning:
                        figs = self._data[stim].plot_image(picks=picks, sigma=0.0, vmin=None, vmax=None, order=None,
                                                           show=False,
                                                           scalings=None, overlay_times=None, combine=None,
                                                           group_by=None, title='ERP of ' + stim,
                                                           clear=False, fig=None, axes=None)
                        for i, figure in enumerate(figs):
                            # remove spacing in the picks for the case of PCA and ICA, and remove the % thing at the end
                            save_name = self._name + '-' + "".join(picks[i].split('|')[0].split()) + '-' + stim + \
                                        ('-' + self._decomposition if self._decomposition else "")
                            self.save(fig=figure, save_name=save_name)

            else:
                with warnings.catch_warnings(record=True) as warning:
                    figs = self._data.plot_image(picks=picks, sigma=0.0, vmin=None, vmax=None, order=None,
                                                 show=False,
                                                 scalings=None, overlay_times=None, combine=None, group_by=None,
                                                 title='ERP',
                                                 clear=False, fig=None, axes=None)
                    for i, figure in enumerate(figs):
                        save_name = self._name + '-' + "".join(picks[i].split('|')[0].split()) + '-all_stim' + \
                                    ('-' + self._decomposition if self._decomposition else "")
                        self.save(fig=figure, save_name=save_name)

    def psd_topomap_plot(self):
        """
        Plot the topomap of the power spectral density across epochs.
            saving name: subj_ID-psd_topomap_plot
        :return: /
        """
        # if we have a decomposition we should take the signal of interest and project it back to the normal
        # electrode location so we can plot this (not done yet I should change maybe the virtual channel extractor in
        # utils part maybe ?
        if self.plot_psd_topomap and isinstance(self._data, mne.BaseEpochs) and self._decomposition is None:
            # bands: [(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta'), (30, 45, 'Gamma')]
            fig = self._data.plot_psd_topomap(bands=None, proj=False, bandwidth=None, adaptive=False,
                                              low_bias=True, normalization='length', ch_type='eeg', cmap=None,
                                              agg_fun=None,
                                              dB=True, n_jobs=-1, normalize=False, cbar_fmt='auto', outlines='skirt',
                                              axes=None,
                                              show=False, sphere=None, vlim=(None, None), verbose=None)
            save_name = self._name + '-psd_topomap_plot'
            self.save(fig=fig, save_name=save_name)

    def topo_image_plot(self):
        """
        Plot Event Related Potential / Fields image on topographies.
            saving name: subj_ID-topo_image-stim(- ICA / PCA)
        :return:
        """
        if not self.plot_topo_image or self._decomposition is not None:
            return

        if isinstance(self._data, mne.BaseEpochs):
            layout = mne.channels.find_layout(self._data.info, ch_type='eeg')
            if self.split:
                for stim, value in self._data.event_id.items():
                    fig = self._data[stim].plot_topo_image(layout=layout, fig_facecolor='w', show=False,
                                                           font_color='k', title='Topo map : ' + self._name
                                                                                 + ' ' + stim)
                    save_name = self._name + '-topo_image-' + stim \
                                + (self._decomposition if self._decomposition else "")
                    self.save(fig=fig, save_name=save_name)

            else:
                fig = self._data.plot_topo_image(layout=layout, fig_facecolor='w', show=False,
                                                 font_color='k', title='Topo map : ' + self._name)
                save_name = self._name + '-topo_image' + ('-' + self._decomposition if self._decomposition else "")
                self.save(fig=fig, save_name=save_name)

        if isinstance(self._data, mne.EvokedArray):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 9))
            layout = mne.channels.find_layout(self._data.info, ch_type='eeg')
            self._data.plot_topo(layout=layout, layout_scale=0.95, color='b', border='none', ylim=None, scalings=None,
                                 title='Evoked topo-image ' + self.stim_type.capitalize() + '  ' + self._name,
                                 proj=False, vline=[0.0], fig_background=None, merge_grads=False,
                                 legend=False, axes=ax, background_color='w', noise_cov=None, show=False)
            fig.tight_layout(pad=3)
            save_name = self._name + '-evoked_topo_image' + '-' + self.stim_type + \
                        ('-' + self._decomposition if self._decomposition else "")
            self.save(fig=fig, save_name=save_name)

    def topo_map_plot(self):
        """
        Plot a topographic map as image.
            saving name: subj_ID-evoked_topo_map-stim
        :return:
        """
        if self.plot_topo_map and isinstance(self._data, mne.EvokedArray) and self._decomposition is None:
            # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 3))
            # times could be "peaks" or "interactive"
            fig = self._data.plot_topomap(times='peaks', ch_type='eeg', vmin=None, vmax=None, cmap=None, sensors=True,
                                          colorbar=True,
                                          scalings=None, units=None, res=64, size=1.3, cbar_fmt='%3.1f', time_unit='s',
                                          time_format=None, proj=False, show=False, show_names=False,
                                          title=self._name + '  Evoked topo-map  ' + self.stim_type.capitalize(),
                                          mask=None, mask_params=None, outlines='skirt',
                                          contours=1, image_interp='bilinear', average=None, axes=None,
                                          extrapolate='head', sphere=None, border='mean', nrows=1, ncols='auto')
            save_name = self._name + '-evoked_topo_map' + '-' + self.stim_type
            self.save(fig=fig, save_name=save_name)

    def evoked_plot_joint(self):
        """
        Plot evoked data as butterfly plot and add topomaps for time points.
            saving name: subj_ID-evoked_joint-stim
        :return: /
        """
        if self.plot_evoked_joint and isinstance(self._data, mne.EvokedArray) and self._decomposition is None:
            # times (where to show topo plot), float, "peaks"
            fig = self._data.plot_joint(times='peaks', title='Evoked ' + self.stim_type.capitalize(),
                                        picks='eeg',
                                        exclude=[],
                                        # exclude=['Fp1', 'Fz', 'F3', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3',
                                        #          'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2', 'Cz', 'C4', 'T8',
                                        #          'FC6', 'FC2', 'F4', 'Fp2', 'AF7', 'AF3', 'F1', 'F5', 'FT7', 'FC3',
                                        #          'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO3', 'POz', 'PO4', 'P6', 'P2',
                                        #          'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8',
                                        #          'CPz'],
                                        show=False, ts_args=None, topomap_args=None)
            save_name = self._name + '-evoked_joint' + '-' + self.stim_type
            self.save(fig=fig, save_name=save_name)

    @staticmethod
    def plot_control(x=None, y=None, y_hat=None, freq=None, group_by='all', cmap='blue_red',
                     fig_name=None, show=False, save=False, vmax=0.1, id_label=None):
        """
        Plot the features of all patients in the same figure to see some general pattern if there is.
            or can plot the features of only one patient.
        :param x: features ,shape (n_patient, n_features).
        :param y: If n_patient = 1 this should be the real signal we are fitting our model to, shape (n_point).
                    if n_patient > 1 this should contain the labels, shape (n_patient).
        :param y_hat: predicted signal, shape (n_point). Ignored if n_patient > 1
        :param freq: frequency of the model.
        :param group_by: can be (int) to create a figure for each group_by subjects. If group_by='all' put all the
                          subjects on the same figure.
        :param cmap: color map can only be ['bleu_red' or 'black_white']. 'black_white' show an illuminating white
                      points where the VMS are non-zero, the 'blue_red' show a gradient from blue (negative value)
                      to red (positive value) of the VMS while white reprends the 0 value.
        :param fig_name: saving figure name.
        :param show: True in order to show the plotted figures.
        :param save: True in order to save the figures.
        :param vmax: The highest absolute value over which the VMS are considered at max or mix, so their color is blue
                      or red (instead of being a color in between).
        :param id_label: If given will put subj_ID at each subject band.
        :return: /
        """
        # Some verifications
        if (y is None) ^ (y_hat is None):
            raise ValueError("The real signal y and the predicted signal y_hat should be given simultaneously "
                             "but got instead y = {:} and y_hat = {:}".format(y, y_hat))

        # x axis is for the features
        # y axis is for time

        # Setting some parameters
        n_patient = x.shape[0]
        n_feature = x.shape[1]
        n_freq = len(freq)
        n_point = len(y) if y is not None else int(n_feature / n_freq)

        # print(f'{n_patient = }')
        # print(f'{n_feature = }')
        # print(f'{n_freq = }')
        # print(f'{n_point = }')

        # Color map choice and settings
        cmap_choice = cmap
        if cmap.lower() not in ['black_white', 'blue_red']:
            raise ValueError('Plot control cmap_type supported {:} but {:} were given'
                             .format(['black_white', 'blue_red'], cmap.lower()))
        if cmap == 'black_white':
            cmap = ListedColormap(['k', 'w'], name='binary')
            x[x != 0] = 1
            vmin, vmax, = 0, 1

        elif cmap == 'blue_red':
            cmap = plt.get_cmap('seismic').copy()
            cmap = truncate_colormap(cmap, 0.2, 0.8, 255)
            white_zero = list(map(cmap, range(255)))
            white_zero[127] = (1.0, 1.0, 1.0, 1.0)
            cmap = cmap.from_list('my_map', white_zero, N=255)
            vmin = -vmax

        # reshaping the data
        try:
            x = np.swapaxes(np.array(np.split(x, n_point, axis=1)), 0, 1)
        except ValueError:
            n_point -= 1
            y = y[1:]
            y_hat = y_hat[1:]
            x = np.swapaxes(np.array(np.split(x, n_point, axis=1)), 0, 1)

        if n_patient == 1:  # show one by one
            if y is not None and y_hat is not None:
                fig, axes = plt.subplots(nrows=2, figsize=(19.2 / 1.7, 10.8 / 1.7), dpi=100)  # 1080 1920
                x_axis_add = 0.01 * n_point if not cmap_choice == 'black_white' else 0.5
                # plotting y and y_hat
                Plotter.plot_y_hat(y, y_hat, fig_name=fig_name, show=show, save=False, ax=axes[0], fig=fig)
                axes[0].set_xlim([-x_axis_add, n_point - 1 + x_axis_add])
                # Plotting the activation
                axes[1].imshow(x.T, interpolation='nearest', aspect='auto', origin='upper',  # 'auto'  'equal'
                               cmap=cmap, vmax=vmax, vmin=vmin)
                # axes[1].xaxis.set_ticks_position('top')
                axes[1].xaxis.set_minor_locator(AutoMinorLocator())
                x1, x2, _, _ = axes[0].axis()
                axes[1].set_xlim([x1, x2])
                ylim = axes[1].get_ylim()
                # managing the ticks
                y_ticks = set_ticks(n_freq)
                axes[1].set_yticks(y_ticks)
                axes[1].set_ylim(ylim)
                y_labels = ['{:1.1f}'.format(freq[int(idx)]) if 0 <= int(idx) < n_freq else 'pb'
                            for idx in axes[1].get_yticks()]
                axes[1].set_yticklabels(y_labels)
                axes[1].set_xlabel('Time [index]')
                axes[1].set_ylabel('Model Frequency [Hz]')
                axes[1].tick_params(top=True, direction='out', which='both')
                axes[1].yaxis.set_minor_locator(AutoMinorLocator())
                axes[1].tick_params(which='major', length=6)

            else:  # y_hat or y not given
                fig, ax = plt.subplots(nrows=1, figsize=(19.2 / 1.5, 10.8 / 3), dpi=100)  # 1080 1920
                ax.imshow(x.T, interpolation='nearest', aspect='auto', origin='upper',  # 'auto'  'equal'
                          cmap=cmap, vmax=vmax, vmin=vmin)
                ylim = ax.get_ylim()
                y_ticks = set_ticks(n_freq)
                ax.set_yticks(y_ticks)
                ax.set_ylim(ylim)
                y_labels = ['{:1.1f}'.format(freq[int(idx)]) if 0 <= int(idx) < n_freq else 'pb'
                            for idx in ax.get_yticks()]
                ax.set_yticklabels(y_labels)
                ax.set_xlabel('Time [index]')
                ax.set_ylabel('Model Frequency [Hz]')
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.tick_params(which='major', length=6)
                ax.tick_params(top=True, direction='out', which='both')
            plt.tight_layout()
        else:  # show a group of pictures
            if not isinstance(group_by, int) and group_by != 'all':
                raise ValueError("Possible values for group_by are [type(int), 'all'] but {:} were given"
                                 .format(group_by))

            n_group = n_patient if group_by == 'all' else group_by

            for x, n_batch in batch(x, n_group):
                fig, ax = plt.subplots(figsize=(19.2 / 1.5, 10.8 / 1.5), dpi=100)
                x = np.concatenate(x, axis=-1)
                # Main plot function
                print(f'{x.shape = }')
                ax.imshow(x, interpolation='nearest', aspect='auto', origin='upper',  # 'auto'  'equal'
                          cmap=cmap, vmax=vmax, vmin=vmin)
                ax.set_ylabel('Time [index]')
                title = 'Model Frequency [Hz]' if id_label is None else 'Model Frequency: {:} [Hz]'.format(str(freq[0]))
                ax.set_xlabel(title)
                ax.xaxis.set_label_position('top')
                ax.xaxis.tick_top()

                if n_freq > 6:
                    ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.tick_params(which='major', length=6)
                ax.tick_params(right=True, top=True, bottom=False, direction='out', which='both')
                xlim = ax.get_xlim()
                x_ticks = set_ticks(n_freq, n_batch)
                ax.set_xticks(x_ticks)

                # # removing the lowest frequency from ticks
                # for i in range(1, shape[0]):
                #     x_ticks.remove(i * n_freq)

                # removing the lowest frequency from ticks (0 <= to add it and 0 < int to remove it)
                if id_label is not None and n_freq == 1:
                    x_labels = id_label
                elif n_freq == 1 and id_label is None:
                    x_labels = [str(freq[0])] * len(ax.get_xticks())
                else:
                    x_labels = [float_string(freq[int(idx) % n_freq]) if 0 < int(idx) % n_freq < n_freq else ''
                                for idx in ax.get_xticks()]

                # Adding dashed vertical lines to separate the subjects
                if cmap_choice == 'black_white':
                    for i in range(1, n_batch):
                        ax.axvline(x=(i + 1) * n_freq - 0.5, ymin=-0.5, ymax=n_point + 0.5,
                                   alpha=0.2, c='w', linestyle='dashed', linewidth=1)
                elif cmap_choice == 'blue_red':
                    for i in range(1, n_batch):
                        ax.axvline(x=(i + 1) * n_freq - 0.5, ymin=-0.5, ymax=n_point + 0.5,
                                   alpha=0.2, c='k', linestyle='dashed', linewidth=1)
                # Horizontal lines
                # ax.axhline(y=100, alpha=0.8, c='k', linestyle='solid', linewidth=3)
                # ax.axhline(y=350, alpha=0.8, c='k', linestyle='solid', linewidth=3)
                ax.set_xticklabels(x_labels, rotation='45')
                ax.set_xlim(xlim)
                plt.tight_layout()

        # I just added this at the end idk if it's okay or no (have no time !)
        if save:
            path = os.path.join(os.getcwd(), 'figures', fig_name + '.svg')
            fig.savefig(path, bbox_inches='tight', dpi=500)
        if not show:
            plt.close()

    # @staticmethod
    # def plot_control_backup(u_hat, freq, fig_name, show=False, save=False, u=None, hist_norm=True):
    #     with warnings.catch_warnings(record=True) as warning:  # because stem doesn't support nan values
    #         if u is not None:
    #             u[u == 0] = np.nan
    #             u = u.T
    #
    #         vbins = 25
    #         hbins = 10
    #         n_point = u_hat.shape[0]
    #         n_modes = u_hat.shape[1]
    #         u_hat[u_hat == 0] = np.nan
    #         idx = np.linspace(0, n_point - 1, n_point)
    #
    #         fig = plt.figure(constrained_layout=True, figsize=(8, 1.8 * (n_modes + 1)))
    #         hist_size = 1
    #         widths = [1, 0.3 * 4 / 16]
    #         heights = [hist_size, *[1] * n_modes]
    #         gs = fig.add_gridspec(ncols=2, nrows=n_modes + 1, left=0.1, right=0.98, top=0.98, bottom=0.05, wspace=0.08,
    #                               hspace=0.3,
    #                               height_ratios=heights, width_ratios=widths)
    #         hist_x_axis_max = (0, 0)
    #         for row in range(n_modes + 1):
    #             if row == 0:  # vertical histograms
    #                 hist_ax = fig.add_subplot(gs[row, 0])  # here should be the first histogram
    #                 hist_ax.tick_params(direction='in', labelbottom=False)
    #                 hist_ax.xaxis.set_minor_locator(AutoMinorLocator())
    #                 hist_ax.set_xlim(left=-2, right=n_point + 1)
    #                 hist_ax.hist(np.argwhere(~np.isnan(u_hat)).T[0], bins=vbins, zorder=10)
    #                 hist_ax.xaxis.grid(True, which='major', linestyle='-', linewidth=0.4)
    #                 hist_ax.yaxis.grid(True, which='major', linestyle='-', linewidth=0.4)
    #
    #             elif row != 0:
    #                 ax = fig.add_subplot(gs[row, 0])
    #                 ax.tick_params(direction='in', right=True)
    #                 if row == 1:
    #                     ax.tick_params(direction='in', top=True)
    #                 if u is not None:
    #                     marker, stem, base = ax.stem(idx, u[:, row - 1], linefmt='k:', markerfmt='kx', basefmt="gray")
    #                     plt.setp(marker, markersize=6)
    #                     plt.setp(stem, linewidth=1)
    #                     plt.setp(base, 'linewidth', 0.6)
    #                 marker, stem, base = ax.stem(idx, u_hat[:, row - 1], linefmt='C0-', markerfmt='C3o', basefmt="gray")
    #                 plt.setp(marker, markersize=6)
    #                 plt.setp(stem, linewidth=0.8)
    #                 plt.setp(base, 'linewidth', 0.6)
    #                 ax.set_ylabel("U(f={:.2f} [Hz])".format(freq[row - 1]))
    #                 ax.yaxis.set_label_coords(-0.08, 0.5)
    #                 ax.xaxis.set_minor_locator(AutoMinorLocator())
    #                 ax.xaxis.grid(True, which='minor', linestyle='--', linewidth=0.4)
    #                 ax.xaxis.grid(True, which='major', linestyle='-', linewidth=0.4)
    #                 ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.4)
    #                 ax.set_xlim(left=-2, right=n_point + 1)
    #                 if row == n_modes:
    #                     ax.set_xlabel("Time [index]")
    #
    #                 hist_ax = fig.add_subplot(gs[row, 1])  # horizontal histograms
    #                 hist_ax.tick_params(direction='in', labelleft=False)
    #                 hist_ax.set_ylim(ax.get_ylim())
    #                 try:
    #                     with warnings.catch_warnings(record=True) as warning:
    #                         hist_ax.hist(u_hat[:, row - 1], bins=hbins, orientation='horizontal', zorder=10)
    #                 except ValueError:
    #                     pass
    #                 hist_ax.xaxis.set_minor_locator(AutoMinorLocator())
    #                 hist_ax.xaxis.grid(True, which='major', linestyle='-', linewidth=0.4)
    #                 hist_ax.xaxis.grid(True, which='minor', linestyle='--', linewidth=0.4)
    #                 if hist_norm and hist_ax.get_xlim()[1] > hist_x_axis_max[1]:
    #                     hist_x_axis_max = hist_ax.get_xlim()
    #
    #         if hist_norm:
    #             for hist_ax in fig.axes[2::2]:
    #                 hist_ax.set_xlim(hist_x_axis_max)
    #
    #         if save:
    #             fig.savefig(os.path.join(os.getcwd(), 'figures', fig_name), bbox_inches='tight', dpi=100)
    #         if not show:
    #             plt.close()

    @staticmethod
    def plot_y_hat(y, y_hat, fig_name, show=True, save=False, ax=None, fig=None):
        """
        Plot the real signal along with the predicted signal if y_hat is given.
        :param y: real EEG signal to be displayed.
        :param y_hat: predicted EEG signal (output of our model) to be displayed.
        :param fig_name: saving figure name.
        :param show: True in order to show the plotted figures.
        :param save: True in order to save the figures.
        :param ax: plot on this ax if provided, else create a new one.
        :param fig: plot on this fig if provided, else create a new one.
        :return: /
        """
        n_point = y.shape[0]
        if ax is None or fig is None:
            fig, ax = plt.subplots(figsize=[12, 8])
        ax.plot(np.arange(0, n_point), y, label='Y_true')
        ax.plot(np.arange(0, n_point), y_hat, 'C3', label='Y_pred')
        # ax.set_ylim(ax.get_ylim()[::-1])  # to invert the y axis
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.grid(True, which='minor', linestyle=':', linewidth=0.3)
        ax.xaxis.grid(True, which='major', linestyle='-', linewidth=0.4)
        ax.yaxis.grid(True, which='major', linestyle='-', linewidth=0.4)
        ax.legend()
        ax.set_xlabel("Time [index]")
        ax.set_ylabel("Amplitude [$\mu$V]")
        fig.tight_layout()
        if save:
            path = os.path.join(os.getcwd(), 'figures', fig_name + '-fit.svg')
            fig.savefig(path, bbox_inches='tight', dpi=200)
        if not show:
            plt.close()

    @staticmethod
    def plot_alpha(alphas, coef_path, residue_path, fig_name, show=True, save=False, selection=None):
        """
        Plot the residue: epsilon = |y - \hat{y}|^2 (+ |\beta|_1 or _2) for a varying level or sparsity in blue.
        The number of features used by the model is displayed in red.
        :param alphas: the sparsity levels (returned from LASSO-Lars).
        :param coef_path: virtual modal stimuli for different sparsity levels.
        :param residue_path: epsilon value for different sparsity levels.
        :param fig_name: saving figure name.
        :param show: True in order to show the plotted figures.
        :param save: True in order to save the figures.
        :param selection: Contain the selected levels of sparsity by the algorithm and mark them in the figure. If None,
                           no marks are added.
        :return: /
        """
        # Plot alpha decay in top sub-plot
        fig, axes = plt.subplots(figsize=[12, 12], nrows=2)
        axes[0].plot(alphas)
        axes[0].set_xlabel("Iteration", fontsize=13)
        axes[0].set_ylabel("$\\alpha$", fontsize=13)
        axes[0].xaxis.set_minor_locator(AutoMinorLocator())
        axes[0].xaxis.grid(True, which='minor', linestyle=':', linewidth=0.3)
        axes[0].xaxis.grid(True, which='major', linestyle='-', linewidth=0.4)
        axes[0].yaxis.grid(True, which='major', linestyle='-', linewidth=0.4)
        if selection is not None:
            axes[0].scatter(selection, alphas[selection], c='k', marker='x', zorder=9, alpha=0.7)
        
        # Plot epsilon in blue in bottom sub-plot
        color = 'tab:blue'
        axes[1].plot(residue_path, color=color, zorder=1)
        axes[1].set_xlabel("Iteration", fontsize=13)
        axes[1].set_ylabel("$\\frac{1}{2N}|y-\\hat{y}|_2^2 + \\alpha|\hat{u}|$", fontsize=13, color=color)
        axes[1].tick_params(axis='y', labelcolor=color)
        axes[1].xaxis.grid(True, which='major', linestyle='-', linewidth=0.4)
        axes[1].xaxis.grid(True, which='minor', linestyle=':', linewidth=0.3)
        if selection is not None:
            axes[1].scatter(selection, residue_path[selection], c='k', marker='x', zorder=9, alpha=0.7)
        
        # Plot of the number of non zero entries of \beta in red in bottom sub-plot
        color = 'tab:red'
        ax = axes[1].twinx()
        ax.plot(np.count_nonzero(coef_path, axis=0), color=color, zorder=2)
        ax.set_xlabel("Iteration", fontsize=13)
        ax.set_ylabel("# of parameters", fontsize=13, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.grid(True, which='major', linestyle='-', linewidth=0.4)
        if selection is not None:
            ax.scatter(selection, np.count_nonzero(coef_path, axis=0)[selection],
                       c='k', marker='x', zorder=9, alpha=0.7)

        fig.tight_layout()
        if save:
            fig.savefig(os.path.join(os.getcwd(), 'figures', fig_name + '-residue.png'),
                        bbox_inches='tight', dpi=200)
        if not show:
            plt.close()

    @staticmethod
    def decision_boundary(x, y, clf):
        """
        Do a scatter plot of the features x with a decision boundary (used for 2 features)
        :param x: features
        :param y: label (class 0 or 1)
        :param clf: classifier to draw the decision boundary
        :return: /
        """
        colors = ["#ff0000", "#0000ff"]
        labels = [0, 1]

        ax: matplotlib.pyplot.Axes
        fig: matplotlib.pyplot.Figure
        fig, ax = plt.subplots(1, 1, figsize=(11.6 / 2, 11.6 / 2 / np.sqrt(2)), tight_layout=True)

        # Scatter points
        for species, color in zip(labels, colors):
            data = x[y == species]
            ax.scatter(data[:, 0], data[:, 1], color=color, alpha=0.55, s=40)

        # Decision boudary
        x1 = np.array([3, np.max(x[:, 0], axis=0) + 10])
        b, w1, w2 = clf.intercept_[0], clf.coef_[0][0], clf.coef_[0][1]
        y1 = -(b + x1 * w1) / w2
        ax.plot(x1, y1, c='k', linestyle='--', linewidth=1)

        # Pretty plots
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlim(-9, 380)
        ax.xaxis.grid(True, which='minor', linestyle=':', color='#e6e6e6', linewidth=0.283, dashes=[0.7, 3])
        ax.yaxis.grid(True, which='minor', linestyle=':', color='#e6e6e6', linewidth=0.283, dashes=[0.7, 3])
        ax.xaxis.grid(True, which='major', linestyle='-', color='#ebebeb', linewidth=0.409, alpha=0.8)
        ax.yaxis.grid(True, which='major', linestyle='-', color='#ebebeb', linewidth=0.409, alpha=0.8)
        ax.set_axisbelow(True)
        ax.legend(['PD', 'CTL'])
        ax.set_xlabel('$F_1$ (ms)')
        ax.set_ylabel('$F_2  (\mu V)$')
        fig.savefig('./figures/decision_boudary.svg')

    @staticmethod
    def scatter(x, y, clf):
        """
        Do a scatter plot of the features x with a decision boundary (used for 1 feature only)
        :param x: feature
        :param y: label (class 0 or 1)
        :param clf: classifier to draw the decision boundary
        :return: /
        """
        colors = ["#ff0000", "#0000ff"]
        labels = [0, 1]

        ax: matplotlib.pyplot.Axes
        fig: matplotlib.pyplot.Figure
        fig, ax = plt.subplots(1, 1, figsize=(11.6 / 1.8, 11.6 / 1.8 / np.sqrt(2)), tight_layout=True)

        # Scatter points
        for species, color in zip(labels, colors):
            data = x[y == species]
            ax.scatter(data, np.zeros_like(data), color=color, alpha=0.5, s=40)

        # Decision boudary
        x1 = np.array([np.min(x), np.max(x)])
        b, w1, w2 = clf.intercept_[0], clf.coef_[0], clf.coef_[0]
        y1 = -(b + x1 * w1) / w2
        ax.plot(x1, y1, c='k', linestyle='--', linewidth=1)

        # Pretty plots
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        # ax.set_xlim(-2.5, 180)
        ax.xaxis.grid(True, which='minor', linestyle=':', color='#e6e6e6', linewidth=0.283, dashes=[0.7, 3])
        ax.xaxis.grid(True, which='major', linestyle='-', color='#ebebeb', linewidth=0.409, alpha=0.8)
        ax.set_axisbelow(True)
        ax.legend(['PD', 'CTL'])
        ax.set_xlabel('$F_1$')

    @staticmethod
    def correlogram(x, y):
        """
        Create a correlogram of the features x
        :param x: features
        :param y: label (class 0 or 1)
        :return: /
        """
        n_features = x.shape[1]
        n_patients = x.shape[0]
        colors = ["#ff0000", "#0000ff"]
        labels = [0, 1]

        fig, axes = plt.subplots(n_features, n_features, figsize=(19.2 / 1.5, 10.8 / 1.5), tight_layout=True)

        if n_features == 1:
            for species, color in zip(labels, colors):
                data = x[y == species]
                axes.hist(data, bins=5, alpha=0.9, color=color)
                # axes.scatter(data, np.zeros_like(data), alpha=0.8, color=color)
            return

        for i in range(n_features):
            for j in range(n_features):
                # If this is the lower-triangule, add a scatterlpot for each group.
                if i > j:
                    for species, color in zip(labels, colors):
                        data = x[y == species]
                        axes[i, j].scatter(data[:, j], data[:, i], color=color, alpha=0.5, s=20)
                if j == 0:
                    axes[i, j].set_ylabel('Feature ' + str(i + 1))
                if i == n_features - 1:
                    axes[i, j].set_xlabel('Feature ' + str(j + 1))

                # If this is the main diagonal, add histograms
                if i == j:
                    for species, color in zip(labels, colors):
                        data = x[y == species]
                        axes[i, j].hist(data[:, j], bins=20, alpha=0.5, color=color)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

        for i in range(n_features):
            for j in range(n_features):
                # If on the upper triangle
                if i < j:
                    axes[i, j].remove()


# create repeated ticks for control_plot
def set_ticks(n_freq, n_batch=None):
    if n_batch is not None:  # case where I group features
        extra_ticks = [(i + 1) * n_freq - 1 for i in range(n_batch)]
        if n_freq < 6:
            ticks = list(np.arange(0, n_batch * n_freq, n_freq)) + extra_ticks
        elif n_freq < 20:
            ticks = list(np.arange(0, n_batch * n_freq, int(n_freq / 5))) + extra_ticks
        else:
            ticks = list(np.arange(0, n_batch * n_freq, int(n_freq / 10))) + extra_ticks
        return list(set(ticks))
    else:  # case where I plot one by one
        if n_freq < 6:
            ticks = list(np.arange(0, n_freq, n_freq)) + [n_freq - 1]
        elif n_freq < 20:
            ticks = list(np.arange(0, n_freq, int(n_freq / 5))) + [n_freq - 1]
        else:
            ticks = list(np.arange(0, n_freq, int(n_freq / 10))) + [n_freq - 1]
        return list(set(ticks))


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)], len(iterable[ndx:min(ndx + n, l)])


# Create new color map from min_val to max_val (to take a segment from a color map and use it as new color map)
def truncate_colormap(cmap, min_val=0.0, max_val=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=min_val, b=max_val),
        cmap(np.linspace(min_val, max_val, n, endpoint=True)))
    return new_cmap


def float_string(num):
    return str(int(num)) if num % 1 == 0 else '{:1.1f}'.format(num)
