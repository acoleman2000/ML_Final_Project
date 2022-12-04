import gc
from astropy.time import Time
import matplotlib.pyplot as plt
import matplotlib
# other magicks
import logging
logger_plot_event_name = 'plot_event'
logger_plot_event = logging.getLogger(logger_plot_event_name)
logger_plot_event.setLevel(logging.INFO)

# BL imports
import blimpy as bl
from blimpy.utils import rebin

from os.path import dirname, abspath, isdir

import linecache
import cv2

from genericpath import getsize
from random import randint

import numpy as np
import os
fontsize=16
font = {'family' : 'DejaVu Sans',
'size' : fontsize}
MAX_IMSHOW_POINTS = (4096, 1268)
MAX_PLT_POINTS      = 65536
plt.rcParams['figure.facecolor']='w'
plt.rcParams['savefig.facecolor']='w'

class Plotting:


    def plot_waterfall(wf, source_name, f_start=None, f_stop=None, **kwargs):
        r"""
        Plot waterfall of data in a .fil or .h5 file.

        Parameters
        ----------
        wf : blimpy.Waterfall object
            Waterfall object of an H5 or Filterbank file containing the dynamic spectrum data.
        source_name : str
            Name of the target.
        f_start : float
            Start frequency, in MHz.
        f_stop : float
            Stop frequency, in MHz.
        kwargs : dict
            Keyword args to be passed to matplotlib imshow().

        Notes
        -----
        Plot a single-panel waterfall plot (frequency vs. time vs. intensity)
        for one of the on or off observations in the cadence of interest, at the
        frequency of the expected event. Calls :func:`~overlay_drift`

        """
        # prepare font
        matplotlib.rc('font', **font)

        # Load in the data from fil
        plot_f, plot_data = wf.grab_data(f_start=f_start, f_stop=f_stop)

        # Make sure waterfall plot is under 4k*4k
        dec_fac_x, dec_fac_y = 1, 1

        # rebinning data to plot correctly with fewer points
        # try:
        #     if plot_data.shape[0] > MAX_IMSHOW_POINTS[0]:
        #         dec_fac_x = plot_data.shape[0] / MAX_IMSHOW_POINTS[0]
        #     if plot_data.shape[1] > MAX_IMSHOW_POINTS[1]:
        #         dec_fac_y =  int(np.ceil(plot_data.shape[1] /  MAX_IMSHOW_POINTS[1]))
        #     plot_data = rebin(plot_data, dec_fac_x, dec_fac_y)
        # except Exception as ex:
        #     print('\n*** Oops, grab_data returned plot_data.shape={}, plot_f.shape={}'
        #         .format(plot_data.shape, plot_f.shape))
        #     print('Waterfall info for {}:'.format(wf.filename))
        #     wf.info()
        #     raise ValueError('*** Something is wrong with the grab_data output!') from ex

        # Rolled back PR #82

        # determine extent of the plotting panel for imshow
        extent=(plot_f[0], plot_f[-1], (wf.timestamps[-1]-wf.timestamps[0])*24.*60.*60, 0.0)

        # plot and scale intensity (log vs. linear)
        kwargs['cmap'] = kwargs.get('cmap', 'viridis')
        # plot_data = 10.0 * np.log10(plot_data)

        # get normalization parameters
        vmin = plot_data.min()
        vmax = plot_data.max()
        median = np.median(plot_data)
        normalized_plot_data = plot_data #- median #(plot_data - vmin) / (vmax - vmin)

        # display the waterfall plot
        this_plot = plt.imshow(normalized_plot_data,
            aspect='auto',
            rasterized=False,
            interpolation='nearest',
            extent=extent,
            **kwargs
        )

        # add plot labels
        # plt.xlabel("Frequency [Hz]",fontdict=font)
        # plt.ylabel("Time [s]",fontdict=font)

        # add source name
        # ax = plt.gca()
        # plt.text(0.03, 0.8, source_name, transform=ax.transAxes, bbox=dict(facecolor='white'))
        # if plot_snr != False:
        #     plt.text(0.03, 0.6, plot_snr, transform=ax.transAxes, bbox=dict(facecolor='white'))
        # return plot

        del plot_f
        gc.collect()


        return this_plot, plot_data


    def make_waterfall_plots(wfs, on_source_name, f_start, f_stop, f_start_u, f_stop_u,
                            filter_level, source_name_list, log_file, plot_dir=None, **kwargs):
        r'''
        Makes waterfall plots of an event for an entire on-off cadence.

        Parameters
        ----------
        fil_file_list : str
            List of filterbank files in the cadence.
        on_source_name : str
            Name of the on_source target.
        f_start : float
            Start frequency, in MHz.
        f_stop : float
            Stop frequency, in MHz.
        drift_rate : float
            Drift rate in Hz/s.
        f_mid : float
            <iddle frequency of the event, in MHz.
        filter_level : int
            Filter level (1, 2, or 3) that produced the event.
        source_name_list : list
            List of source names in the cadence, in order.
        bandwidth : int
            Width of the plot, incorporating drift info.
        kwargs : dict
            Keyword args to be passed to matplotlib imshow().

        Notes
        -----
        Makes a series of waterfall plots, to be read from top to bottom, displaying a full cadence
        at the frequency of a recorded event from find_event. Calls :func:`~plot_waterfall`

        '''
        global logger_plot_event
        # prepare for plotting
        matplotlib.rc('font', **font)

        # set up the sub-plots
        n_plots = len(wfs)
        fig = plt.subplots(n_plots, sharex=True, sharey=True,figsize=(10, 2*n_plots))


        if not isdir(plot_dir):
            os.makedirs(plot_dir)
        dirpath = plot_dir


        # read in data for the first panel
        #print('plot_event make_waterfall_plots: max_load={} is required for {}'.format(max_load, fil_file_list[0]))
        wf1 = wfs[0]
        t0 = wf1.header['tstart']
        plot_f1, plot_data1 = wf1.grab_data()

        # # rebin data to plot correctly with fewer points
        # dec_fac_x, dec_fac_y = 1, 1
        # if plot_data1.shape[0] > MAX_IMSHOW_POINTS[0]:
        #     dec_fac_x = plot_data1.shape[0] / MAX_IMSHOW_POINTS[0]
        # if plot_data1.shape[1] > MAX_IMSHOW_POINTS[1]:
        #     dec_fac_y =  int(np.ceil(plot_data1.shape[1] /  MAX_IMSHOW_POINTS[1]))
        # plot_data1 = rebin(plot_data1, dec_fac_x, dec_fac_y)

        # define more plot parameters
        # never used: delta_f = 0.000250
        mid_f = np.abs(f_start+f_stop)/2.

        subplots = []
        del wf1, plot_f1, plot_data1
        gc.collect()

        on_data = []
        off_data = []

        # Fill in each subplot for the full plot
        for ii, wf in enumerate(wfs):
            # identify panel
            subplot = plt.subplot(n_plots, 1, ii + 1)
            subplots.append(subplot)


            # make plot with plot_waterfall
            source_name = source_name_list[ii]
            if ii % 2 == 0:
                figure = plt.figure()
                this_plot, data = Plotting.plot_waterfall(wf,
                                        source_name,
                                        f_start=f_start,
                                        f_stop=f_stop,
                                        **kwargs)
                on_data.extend(data)
                del data

                figure.canvas.draw()
                plt.savefig("On%s.png"%(ii//2))
                plt.clf()
                plt.close(figure)

            else:
                figure = plt.figure()
                this_plot, data = Plotting.plot_waterfall(wf,
                                    source_name,
                                    f_start=f_start_u,
                                    f_stop=f_stop_u,
                                    **kwargs)
                off_data.extend(data)
                del data
                figure.canvas.draw()
                plt.savefig("Off%s.png"%(ii//2))
                plt.clf()
                plt.close(figure)



            # calculate parameters for estimated drift line
            t_elapsed = Time(wf.header['tstart'], format='mjd').unix - Time(t0, format='mjd').unix


            # plot estimated drift line
            # Title the full plot

            # Format full plot
            # if ii < len(wfs)-1:
            #     plt.xticks(np.linspace(f_start, f_stop, num=4), ['','','',''])


            del wf
            gc.collect()


        # More overall plot formatting, axis labelling
        factor = 1e6
        units = 'Hz'

        # ax = plt.gca()
        # ax.get_xaxis().get_major_formatter().set_useOffset(False)
        # xloc = np.linspace(f_start, f_stop, 5)
        # xticks = [round(loc_freq) for loc_freq in (xloc - mid_f)*factor]
        # if np.max(xticks) > 1000:
        #     xticks = [xt/1000 for xt in xticks]
        #     units = 'kHz'
        # plt.xticks(xloc, xticks)
        # plt.xlabel("Relative Frequency [%s] from %f MHz"%(units,mid_f),fontdict=font)

        # Add colorbar
        cax = fig[0].add_axes([0.94, 0.11, 0.03, 0.77])
        fig[0].colorbar(this_plot,cax=cax,label='Normalized Power (Arbitrary Units)')

        # Adjust plots
        plt.subplots_adjust(hspace=0,wspace=0)

        # save the figures
        interesting = "uninteresting" if f_start == f_start_u else "interesting"
        path_png = dirpath + str(filter_level) + '_' + on_source_name + '_dr_' + '_freq_' "{:0.6f}".format(f_start) + "_" + interesting + ".png"
        hist_png = dirpath + str(filter_level) + '_' + on_source_name + '_dr_' + '_freq_' "{:0.6f}".format(f_start) + "_hist.png"
        plt.savefig(path_png, bbox_inches='tight', transparent=False)
        with open(log_file, "a") as f:
            f.write("%s,%s\n"%(path_png,on_source_name))
        logger_plot_event.debug('make_waterfall_plots: Saved file {}'.format(path_png))

        # show figure before closing if this is an interactive context
        mplbe = matplotlib.get_backend()
        logger_plot_event.debug('make_waterfall_plots: backend = {}'.format(mplbe))
        if mplbe != 'agg':
            plt.show()

        # close all figure windows
        plt.close('all')

        plt.hist(on_data, bins=30, alpha=0.75,label="On cadences")
        plt.hist(off_data, bins=30,alpha=0.75,label="Off cadences")
        plt.legend(loc='upper right')
        plt.savefig(hist_png, bbox_inches='tight',transparent=False)
        if mplbe != 'agg':
            plt.show()


        plt.close("all")
        # read image
        for i in range(0, 3):
            im = cv2.imread('On%s.png'%i)
            os.remove('On%s.png'%i)
            # calculate mean value from RGB channels and flatten to 1D array
            vals = im.mean(axis=2).flatten()
            # calculate histogram
            counts, bins = np.histogram(vals, range(257))
            # plot histogram centered on values 0..255

            subplot = plt.subplot(n_plots, 1, i * 2 + 1)
            plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
            plt.xlim([-0.5, 255.5])
            # plt.text(0.03, 0.8, "On Cadence %s"%(i+1), transform=subplot.transAxes, bbox=dict(facecolor='white'))
            subplots.append(subplot)

            im = cv2.imread('Off%s.png'%i)

            os.remove('Off%s.png'%i)
            # calculate mean value from RGB channels and flatten to 1D array
            vals = im.mean(axis=2).flatten()
            # calculate histogram
            counts, bins = np.histogram(vals, range(257))
            # plot histogram centered on values 0..255
            subplot = plt.subplot(n_plots, 1, i * 2 + 2)
            plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
            plt.xlim([-0.5, 255.5])
            # plt.text(0.03, 0.8, "Off Cadence %s"%(i+1), transform=subplot.transAxes, bbox=dict(facecolor='white'))
            subplots.append(subplot)

        plt.savefig(hist_png)
        if mplbe != 'agg':
            plt.show()


        return subplots


    def plot_spectrum(wf, t=0, f_start=None, f_stop=None, logged=False, if_id=0, c=None, **kwargs):
        """ Plot frequency spectrum of a given file
        Args:
            t (int): integration number to plot (0 -> len(data))
            logged (bool): Plot in linear (False) or dB units (True)
            if_id (int): IF identification (if multiple IF signals in file)
            c: color for line
            kwargs: keyword args to be passed to matplotlib plot()
        """
        if wf.header['nbits'] <= 2:
            logged = False
            t = 'all'
        ax = plt.gca()

        plot_f, plot_data = wf.grab_data(f_start, f_stop, if_id)

        # Using accending frequency for all plots.
        if wf.header['foff'] < 0:
            plot_data = plot_data[..., ::-1]  # Reverse data
            plot_f = plot_f[::-1]

        if isinstance(t, int):
            plot_data = plot_data[t]
        elif t == 'all':
            print("averaging along time axis...")
            # Since the data has been squeezed, the axis for time goes away if only one bin, causing a bug with axis=1
            if len(plot_data.shape) > 1:
                plot_data = plot_data.mean(axis=0)
            else:
                plot_data = plot_data.mean()
        else:
            raise RuntimeError("Unknown integration %s" % t)

        # Rebin to max number of points
        dec_fac_x = 1
        if plot_data.shape[0] > MAX_PLT_POINTS:
            dec_fac_x = int(plot_data.shape[0] / MAX_PLT_POINTS)

        plot_data = rebin(plot_data, dec_fac_x, 1)
        plot_f = rebin(plot_f, dec_fac_x, 1)

        if not c:
            kwargs['c'] = '#333333'


        plt.plot(plot_f, plot_data, **kwargs)


        plt.xlim(plot_f[0], plot_f[-1])
        return plt

    def plot_histograms(wfOn, wfOff, name):
        for i in range(0, len(wfOn)):
            fig = plt.figure()
            im = Plotting.plot_waterfall(wfOn[i],"")
            fig.canvas.draw()
            plt.savefig("On%s.png"%i)
            plt.clf()
            plt.close(fig)
        for i in range(0, len(wfOff)):
            fig = plt.figure()
            im = Plotting.plot_waterfall(wfOff[i],"")
            fig.canvas.draw()
            plt.savefig("Off%s.png"%i)
            plt.clf()
            n_plots = 6
            plt.close(fig)
        fig = plt.subplots(n_plots, sharex=True, sharey=True,figsize=(10, 2*n_plots))


        subplots = []



        # read image
        for i in range(0, len(wfOn)):
            im = cv2.imread('On%s.png'%i)
            os.remove('On%s.png'%i)
            # calculate mean value from RGB channels and flatten to 1D array
            vals = im.mean(axis=2).flatten()
            # calculate histogram
            counts, bins = np.histogram(vals, range(257))
            # plot histogram centered on values 0..255

            subplot = plt.subplot(n_plots, 1, i * 2 + 1)
            plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
            plt.xlim([-0.5, 255.5])
            plt.text(0.03, 0.8, "On Cadence %s"%(i+1), transform=subplot.transAxes, bbox=dict(facecolor='white'))
            subplots.append(subplot)
            if len(wfOff) > i:
                im = cv2.imread('Off%s.png'%i)

                os.remove('Off%s.png'%i)
                # calculate mean value from RGB channels and flatten to 1D array
                vals = im.mean(axis=2).flatten()
                # calculate histogram
                counts, bins = np.histogram(vals, range(257))
                # plot histogram centered on values 0..255
                subplot = plt.subplot(n_plots, 1, i * 2 + 2)
                plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
                plt.xlim([-0.5, 255.5])
                plt.text(0.03, 0.8, "Off Cadence %s"%(i+1), transform=subplot.transAxes, bbox=dict(facecolor='white'))
                subplots.append(subplot)

        plt.savefig(name)
        plt.close(fig[0])

    def plot_frequencies(wfOn, wfOff, name):
        n_plots = len(wfOn) + len(wfOff)
        fig = plt.subplots(n_plots, sharex=True, sharey=True,figsize=(10, 2*n_plots))



        subplots = []




        # read image
        for i in range(0, len(wfOn)):
            Plotting.plot_spectrum(wfOn[i])
            # calculate mean value from RGB channels and flatten to 1D array

            subplot = plt.subplot(n_plots, 1, i * 2 + 1)
            plt.text(0.03, 0.8, "On Cadence %s"%(i+1), transform=subplot.transAxes, bbox=dict(facecolor='white'))
            subplots.append(subplot)

            Plotting.plot_spectrum(wfOff[i])
            subplot = plt.subplot(n_plots, 1, i * 2 + 2)
            subplots.append(subplot)
            plt.text(0.03, 0.8, "Off Cadence %s"%(i+1), transform=subplot.transAxes, bbox=dict(facecolor='white'))
        plt.ylabel("Power [counts]")
        plt.xlabel("Frequency [MHz]")
        plt.savefig(name)
        plt.clf()
        plt.close(fig[0])
