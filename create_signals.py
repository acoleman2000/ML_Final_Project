from abc import ABC, abstractmethod

import setigen as stg
import random
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import itertools
from plotting import Plotting, font
import os

import math
df = 2.7939677238464355
class Signal_Function(ABC):
    def __init__(self, type) -> None:
        self.type = type
        self.function = None

    @abstractmethod
    def set_function(self):
        pass

    def get_function(self):
        return self.function


class Path(Signal_Function):
    def __init__(self, type, index, freqs, drift_rate, spread = None) -> None:
        super().__init__(type)
        self.index = index
        self.freqs = freqs
        self.drift_rate = drift_rate
        self.spread = spread
        self.set_function()

    def set_function(self):
        if self.type == "constant":
            self.function = stg.constant_path(f_start=self.freqs[self.index], drift_rate=self.drift_rate*u.Hz/u.s)
        else:
            self.function = stg.simple_rfi_path(f_start=self.freqs[self.index],
                                drift_rate=self.drift_rate,
                                spread=self.spread*u.Hz,
                                spread_type='uniform',
                                rfi_type=self.type)
    def set_index(self, index):
        self.index = index
        self.set_function()



class Intensity(Signal_Function):
    def __init__(self, type, level, amplitude=None, period = None) -> None:
        super().__init__(type)
        self.level = level
        if amplitude:
            self.amplitude = amplitude
            self.period = period

        self.set_function()

    def set_function(self):
        if self.type == "constant":
            self.function = stg.constant_t_profile(level=self.level)
        else:
            self.function = stg.sine_t_profile(period=self.period*u.s,
                                             amplitude=self.amplitude,
                                             level=self.level)

class F_profile(Signal_Function):
    def __init__(self, type, width, trunc = None) -> None:
        super().__init__(type)
        self.width = width
        if trunc != None:
            self.trunc = trunc
        self.set_function()


    def set_function(self):
        if self.type == "box":
            self.function = stg.box_f_profile(width=self.width*u.Hz)
        elif self.type == "gaussian":
            self.function = stg.gaussian_f_profile(width=self.width*u.Hz)
        elif self.type == "multiple_gaussian":
            self.function = stg.multiple_gaussian_f_profile(width=self.width*u.Hz)
        elif self.type == "sinc":
            self.function = stg.sinc2_f_profile(width=self.width*u.Hz, trunc=self.trunc)

class Signal(object):


    rfi_paths = ["constant", "random_walk","stationary"]
    intensities = ["constant", "sine"]
    f_profiles = ["gaussian", "multiple_gaussian","sinc", "box"]


    def __init__(self, index, drift_rate, df, args) -> None:
        self.cur_index = index
        self.df = df
        self.drift_rate = drift_rate
        self.path = Path(*args["path"])
        self.intensity = Intensity(*args["intensity"])
        self.f_profile = F_profile(*args["f_profile"])
        self.bp_profile = stg.constant_bp_profile(level=1)


    def update_index(self, time, max_index):
        freq_shift = (time * self.drift_rate)
        index_shift = round(freq_shift/df)
        self.cur_index += index_shift
        self.cur_index = max(0, self.cur_index)
        self.cur_index = min(self.cur_index, max_index)
        self.path.set_index(self.cur_index)

    def get_args(self):
        args = []
        args.append(self.path.get_function())
        args.append(self.intensity.get_function())
        args.append(self.f_profile.get_function())
        args.append(self.bp_profile)
        return args
    @staticmethod
    def generate_signal(fchans, interesting, freqs,qual_df,time):
        args = {
            "path": [],
            "intensity": [],
            "f_profile": []
        }
        # Set up path args
        drift_rate = random.randrange(-2,2)
        if interesting:
            index = random.randrange(0, fchans)
        else:
            freq_shift = (time * drift_rate)
            index_shift = abs(round(freq_shift/df))
            index = random.randrange(index_shift * 2, fchans - index_shift * 2)
        if interesting:
            args["path"].append("constant")
        else:
            args["path"].append(random.choice(Signal.rfi_paths))

        args["path"].append(index)
        args["path"].append(freqs)
        args["path"].append(drift_rate)

        if args["path"][0] != "constant":
            args["path"].append(random.randint(0, 400))

        # Set up intensity args
        args["intensity"].append(random.choice(Signal.intensities))
        args["intensity"].append(random.randint(1, 30))

        if args["intensity"][0] == "sine":
            args["intensity"].append(random.randint(1, 4))
            args["intensity"].append(random.randint(50, 200))

        # Set up profile args
        args["f_profile"].append(random.choice(Signal.f_profiles))
        args["f_profile"].append(random.randint(20, 300))
        if args["f_profile"][0] == "sinc":
            args["f_profile"].append(random.randint(0, 1) == 0)

        return Signal(index,drift_rate, qual_df, args)


class Cadence(object):
    def __init__(self, interesting_signals, uninteresting_signals, fch1, fchans = 1024, tchans = 16, df =df*u.Hz, dt = 18.253611008*u.s, x_mean = 10) -> None:

        self.time = tchans *  18.253611008
        self.frame = stg.Frame(fchans=fchans,
                            tchans=tchans,
                            df = df,
                            dt = dt,
                            fch1=fch1)
        self.frame.add_noise(x_mean=x_mean, noise_type='chi2')

        self.max_freq = fchans - 1
        self.interesting_signals = interesting_signals
        self.uninteresting_signals = uninteresting_signals


    def insert_signals(self):

        for signal in self.interesting_signals:
            args = signal.get_args()
            if signal.cur_index > 0 and signal.cur_index < self.max_freq:
                self.frame.add_signal(*args)
                signal.update_index(self.time * 2,self.max_freq)

        for signal in self.uninteresting_signals:
            args = signal.get_args()
            if signal.cur_index > 0 and signal.cur_index < self.max_freq:
                self.frame.add_signal(*args)
                signal.update_index(self.time,self.max_freq)
        self.frame._update_waterfall()

class CadenceGroup(object):

    def __init__(self, save_dir, name, num_interesting = 0, num_rfi = 1,fchans = 1024, df= 2.7939677238464355*u.Hz, tchans = 16, dt = 18.253611008*u.s) -> None:
        fch1 = random.uniform(1000, 10000)*u.MHz

        frame = stg.Frame(fchans=fchans,
                            tchans=tchans,
                            df = df,
                            dt = dt,
                            fch1=fch1)

        self.interesting_signals = []
        for _ in itertools.repeat(None, num_interesting):
            self.interesting_signals.append(Signal.generate_signal(fchans= fchans, interesting=True, freqs=frame.fs, qual_df=df, time = tchans *  18.253611008))


        self.uninteresting_signals = []
        for _ in itertools.repeat(None, num_rfi):
            self.uninteresting_signals.append(Signal.generate_signal(fchans= fchans, interesting=False, freqs=frame.fs, qual_df=df, time = tchans *  18.253611008))

        self.cadences = []

        for i in range(0, 6):
            if i % 2 == 0:
                self.cadences.append(Cadence(interesting_signals=self.interesting_signals, uninteresting_signals=self.uninteresting_signals,fch1=fch1,fchans=fchans,tchans=tchans,df=df,dt=dt))
            else:
                self.cadences.append(Cadence(interesting_signals=[], uninteresting_signals=self.uninteresting_signals,fch1=fch1,fchans=fchans,tchans=tchans,df=df,dt=dt))
        self.save_dir = save_dir
        self.name = name

    def update_frames(self):
        for cadence in self.cadences:
            cadence.insert_signals()



    def save_waterfall_plots(self, manifest):

        subplots = []
        fig = plt.subplots(6, sharex=True, sharey=True,figsize=(10, 12))
        for i, cadence in enumerate(self.cadences):
                subplot = plt.subplot(6, 1, i + 1)
                subplots.append(subplot)
                this_plot = Plotting.plot_waterfall(cadence.frame.waterfall, "")
        f_start = cadence.frame.waterfall.get_freqs()[-1]
        f_stop = cadence.frame.waterfall.get_freqs()[0]
        # mid_f = np.abs(f_start+f_stop)/2.
        # factor = 1e6
        # units = 'Hz'

        # xloc = np.linspace(f_start, f_stop, 5)
        # xticks = [round(loc_freq) for loc_freq in (xloc - mid_f)*factor]
        # if np.max(xticks) > 1000:
        #     xticks = [xt/1000 for xt in xticks]
        #     units = 'kHz'
        # plt.xticks(xloc, xticks)
        # plt.xlabel("Relative Frequency [%s] from %f MHz"%(units,mid_f),fontdict=font)


        # cax = fig[0].add_axes([0.94, 0.11, 0.03, 0.77])
        # print(cax)
        # fig[0].colorbar(this_plot,cax=cax,label='Normalized Power (Arbitrary Units)')

        plt.subplots_adjust(hspace=0,wspace=0)

        if not os.path.exists("%s"%(self.save_dir)):
            os.makedirs("%s"%(self.save_dir))

        plt.savefig("%s%s.png"%(self.save_dir, self.name))
        plt.close(fig[0])
        classifcation = "interesting" if len(self.interesting_signals) > 0 else "uninteresting"
        with open(manifest, "a") as f:
           f.write("%s,%s\n"%("%s/waterfall/%s.png"%(self.save_dir, self.name), classifcation))

    def save_h5_files(self):
        for i, cadence in enumerate(self.cadences):
            frame_index = "On" if i % 2 == 0 else "Off"
            frame_index += "_" + str(math.floor(1 + 0.5 * (i)))


            if not os.path.exists("%s%s/"%(self.save_dir, self.name)):
                os.makedirs("%s%s/"%(self.save_dir, self.name))


            cadence.frame.save_h5(filename='%s%s/frame_%s.h5'%(self.save_dir, self.name, frame_index))