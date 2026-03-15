import sys

import numpy as np
from astropy import constants as c
from astropy import units as u
from matplotlib import pyplot as plt
from scipy.signal import argrelmin, argrelmax

sys.path.insert(0, "/home/2649/repos/ABC-SN/code")
import data_degrading as dg

sys.path.insert(0, "../code")
import spectral_features as sf


def prnt(msg):
    # print(msg)
    return


class SpectrumSNR():
    def __init__(
        self,
        sn_name,
        sn_subtype,
        sn_phase,
        wvl,
        spectrum
    ):
        self.name = sn_name
        self.subtype = sn_subtype
        self.phase = sn_phase
        self.wvl = wvl
        self.spectrum = spectrum
        self.spec_min = np.min(spectrum)
        self.spec_max = np.max(spectrum)

    def summarize(self):
        summary = f"Supernova \"{self.name}\" ({self.subtype}) at phase {self.phase}"
        prnt(summary)

    def minmax_normalize(self):
        self.spectrum -= self.spec_min
        self.spectrum /= (self.spec_max - self.spec_min)
        assert np.max(self.spectrum) == 1
        assert np.min(self.spectrum) == 0

    def minmax_inverse(self):
        self.spectrum *= (self.spec_max - self.spec_min)
        self.spectrum += self.spec_min

    def denoise_gaussian(self, stddev):
        self.denoising_parameter = stddev
        self.signal = dg.special_convolution(self.wvl, self.spectrum, self.wvl, stddev)
        self.noise = self.spectrum - self.signal
        return

    def set_spectral_feature(self):
        line, name = sf.get_spectral_feature(self.subtype, self.phase)
        self.line = line
        self.line_name = name
        prnt(f"Using the {name} feature at {line}Å.")

        if self.subtype == "Ibn":
            self.emission = True
            prnt(f"Finding an emission line.")
        else:
            self.emission = False

        return

    def find_spectral_line(self, feature_search_bounds=(500, 0), minima_i=None, emission=False, plot=False):
        """
        Find a spectral feature on a spectrum.
    
        Arguments:
        feature_search_bounds (tuple) -- This function will look for the spectral feature between `self.line - feature_search_bounds[0]` and `self.line - feature_search_bounds[1]` where `self.line` is the location (in angstroms) of the spectral feature.
        """
        
        # Define the range of wavelength values to look for the flux minimum.
        feature_search_range = (self.line - feature_search_bounds[0], self.line + feature_search_bounds[1])
        prnt(f"Searching for a local minimum in the spectrum between {feature_search_range[0]} and {feature_search_range[1]}...")
        feature_search_inds = np.where(
            np.logical_and(
                self.wvl >= feature_search_range[0],
                self.wvl <= feature_search_range[1],
            ),
        )[0]
        
        # Find the minimum in that range of wavelengths.
        feature_search_fluxes = self.signal[feature_search_inds]
        if emission or self.emission:  # Finding maxima but we're calling them minima... I know, I know...
            minima_ind = argrelmax(feature_search_fluxes)[0]
        else:
            minima_ind = argrelmin(feature_search_fluxes)[0]
        minima_wvl = self.wvl[feature_search_inds][minima_ind]
        potential_velocities = wavelength_to_velocity(minima_wvl, self.line)
        
        if minima_wvl.size == 1:
            chosen_minimum_ind = minima_ind[0]
            chosen_minimum_wvl = minima_wvl[0]
            line_velocity = potential_velocities[0]
            prnt(f"One minimum found at {chosen_minimum_wvl:.3f}Å giving a line velocity of {line_velocity:.0f}km/s.")
        elif minima_wvl.size == 0:
            print("No minima found in the specified region.")
            return
        else:
            prnt(f"{minima_wvl.size} minima found in the specified region.")
            if minima_i is not None:
                minima_i = int(minima_i)
                chosen_minimum_ind = minima_ind[minima_i]
                chosen_minimum_wvl = minima_wvl[minima_i]
                line_velocity = potential_velocities[minima_i]
                prnt(f"Selecting minimum {minima_i}:")
                prnt(f"    {minima_i}: {chosen_minimum_wvl:.3f}Å giving a line velocity of {line_velocity:.0f}km/s")
            else:
                feature_search_range_midpoint = (feature_search_range[0] + feature_search_range[1]) / 2
                middle_minimum_i = np.argmin(np.abs(minima_wvl - feature_search_range_midpoint))
                chosen_minimum_ind = minima_ind[middle_minimum_i]
                chosen_minimum_wvl = minima_wvl[middle_minimum_i]
                line_velocity = potential_velocities[middle_minimum_i]
                prnt(f"Defaulting to the center-most minimum:")
                prnt(f"    {middle_minimum_i}: {chosen_minimum_wvl:.3f}Å giving a line velocity of {line_velocity:.0f}km/s")

        self.line_observed_ind = feature_search_inds[chosen_minimum_ind]
        self.line_observed = chosen_minimum_wvl
        self.line_velocity = line_velocity
        self.line_flux = self.signal[feature_search_inds][chosen_minimum_ind]

        if plot is True:
            fig, ax = self.visualize_minima(minima_wvl, feature_search_range, potential_velocities)
            fig.show()
        return

    def find_spectral_shoulders(
        self,
        blu_shoulder_nudge=0,
        red_shoulder_nudge=0,
        emission=False,
        plot=False,
    ):
        # Split the spectrum into two arrays at the location of the observed
        # spectral feature.
        flux_above = self.signal[self.line_observed_ind:]
        flux_below = self.signal[:self.line_observed_ind]
    
        # The "red shoulder" of the spectral feature will be the first maximum
        # in the fluxes to the right of the location of the spectral feature.
        # The "blue shoulder" of the spectral feature will be the first maximum
        # in the fluxes to the left of the location of the spectral feature.
        if emission or self.emission:
            prnt("Finding emission shoulders.")
            inds_shoulder_red = argrelmin(flux_above)[0]
            inds_shoulder_blu = argrelmin(flux_below)[0]
        else:
            inds_shoulder_red = argrelmax(flux_above)[0]
            inds_shoulder_blu = argrelmax(flux_below)[0]
    
        # # In the event that more than one maximum to the left or right of the
        # # location of the spectral feature is found (which is likely) we take
        # # the first maximum on the left and the first maximum on the right of
        # # the spectral feature.
        self.ind_shoulder_red = inds_shoulder_red[0+red_shoulder_nudge]
        self.ind_shoulder_blu = inds_shoulder_blu[-1-blu_shoulder_nudge]
    
        # Get the wavelength and flux values corresponding to the spectral
        # shoulders. This is what we need to in order to calculate the
        # pseudo-continuum.
        self.wvl_shoulder_red = self.wvl[self.line_observed_ind + self.ind_shoulder_red]
        self.wvl_shoulder_blu = self.wvl[self.ind_shoulder_blu]
        self.flx_shoulder_red = self.signal[self.line_observed_ind + self.ind_shoulder_red]
        self.flx_shoulder_blu = self.signal[self.ind_shoulder_blu]
        
        # Calculate the slope of the pseudo-continuum and return that value. We
        # can threshold it later on with the idea that steeper
        # pseudo-continuums might not be very representative of the continuum.
        rise = self.flx_shoulder_red - self.flx_shoulder_blu
        run = self.wvl_shoulder_red - self.wvl_shoulder_blu
        self.slope_pseudo_cont = rise / run

        if plot is True:
            fig, ax = self.visualize_shoulders()
            fig.show()
        return

    def calc_pEW(self, plot=False):
        # Get the indices of the flux and wavelength arrays between the
        # left/blue and right/red spectral shoulders.
        # These indices denote the entirety of the spectral feature.
        self.interp_ind = np.where(
            np.logical_and(
                self.wvl >= self.wvl_shoulder_blu,
                self.wvl <= self.wvl_shoulder_red,
            )
        )[0]
        
        # We want to create an array of fluxes that are the pseudo-continuum,
        # and we want this to be on the same set of wavelength bins as the
        # original fluxes, so we construct this array `x`. We will linearly
        # interpolate between the left/blue and right/red shoulders at each
        # wavelength in `pc_wvl`.
        self.pc_wvl = self.wvl[self.interp_ind]
        
        # In order to create the pseudo-continuum we simply interpolate
        # linearly between the left/blue shoulder and the right/red shoulder at
        # each wavelength in `pc_wvl`. np.interp does this easily.
        self.xp = [self.wvl_shoulder_blu, self.wvl_shoulder_red]
        self.fp = [self.flx_shoulder_blu, self.flx_shoulder_red]
        self.pseudo_cont = np.interp(self.pc_wvl, self.xp, self.fp)
        
        # Calculate the psuedo-equivalent width (pEW).
        self.pEW_integrand = 1 - (self.pseudo_cont / self.signal[self.interp_ind])
        self.pEW = np.trapz(np.abs(self.pEW_integrand), self.pc_wvl)
        # self.pEW = np.trapz(self.pEW_integrand, self.pc_wvl)
        """
        WWW Write down how we come to this pEA in excruciating detail.
        IT WILL BE PAINFUL TO READ AND WRITE.
        """
        
        # Calculate the depth of the spectral feature. To do this, we need the
        # flux of the pseudo-continuum at the location of the spectral feature,
        # `line_observed`. To find this index, compare `line_observed` to
        # `pc_wvl` because that is the wavelength array that `pseudo_cont`
        # is defined on.
        self.pc_min_ind = np.where(self.pc_wvl == self.line_observed)[0][0]
        self.pc_min = self.pseudo_cont[self.pc_min_ind]
        self.depth = self.pc_min - self.line_flux
        # self.tophat_width = self.pEW / self.depth

        if plot is True:
            fig, ax = self.visualize_pEW()
            fig.show()
        return
    
    def measure_feature_noise(
        self,
        noise_window_blu=100,
        noise_window_red=100,
        useBlu=True,
        useRed=True,
        plot=False,
    ):
        self.useBlu = useBlu
        self.useRed = useRed

        # Calculate the standard deviation of the noise array on the red and
        # blue sides within these wavelength values.
        self.blu_range = self.wvl_shoulder_blu - noise_window_blu, self.wvl_shoulder_blu
        self.red_range = self.wvl_shoulder_red, self.wvl_shoulder_red + noise_window_red
    
        # Indices for the wavelength array corresponding to the noise values
        # that we will take the standard deviation of for finding the noise on
        # the red and blue shoulders of the spectral feature.
        self.blu_inds = np.where(
            np.logical_and(
                self.wvl >= self.blu_range[0],
                self.wvl <= self.blu_range[1],
            )
        )[0]
        self.red_inds = np.where(
            np.logical_and(
                self.wvl >= self.red_range[0],
                self.wvl <= self.red_range[1],
            )
        )[0]
    
        self.blu_stddev = np.std(self.noise[self.blu_inds])
        self.red_stddev = np.std(self.noise[self.red_inds])
        if useRed and useBlu:
            self.N = 0.5 * (self.blu_stddev + self.red_stddev)
        elif useBlu:
            self.N = self.blu_stddev
        elif useRed:
            self.N = self.red_stddev
        else:
            # WWW Can we just let this be the noise?
            # Perhaps we would need to gaurantee some minimum amount of points?
            self.N = np.std(self.noise[self.interp_ind])

        if plot is True:
            fig, axes = self.visualize_feature_noise()
            fig.show()
        return

    def measure_SNR(self, plot=False):
        # Define the "signal" of a spectral feature as the pEW / the range of wavelengths of the entire spectral feature.
        self.feature_wvl_range = self.wvl_shoulder_red - self.wvl_shoulder_blu
        self.S = self.pEW / self.feature_wvl_range
    
        self.SNR = self.S / self.N
        if plot is True:
            fig, axes = self.SNR_diagnostic_visualization()
            fig.show()
        return

    def visualize_minima(self, minima_wvl, feature_search_range, potential_velocities):
        fig, ax = plt.subplots()
        ax.set_title(f"{self.subtype} | {self.name} at {self.phase}")
        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel("Normalized Flux")
        # ax.axhline(y=0, c="k", ls=":")
        ax.plot(self.wvl, self.spectrum, c="k", label="Original Spectrum")
        ax.plot(self.wvl, self.signal, c="tab:orange", marker="o", ms=5, label="Denoised Spectrum")
        ax.set_xlim(
            (
                feature_search_range[0]*.99,
                feature_search_range[1]*1.01,
            )
        )
        ax.vlines(
            x=minima_wvl,
            ymin=ax.get_ylim()[0],
            ymax=ax.get_ylim()[1],
            colors="k",
            linestyles="--",
        )
        for i, (minimum, velocity) in enumerate(zip(minima_wvl, potential_velocities)):
            if minimum == self.line_observed:
                color = "tab:red"
            else:
                color = "k"
            
            ax.annotate(
                f"{i}: ${minimum:.1f}\AA \cdot$ {velocity:.0f} km/s",
                (minimum, np.sum(ax.get_ylim())/2),
                ha="right", va="center", rotation=90, fontsize=11, c=color,
            )
        
        ax.axvline(x=feature_search_range[0], c="k")
        ax.axvline(x=feature_search_range[1], c="k")
        ax.annotate(
            f"{feature_search_range[0]}Å",
            (feature_search_range[0], ax.get_ylim()[1]),
            ha="center", va="bottom", fontweight="bold",
        )
        ax.annotate(
            f"{feature_search_range[1]}Å",
            (feature_search_range[1], ax.get_ylim()[1]),
            ha="center", va="bottom", fontweight="bold",
        )
        # ax.legend()
        return fig, ax

    def visualize_shoulders(self):
        fig, ax = plt.subplots()
        ax.set_title(f"{self.subtype} | {self.name} at {self.phase}")
        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel("Normalized Flux")

        ax.axvline(x=self.wvl_shoulder_red, c="tab:red")
        ax.axvline(x=self.wvl_shoulder_blu, c="tab:blue")
        # margin = 500
        margin = 25
        ax.set_xlim((ax.get_xlim()[0]-margin, ax.get_xlim()[1]+margin))

        ax.plot(self.wvl, self.spectrum, c="k", label="Original Spectrum")
        ax.plot(self.wvl, self.signal, c="tab:orange", marker="o", ms=5, label="Denoised Spectrum")
        
        return fig, ax

    def visualize_pEW(self):
        fig, ax = plt.subplots()
        ax.set_title(f"{self.subtype} | {self.name} at {self.phase}")
        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel("Normalized Flux")

        ax.axvline(x=self.wvl_shoulder_red, c="tab:red")
        ax.axvline(x=self.wvl_shoulder_blu, c="tab:blue")
        ax.set_xlim((ax.get_xlim()[0]-500, ax.get_xlim()[1]+500))

        ax.plot(self.wvl, self.spectrum, c="k", label="Original Spectrum")
        ax.plot(self.wvl, self.signal, c="tab:orange", label="Denoised Spectrum")
        ax.plot(self.pc_wvl, self.pseudo_cont, c="tab:purple", label="Pseudo-continuum")

        ax.fill_between(
            self.pc_wvl,
            y1=self.pseudo_cont,
            y2=self.signal[self.interp_ind],
            color="tab:blue",
            alpha=0.5,
            label=f"pEW = {self.pEW:.4f}",
        )

        feature_midpoint = self.line_flux + self.depth*0.50
        # ax.annotate(
        #     f"pEW\n= {self.pEW:.1f}",
        #     (self.line_observed, 0),
        #     ha="center", va="center"
        # )

        ax.vlines(
            x=self.line_observed,
            ymin=self.line_flux,
            ymax=self.pc_min,
            colors="k", linestyles=":"
        )
        ax.annotate(
            f"Depth = {self.depth:.2f}",
            (self.line_observed, feature_midpoint),
            ha="right", va="center", rotation=90, fontweight="bold",
        )

        # tophat_left = self.line_observed - self.tophat_width/2
        # tophat_right = self.line_observed + self.tophat_width/2
        # tophat_top = self.pc_min
        # tophat_bottom = self.pc_min - self.depth
        # ax.vlines(
        #     x=[tophat_left, tophat_right],
        #     ymin=[tophat_bottom, tophat_bottom],
        #     ymax=[tophat_top, tophat_top],
        #     colors="k",
        # )
        # ax.hlines(
        #     y=[tophat_top, tophat_bottom],
        #     xmin=[tophat_left, tophat_left],
        #     xmax=[tophat_right, tophat_right],
        #     colors="k",
        # )
        # ax.annotate(
        #     f"pEW\n= {self.tophat_width:.2f}",
        #     (self.line_observed, self.pc_min),
        #     ha="center", va="bottom", fontweight="bold",
        # )
        ax.legend(loc="lower right")
        return fig, ax

    def visualize_feature_noise(self):
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, height_ratios=(4, 1))
        axes[0].set_title(f"{self.subtype} | {self.name} at {self.phase}")
        axes[1].set_xlabel("Wavelength [Å]")
        axes[0].set_ylabel("[Normalized Flux]")
        axes[1].set_ylabel("Noise Only\n[Normalized Flux]")
        
        axes[0].axvline(x=self.wvl_shoulder_red, c="tab:red")
        axes[0].axvline(x=self.wvl_shoulder_blu, c="tab:blue")
        axes[0].set_xlim((axes[0].get_xlim()[0]-500, axes[0].get_xlim()[1]+500))
        axes[0].plot(self.wvl, self.spectrum, c="k", label="Original Spectrum")
        axes[0].plot(self.wvl, self.signal, c="tab:orange", label="Denoised Spectrum")
        axes[0].plot(self.pc_wvl, self.pseudo_cont, c="tab:green", label="Pseudo-continuum")
        axes[0].set_ylim(axes[0].get_ylim())

        if self.useBlu:
            axes[0].fill_between(
                self.wvl[self.blu_inds],
                y1=[-1000]*self.blu_inds.size,
                y2=[1000]*self.blu_inds.size,
                color="tab:blue",
                alpha=0.5)
            axes[0].annotate(
                f"\n$\sigma_b = {self.blu_stddev:.4f}$",
                (self.wvl[self.blu_inds[0]], axes[0].get_ylim()[0]),
                ha="right", va="bottom")
        if self.useRed:
            axes[0].fill_between(
                self.wvl[self.red_inds],
                y1=[-1000]*self.red_inds.size,
                y2=[1000]*self.red_inds.size,
                color="tab:red",
                alpha=0.5)
            axes[0].annotate(
                f"\n$\sigma_r = {self.red_stddev:.4f}$",
                (self.wvl[self.red_inds[-1]], axes[0].get_ylim()[0]),
                ha="left", va="bottom")

        axes[1].set_title(f"Noise: $\sigma_N={self.N:.4f}$")
        axes[1].plot(self.wvl, self.noise, c="tab:green")#, label="Extracted Noise")
        axes[1].axvline(x=self.wvl_shoulder_red, c="tab:red")
        axes[1].axvline(x=self.wvl_shoulder_blu, c="tab:blue")
        axes[1].set_ylim(axes[1].get_ylim())
        
        axes[1].fill_between(
            self.wvl[self.blu_inds],
            y1=[-1000]*self.blu_inds.size,
            y2=[1000]*self.blu_inds.size,
            color="tab:blue",
            alpha=0.5
        )
        axes[1].fill_between(
            self.wvl[self.red_inds],
            y1=[-1000]*self.red_inds.size,
            y2=[1000]*self.red_inds.size,
            color="tab:red",
            alpha=0.5
        )
        return fig, axes

    def SNR_diagnostic_visualization(self):
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            height_ratios=(4, 1),
            figsize=(8, 6),
        )
        axes[0].set_title(f"{self.subtype} | {self.name} at {self.phase}")

        axes[1].set_xlabel("Wavelength [Å]")
        axes[0].set_ylabel("Normalized Flux")
        axes[1].set_ylabel("Normalized Flux")

        # Plot the lines designating the start of the red and blue shoulders.
        # Set the xlim (for both axes).
        axes[0].axvline(x=self.wvl_shoulder_red, c="tab:red")
        axes[0].axvline(x=self.wvl_shoulder_blu, c="tab:blue")
        axes[1].axvline(x=self.wvl_shoulder_red, c="tab:red")
        axes[1].axvline(x=self.wvl_shoulder_blu, c="tab:blue")
        axes[0].set_xlim((axes[0].get_xlim()[0]-500, axes[0].get_xlim()[1]+500))

        # Plot the spectrum, signal, pseudo-continuum, and pEW area.
        # Set the ylim for the axes[0].
        axes[0].plot(self.wvl, self.spectrum, c="k", label="Original Spectrum")
        axes[0].plot(self.wvl, self.signal, c="tab:orange", label="Denoised Spectrum")
        axes[0].plot(self.pc_wvl, self.pseudo_cont, c="tab:purple", label="Pseudo-continuum")
        axes[0].set_ylim(axes[0].get_ylim())
        axes[0].fill_between(
            self.pc_wvl,
            y1=self.pseudo_cont,
            y2=self.signal[self.interp_ind],
            color="tab:blue",
            alpha=0.5,
            label=f"pEA = {self.pEW:.4f}",
        )
        

        # Plot the noise.
        # Set the ylim for axes[1].
        axes[1].plot(self.wvl, self.noise, c="tab:green")
        axes[1].set_ylim(axes[1].get_ylim())

        # Color in the regions used to calculate the noise on the red and blue
        # shoulders.
        if self.useBlu:
            axes[0].fill_between(
                self.wvl[self.blu_inds],
                y1=[-1000]*self.blu_inds.size,
                y2=[1000]*self.blu_inds.size,
                color="tab:blue",
                alpha=0.5)
            axes[1].fill_between(
                self.wvl[self.blu_inds],
                y1=[-1000]*self.blu_inds.size,
                y2=[1000]*self.blu_inds.size,
                color="tab:blue",
                alpha=0.5)

        if self.useRed:
            axes[0].fill_between(
                self.wvl[self.red_inds],
                y1=[-1000]*self.red_inds.size,
                y2=[1000]*self.red_inds.size,
                color="tab:red",
                alpha=0.5)
            axes[1].fill_between(
                self.wvl[self.red_inds],
                y1=[-1000]*self.red_inds.size,
                y2=[1000]*self.red_inds.size,
                color="tab:red",
                alpha=0.5)

        # # Draw the equivalent tophat rectangle on the feature.
        # tophat_left = self.line_observed - self.tophat_width/2
        # tophat_right = self.line_observed + self.tophat_width/2
        # tophat_top = self.pc_min
        # tophat_bottom = self.pc_min - self.depth
        # axes[0].vlines(
        #     x=[tophat_left, tophat_right, self.line_observed],
        #     ymin=[tophat_bottom, tophat_bottom, self.line_flux],
        #     ymax=[tophat_top, tophat_top, self.pc_min],
        #     colors="k", linestyles=["--", "--", ":"])
        # axes[0].hlines(
        #     y=[tophat_top, tophat_bottom],
        #     xmin=[tophat_left, tophat_left],
        #     xmax=[tophat_right, tophat_right],
        #     colors="k", linestyles="--")

        diagnostics = f"{self.line_name} at {self.line} Å."
        diagnostics += f"\n$\lambda_e = {self.line}\AA$"
        diagnostics += f"\n$\lambda_o = {self.line_observed:.2f}\AA$"
        diagnostics += f"\n$v = {self.line_velocity:.0f}$ km/s"
        
        diagnostics += f"\n\nLeft/Blue shoulder: {self.wvl_shoulder_blu:.2f} Å"
        diagnostics += f"\nRight/Red shoulder: {self.wvl_shoulder_red:.2f} Å"
        
        diagnostics += f"\n\n$pEW = {self.pEW:.2f}$"
        # diagnostics += f"\n$THW = {self.tophat_width:.2f}\AA$"
        diagnostics += f"\n$d = {self.depth:.2f}$"
        diagnostics += f"\n$S = {self.S:.3f}$"

        diagnostics += f"\n\n$\sigma_b = {self.blu_stddev:.3e}$"
        diagnostics += f"\n$\sigma_r = {self.red_stddev:.3e}$"
        diagnostics += f"\n$N = {self.N:.3e}$"

        diagnostics += f"\n\n$SNR = {self.SNR:.3f}$"
        
        fig.text(
            .91, .85, diagnostics,
            ha="left", va="top"
        )

        # # axes[0].plot(self.pc_wvl, self.pEW_integrand, color="tab:green")
        # axes[0].fill_between(self.pc_wvl, np.abs(self.pEW_integrand), color="tab:green")
        # axes[0].axvline(x=self.pc_wvl[np.argmax(np.abs(self.pEW_integrand))])
        # # axes[0].set_ylim((None, np.abs(self.pEW_integrand).max()))

        return fig, axes


def wavelength_to_velocity(lambda_obs, lambda_em):
    """
    Calculate the velocity of a spectral feature.

    Arguments:
    lambda_obs -- the observed wavelength of a spectreal feature. Units must be the same as lambda_em.
    lambda_em -- the assumed emitted wavelength of the spectral feature. Units must be the same as lambda_obs.
    
    Returns:
    vel -- the velocity of the spectral feature in km/s.
    """
    vel = c.c * ((lambda_obs / lambda_em) - 1)
    return vel.to(u.km / u.s).value
