"""
Module for calculating the phasor notation components of a DMD model.
"""

import numpy as np
import copy


def mode_magnitude_reorder(
    dmd,
    delay=1,
):
    """
    Reorders the DMD eigenvalues/vectors by mode magnitude.
    """

    # Sort eigenvalues, modes, and dynamics according to amplitude magnitude.
    mode_order = np.argsort(-np.abs(dmd.amplitudes))
    lead_eigs = dmd.eigs[mode_order]
    lead_modes = dmd.modes[:, mode_order]
    lead_amplitudes = np.abs(dmd.amplitudes[mode_order])

    # Get mode averages across delays if time-delay was used.
    if delay > 1:

        # For the modes, we can simply select phi from the first delay since
        # all delays result in identical W and S.
        lead_modes = lead_modes.reshape(
            delay,
            lead_modes.shape[0] // delay,
            lead_modes.shape[1],
        )
        lead_modes = lead_modes[0]

    else:
        lead_mode_mag = np.sqrt(lead_modes.real**2 + lead_modes.imag**2)
        lead_modes = copy.deepcopy(lead_modes)

    return (
        lead_eigs,
        lead_amplitudes,
        lead_modes,
    )


def calculate_phasor_terms(dmd, tgrid, mode=None, delay=1):
    """
    Calculates the phasor notation terms of a DMD model.
    """

    (
        omega,
        b,
        phi,
    ) = mode_magnitude_reorder(dmd, delay=delay)

    # Return the phasor terms for all modes.
    if mode == None:
        varphi = np.arctan2(
            phi.imag,
            phi.real,
        )
        num_eigs = len(dmd.eigs)
        nt, nx = tgrid.shape
        W = np.zeros((nt, nx, num_eigs))
        for n in range(num_eigs):
            W[:, :, n] = np.cos(omega.imag[n] * tgrid + varphi[:, n])
        S = np.sqrt(phi.real**2 + phi.imag**2)
    # Return the phasor terms for the specified modes.
    else:
        varphi = np.arctan2(
            phi[:, mode].imag,
            phi[:, mode].real,
        )

        W = np.cos(omega.imag[mode] * tgrid + varphi)
        S = np.sqrt(phi.real**2 + phi.imag**2)[:, mode]

    return S, W, b


def mode_reconstruction(dmd, time, mode_index, delay=1):
    """Reconstruct individual components of the DMD"""
    (
        omega,
        b,
        phi,
    ) = mode_magnitude_reorder(dmd, delay=delay)

    omega = np.atleast_2d(omega).T
    phi = np.atleast_2d(phi)

    # Get the size of the data to be reconstructed
    nx = phi.shape[0]
    nt = time.size

    xr = np.zeros((nx, nt), np.complex128)
    for j in mode_index:
        xr += np.linalg.multi_dot(
            [
                np.atleast_2d(phi[:, j]).T,
                np.diag(np.atleast_1d(b[j])),
                np.atleast_2d(np.exp(omega[j] * time)),
            ]
        )

    return xr


def mrc_phasor_components(
    mrc,
):
    """Extract out the mrCOSTS phasor notation components."""

    # Each individual reconstructed window
    decomp_mag_exponential = np.zeros(
        (
            mrc.n_decompositions,
            mrc._n_components_global,
            np.max(mrc._svd_rank_array),
            mrc._n_data_vars,
            mrc._n_time_steps,
        ),
    )

    combined_waveform = np.zeros(
        (
            mrc.n_decompositions,
            mrc._n_components_global,
            np.max(mrc._svd_rank_array),
            mrc._n_data_vars,
            mrc._n_time_steps,
        ),
    )

    b_sep = np.zeros(
        (
            mrc.n_decompositions,
            mrc._n_components_global,
            np.max(mrc._svd_rank_array),
            mrc._n_time_steps,
        ),
        np.complex128,
    )

    omega_classes_list = mrc.multi_res_deterp()
    for n_mrd, mrd in enumerate(mrc._costs_array):
        # Track the total contribution from all windows to each time step
        xn = np.zeros(mrc._n_time_steps)

        # Convolve each windowed reconstruction with a gaussian filter.
        # Std dev of gaussian filter
        recon_filter = mrd.build_kern(
            mrd._window_length,
            relative_filter_length=mrd._relative_filter_length,
        )

        omega_classes = omega_classes_list[n_mrd]

        if mrd.svd_rank < np.max(mrc._svd_rank_array):
            truncate_slice = slice(None, mrd.svd_rank)
            omega_classes = omega_classes[:, truncate_slice]

        # Iterate over each window slide performed.
        for k in range(mrd.n_slides):
            w = mrd.modes_array[k]
            b = mrd.amplitudes_array[k]
            omega = np.atleast_2d(mrd.omega_array[k]).T
            classification = omega_classes[k]

            # Compute each segment of xr starting at "t = 0"
            t = mrd.time_array[k]
            t_start = mrd.time_array[k, 0]
            t = t - t_start

            # Get the indices for this window.
            if k == mrd.n_slides - 1 and mrd._non_integer_n_slide:
                # Handle non-integer number of window slides by slightly
                # shortening the last window's slide.
                window_indices = slice(-mrd.window_length, None)
            else:
                window_indices = slice(
                    k * mrd.step_size,
                    k * mrd.step_size + mrd.window_length,
                )

            for j in np.arange(0, mrc._n_components_global):
                class_bool = classification == j

                # Assign the DMD components
                if any(class_bool):
                    class_ind = np.flatnonzero(class_bool)
                    b_local = b[class_ind].real
                    omega_local = omega[class_ind]
                    phi_local = w[:, class_ind]

                    for r in class_ind:
                        phi_local = np.atleast_2d(w[:, r]).T
                        decomp_mag_exponential[
                            n_mrd, j, r, :, window_indices
                        ] += (
                            np.linalg.multi_dot(
                                [
                                    np.sqrt(
                                        phi_local.real**2 + phi_local.imag**2
                                    ),
                                    np.atleast_2d(b[r].real),
                                    np.atleast_2d(np.exp(omega[r].real * t)),
                                ]
                            )
                            * recon_filter
                        )

                        b_sep[n_mrd, j, r, window_indices] += (
                            np.linalg.multi_dot(
                                [
                                    b[r],
                                    np.exp(omega[r].real * t),
                                ]
                            )
                            * recon_filter
                        )

                        phase_phi_local_real = np.linalg.multi_dot(
                            [
                                np.atleast_2d(w[:, r]).T.real,
                                # np.atleast_2d(b[r]),
                                # np.atleast_2d(np.exp(omega[r].real * t)),
                                np.atleast_2d(np.exp(0 * t)),
                            ]
                        )
                        phase_phi_local_imag = np.linalg.multi_dot(
                            [
                                np.atleast_2d(w[:, r]).T.imag
                                * np.sign(omega[r].imag),
                                # np.atleast_2d(b[r]),
                                # np.atleast_2d(np.exp(omega[r].real * t)),
                                np.atleast_2d(np.exp(0 * t)),
                            ]
                        )

                        local_varphi = np.arctan2(
                            phase_phi_local_imag.real, phase_phi_local_real.real
                        )

                        combined_waveform[n_mrd, j, r, :, window_indices] += (
                            np.exp(omega[r].real * t).real
                            * b[r].real
                            * np.cos(
                                omega[r].imag * t * np.sign(omega[r].imag)
                                + local_varphi
                            )
                        ) * recon_filter

            # A normalization factor which weights the global reconstruction
            # by the number of window centers it contains. This accounts
            # for the convolution above.
            xn[window_indices] += recon_filter

        # Normalize by the reconstruction filter.
        decomp_mag_exponential[n_mrd, :, :, :, :] = (
            decomp_mag_exponential[n_mrd, :, :, :, :] / xn
        )
        b_sep[n_mrd] = b_sep[n_mrd] / xn
        combined_waveform[n_mrd] = combined_waveform[n_mrd] / xn

    return (
        decomp_mag_exponential,
        b_sep.real,
        combined_waveform,
    )
