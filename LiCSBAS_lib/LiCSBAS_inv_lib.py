#!/usr/bin/env python3
"""
v1.4 20200622 Chengyan Fan, Lanzhou University

========
Overview
========
Python3 library of time series inversion functions for LiCSBAS.

=========
Changelog
=========
v1.4 20200622 Chengyan Fan, Lanzhou University
 - GPU support 
v1.3 20200103 Yu Morioshita, Uni of Leeds and GSI
 - Bag fix in calc_stc (return nonzero even if two adjacent pixels have identical ts)
v1.2 20190823 Yu Morioshita, Uni of Leeds and GSI
 - Bag fix in calc_velstd_withnan
 - Remove calc_velstd
v1.1 20190807 Yu Morioshita, Uni of Leeds and GSI
 - Add calc_velsin
v1.0 20190730 Yu Morioshita, Uni of Leeds and GSI
 - Original implementation
"""

import warnings
import numpy as np
import datetime as dt
import multiprocessing as multi
import LiCSBAS_tools_lib as tools_lib


class NumpyRNGContext:
    """Modified from astropy.utils.NumpyRNGContext to support cupy"""

    def __init__(self, seed, gpu=False):
        self.seed = seed
        self.gpu = gpu

    def __enter__(self):
        if self.gpu:
            from cupy import random
        else:
            from numpy import random

        self.startstate = random.get_state()
        random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.gpu:
            from cupy import random
        else:
            from numpy import random

        random.set_state(self.startstate)


def bootstrap(data, bootnum=100, samples=None, bootfunc=None, gpu=False):
    """Modified from astropy.stats.bootstrap to support cupy"""
    if gpu:
        import cupy as xp
    else:
        xp = np

    if samples is None:
        samples = data.shape[0]

    # make sure the input is sane
    if samples < 1 or bootnum < 1:
        raise ValueError("neither 'samples' nor 'bootnum' can be less than 1.")

    try:
        resultdims = (bootnum, len(bootfunc(data)))
    except TypeError:
        resultdims = (bootnum,)

    # create empty boot array
    boot = xp.empty(resultdims)

    for i in range(bootnum):
        bootarr = xp.random.randint(low=0, high=data.shape[0], size=samples)
        if bootfunc is None:
            boot[i] = data[bootarr]
        else:
            boot[i] = bootfunc(data[bootarr])

    return boot


def make_sb_matrix(ifgdates):
    """
    Make small baseline incidence-like matrix.
    Composed of 1 between master and slave. (n_ifg, n_im-1)
    Unknown is incremental displacement.
    """
    imdates = tools_lib.ifgdates2imdates(ifgdates)
    n_im = len(imdates)
    n_ifg = len(ifgdates)

    G = np.zeros((n_ifg, n_im - 1), dtype=np.int16)
    for ifgix, ifgd in enumerate(ifgdates):
        masterdate = ifgd[:8]
        masterix = imdates.index(masterdate)
        slavedate = ifgd[-8:]
        slaveix = imdates.index(slavedate)
        G[ifgix, masterix:slaveix] = 1

    return G


# %%
def make_sb_matrix2(ifgdates):
    """
    Make small baseline incidence-like matrix.
    Composed of -1 at master and 1 at slave. (n_ifg, n_im)
    Unknown is cumulative displacement.
    """
    imdates = tools_lib.ifgdates2imdates(ifgdates)
    n_im = len(imdates)
    n_ifg = len(ifgdates)

    A = np.zeros((n_ifg, n_im), dtype=np.int16)
    for ifgix, ifgd in enumerate(ifgdates):
        masterdate = ifgd[:8]
        masterix = imdates.index(masterdate)
        slavedate = ifgd[-8:]
        slaveix = imdates.index(slavedate)
        A[ifgix, masterix] = -1
        A[ifgix, slaveix] = 1
    return A

# %%


def invert_nsbas(unw, G, dt_cum, gamma, n_core, gpu=False):
    """
    Calculate increment displacement difference by NSBAS inversion. Points with all unw data are solved by simple SB inversion firstly at a time.

    Inputs:
      unw : Unwrapped data block for each point (n_pt, n_ifg)
            Still include nan to keep dimention
      G    : Design matrix (1 between master and slave) (n_ifg, n_im-1)
      dt_cum : Cumulative years(or days) for each image (n_im)
      gamma  : Gamma value for NSBAS inversion, should be small enough (e.g., 0.0001)
      n_core : Number of cores for parallel processing

    Returns:
      inc     : Incremental displacement (n_im-1, n_pt)
      vel     : Velocity (n_pt)
      vconst  : Constant part of linear velocity (c of vt+c) (n_pt)
    """
    global xp
    if n_core != 1:
        global Gall, unw_tmp, mask  # for para_wrapper
    if gpu:
        import cupy as xp
    else:
        xp = np

    # Settings
    n_pt, n_ifg = unw.shape
    n_im = G.shape[1]+1

    result = xp.zeros((n_im+1, n_pt), dtype=xp.float32) * \
        xp.nan  # [inc, vel, const]

    # Set matrix of NSBAS part (bottom)
    Gbl = xp.tril(xp.ones((n_im, n_im-1), dtype=xp.float32),
                  k=-1)  # lower tri matrix without diag
    Gbr = -xp.ones((n_im, 2), dtype=xp.float32)
    Gbr[:, 0] = -dt_cum
    Gb = xp.concatenate((Gbl, Gbr), axis=1)*gamma
    Gt = xp.concatenate((G, xp.zeros((n_ifg, 2), dtype=xp.float32)), axis=1)
    Gall = (xp.concatenate((Gt, Gb))).astype(xp.float32)

    # Solve points with full unw data at a time. Very fast.
    bool_pt_full = xp.all(~xp.isnan(unw), axis=1)
    n_pt_full = int(bool_pt_full.sum())

    if n_pt_full != 0:
        print('  Solving {0:6}/{1:6}th points with full unw at a time...'.format(
            n_pt_full, n_pt), flush=True)

        # Sovle
        unw_tmp = xp.concatenate((unw[bool_pt_full, :], xp.zeros(
            (n_pt_full, n_im), dtype=xp.float32)), axis=1).transpose()
        result[:, bool_pt_full] = xp.linalg.lstsq(Gall, unw_tmp, rcond=-1)[0]

    # Solve other points with nan point by point.
    unw_tmp = xp.concatenate((unw[~bool_pt_full, :], xp.zeros(
        (n_pt-n_pt_full, n_im), dtype=xp.float32)), axis=1).transpose()
    mask = (~xp.isnan(unw_tmp))
    unw_tmp[xp.isnan(unw_tmp)] = 0
    print(
        '  Next, solve {0} points including nan point-by-point...'.format(n_pt-n_pt_full), flush=True)

    if n_core == 1:
        # (n_im+1, n_pt)
        result[:, ~bool_pt_full] = censored_lstsq_slow(
            Gall, unw_tmp, mask)
    else:
        print('  {} parallel processing'.format(n_core), flush=True)

        args = [i for i in range(n_pt-n_pt_full)]
        with multi.Pool(n_core) as p:
            _result = p.map(censored_lstsq_slow_para_wrapper,
                            args)  # list[n_pt][length]
        result[:, ~bool_pt_full] = xp.array(_result).T

    inc = result[:n_im-1, :]
    vel = result[n_im-1, :]
    vconst = result[n_im, :]

    # free memory
    del Gall, unw_tmp, mask

    return inc, vel, vconst


def invert_nsbas_custom(unw, G, dt_cum, gamma, n_core, functions, gpu=False):
    """
    Calculate increment displacement difference by NSBAS inversion. Points with all unw data are solved by simple SB inversion firstly at a time.

    Inputs:
      unw : Unwrapped data block for each point (n_pt, n_ifg)
            Still include nan to keep dimention
      G    : Design matrix (1 between master and slave) (n_ifg, n_im-1)
      dt_cum : Cumulative years(or days) for each image (n_im)
      gamma  : Gamma value for NSBAS inversion, should be small enough (e.g., 0.0001)
      n_core : Number of cores for parallel processing
      functions: a list contains functions that describe the deformation in time series.
                functions should be ufunc(universal function) of numpy. You can use `np.vectorize`
                or `np.frompyfunc` to define your own ufunc.

    Returns:
      inc           : Incremental displacement (n_im-1, n_pt)
      parameters    ： parameters define by functions
    """
    global xp
    if n_core != 1:
        global Gall, unw_tmp, mask  # for para_wrapper
    if gpu:
        import cupy as xp
    else:
        xp = np

    # Settings
    n_pt, n_ifg = unw.shape
    n_im = G.shape[1]+1

    result = xp.zeros((n_im+1, n_pt), dtype=xp.float32) * \
        xp.nan  # [inc, vel, const]

    # Set matrix of NSBAS part (bottom)
    Gbl = xp.tril(xp.ones((n_im, n_im-1), dtype=xp.float32),
                  k=-1)  # lower tri matrix without diag
    Gbr = -xp.empty((n_im, len(functions)), dtype=xp.float32)
    for i, func in enumerate(functions):
        if gpu:
            dt_cum_cp = xp.asnumpy(dt_cum)
            Gbr[:, i] = xp.asarray(func(dt_cum_cp))
        else:
            Gbr[:, i] = func(dt_cum)
    Gb = xp.concatenate((Gbl, Gbr), axis=1)*gamma
    Gt = xp.concatenate((G, xp.zeros((n_ifg, len(functions)),
                                     dtype=xp.float32)), axis=1)
    Gall = (xp.concatenate((Gt, Gb))).astype(xp.float32)

    # Solve points with full unw data at a time. Very fast.
    bool_pt_full = xp.all(~xp.isnan(unw), axis=1)
    n_pt_full = int(bool_pt_full.sum())

    if n_pt_full != 0:
        print('  Solving {0:6}/{1:6}th points with full unw at a time...'.format(
            n_pt_full, n_pt), flush=True)

        # Sovle
        unw_tmp = xp.concatenate((unw[bool_pt_full, :], xp.zeros(
            (n_pt_full, n_im), dtype=xp.float32)), axis=1).transpose()
        result[:, bool_pt_full] = xp.linalg.lstsq(Gall, unw_tmp, rcond=-1)[0]

    # Solve other points with nan point by point.
    unw_tmp = xp.concatenate((unw[~bool_pt_full, :], xp.zeros(
        (n_pt-n_pt_full, n_im), dtype=xp.float32)), axis=1).transpose()
    mask = (~xp.isnan(unw_tmp))
    unw_tmp[xp.isnan(unw_tmp)] = 0
    print(
        '  Next, solve {0} points including nan point-by-point...'.format(n_pt-n_pt_full), flush=True)

    if n_core == 1:
        # (n_im+1, n_pt)
        result[:, ~bool_pt_full] = censored_lstsq_slow(Gall, unw_tmp, mask)
    else:
        print('  {} parallel processing'.format(n_core), flush=True)

        args = [i for i in range(n_pt-n_pt_full)]
        with multi.Pool(n_core) as p:
            _result = p.map(censored_lstsq_slow_para_wrapper,
                            args)  # list[n_pt][length]
        result[:, ~bool_pt_full] = xp.array(_result).T

    inc = result[:n_im-1, :]
    parameters = result[n_im-1:, :]

    # free memory
    del Gall, unw_tmp, mask

    return inc, parameters


def censored_lstsq_slow_para_wrapper(i):
    # Use global value
    if xp.mod(i, 1000) == 0:
        print('  Running {0:6}/{1:6}th point...'.format(i,
                                                        unw_tmp.shape[1]), flush=True)
    m = mask[:, i]  # drop rows where mask is zero
    try:
        X = xp.linalg.lstsq(Gall[m], unw_tmp[m, i], rcond=-1)[0]
    except:
        X = xp.zeros((Gall.shape[1]), dtype=xp.float32)*xp.nan
    return X


def invert_nsbas_wls(unw, var, G, dt_cum, gamma, n_core, gpu=False):
    """
    Calculate increment displacement difference by NSBAS inversion with WLS.

    Inputs:
      unw : Unwrapped data block for each point (n_pt, n_ifg)
            Still include nan to keep dimention
      var : Variance estimated from coherence (n_pt, n_ifg)
      G    : Design matrix (1 between master and slave) (n_ifg, n_im-1)
      dt_cum : Cumulative years(or days) for each image (n_im)
      gamma  : Gamma value for NSBAS inversion, should be small enough (e.g., 0.0001)
      n_core : Number of cores for parallel processing

    Returns:
      inc     : Incremental displacement (n_im-1, n_pt)
      vel     : Velocity (n_pt)
      vconst  : Constant part of linear velocity (c of vt+c) (n_pt)
    """
    global Gall, unw_tmp, var_tmp, mask, xp
    if gpu:
        import cupy as xp
    else:
        xp = np

    # Settings
    n_pt, n_ifg = unw.shape
    n_im = G.shape[1]+1

    result = xp.zeros((n_im+1, n_pt), dtype=xp.float32) * \
        xp.nan  # [inc, vel, const]

    # Set matrix of NSBAS part (bottom)
    Gbl = xp.tril(xp.ones((n_im, n_im-1), dtype=xp.float32),
                  k=-1)  # lower tri matrix without diag
    Gbr = -xp.ones((n_im, 2), dtype=xp.float32)
    Gbr[:, 0] = -dt_cum
    Gb = xp.concatenate((Gbl, Gbr), axis=1)*gamma
    Gt = xp.concatenate((G, xp.zeros((n_ifg, 2), dtype=xp.float32)), axis=1)
    Gall = (xp.concatenate((Gt, Gb))).astype(xp.float32)
    # Make unw_tmp, var_tmp, and mask
    unw_tmp = xp.concatenate(
        (unw, xp.zeros((n_pt, n_im), dtype=xp.float32)), axis=1).transpose()
    mask = (~xp.isnan(unw_tmp))
    unw_tmp[xp.isnan(unw_tmp)] = 0
    # 50 is var for coh=0.1, to scale bottom part of Gall
    var_tmp = xp.concatenate(
        (var, 50*xp.ones((n_pt, n_im), dtype=xp.float32)), axis=1).transpose()

    if n_core == 1:
        for i in range(n_pt):
            result[:, i] = wls_nsbas(i)  # (n_im+1, n_pt)
    else:
        print('  {} parallel processing'.format(n_core), flush=True)

        args = [i for i in range(n_pt)]
        with multi.Pool(n_core) as p:
            _result = p.map(wls_nsbas, args)  # list[n_pt][length]
        result = xp.array(_result).T

    inc = result[:n_im-1, :]
    vel = result[n_im-1, :]
    vconst = result[n_im, :]

    # free memory
    del Gall, unw_tmp, var_tmp, mask

    return inc, vel, vconst


def invert_nsbas_wls_custom(unw, var, G, dt_cum, gamma, n_core, functions, gpu=False):
    """
    Calculate increment displacement difference by NSBAS inversion with WLS.

    Inputs:
      unw : Unwrapped data block for each point (n_pt, n_ifg)
            Still include nan to keep dimention
      var : Variance estimated from coherence (n_pt, n_ifg)
      G    : Design matrix (1 between master and slave) (n_ifg, n_im-1)
      dt_cum : Cumulative years(or days) for each image (n_im)
      gamma  : Gamma value for NSBAS inversion, should be small enough (e.g., 0.0001)
      n_core : Number of cores for parallel processing

    Returns:
      inc     : Incremental displacement (n_im-1, n_pt)
      vel     : Velocity (n_pt)
      vconst  : Constant part of linear velocity (c of vt+c) (n_pt)
    """
    global Gall, unw_tmp, var_tmp, mask, xp
    if gpu:
        import cupy as xp
    else:
        xp = np

    # Settings
    n_pt, n_ifg = unw.shape
    n_im = G.shape[1]+1

    result = xp.zeros((n_im+1, n_pt), dtype=xp.float32) * xp.nan
    # [inc, vel, const]

    # Set matrix of NSBAS part (bottom)
    Gbl = xp.tril(xp.ones((n_im, n_im-1), dtype=xp.float32), k=-1)
    # lower tri matrix without diag

    Gbr = -xp.empty((n_im, len(functions)), dtype=xp.float32)
    for i, func in enumerate(functions):
        if gpu:
            dt_cum_cp = xp.asnumpy(dt_cum)
            Gbr[:, i] = xp.asarray(func(dt_cum_cp))
        else:
            Gbr[:, i] = func(dt_cum)

    Gb = xp.concatenate((Gbl, Gbr), axis=1)*gamma
    Gt = xp.concatenate((G, xp.zeros((n_ifg, len(functions)),
                                     dtype=xp.float32)), axis=1)
    Gall = (xp.concatenate((Gt, Gb))).astype(xp.float32)
    # Make unw_tmp, var_tmp, and mask
    unw_tmp = xp.concatenate((unw, xp.zeros((n_pt, n_im), dtype=xp.float32)),
                             axis=1).transpose()
    mask = (~xp.isnan(unw_tmp))
    unw_tmp[xp.isnan(unw_tmp)] = 0
    # 50 is var for coh=0.1, to scale bottom part of Gall
    var_tmp = xp.concatenate((var, 50 * xp.ones((n_pt, n_im), dtype=xp.float32)),
                             axis=1).transpose()

    if n_core == 1:
        for i in range(n_pt):
            result[:, i] = wls_nsbas(i)  # (n_im+1, n_pt)
    else:
        print('  {} parallel processing'.format(n_core), flush=True)

        args = [i for i in range(n_pt)]
        with multi.Pool(n_core) as p:
            _result = p.map(wls_nsbas, args)  # list[n_pt][length]
        result = xp.array(_result).T

    inc = result[:n_im-1, :]
    parameters = result[n_im-1:, :]

    # free memory
    del Gall, unw_tmp, var_tmp, mask

    return inc, parameters


def wls_nsbas(i):
    # Use global value of Gall, unw_tmp, mask
    if xp.mod(i, 1000) == 0:
        print('  Running {0:6}/{1:6}th point...'.format(i,
                                                        unw_tmp.shape[1]), flush=True)

    # Weight unw and G

    Gall_w = Gall/xp.sqrt((var_tmp[:, i][:, xp.newaxis]).astype(xp.float64))
    unw_tmp_w = unw_tmp[:, i]/xp.sqrt((var_tmp[:, i]).astype(xp.float64))
    m = mask[:, i]  # drop rows where mask is zero

    try:
        X = xp.linalg.lstsq(Gall_w[m], unw_tmp_w[m], rcond=-1)[0]
    except:
        X = xp.zeros((Gall.shape[1]), dtype=xp.float32) * xp.nan
    return X


def calc_vel(cum, dt_cum, gpu=False):
    """
    Calculate velocity.

    Inputs:
      cum    : cumulative phase block for each point (n_pt, n_im)
      dt_cum : Cumulative days for each image (n_im)

    Returns:
      vel    : Velocity (n_pt)
      vconst : Constant part of linear velocity (c of vt+c) (n_pt)
    """
    global xp
    if gpu:
        import cupy as xp
    else:
        xp = np

    n_pt, n_im = cum.shape
    result = xp.zeros((2, n_pt), dtype=xp.float32)*xp.nan  # [vconst, vel]

    G = xp.stack((xp.ones_like(dt_cum), dt_cum), axis=1)
    vconst = xp.zeros((n_pt), dtype=xp.float32)*xp.nan
    vel = xp.zeros((n_pt), dtype=xp.float32)*xp.nan

    bool_pt_full = xp.all(~xp.isnan(cum), axis=1)
    n_pt_full = bool_pt_full.sum()

    if n_pt_full != 0:
        print('  Solving {0:6}/{1:6}th points with full cum at a time...'.format(
            n_pt_full, n_pt), flush=True)

        # Sovle
        result[:, bool_pt_full] = xp.linalg.lstsq(
            G, cum[bool_pt_full, :].transpose(), rcond=-1)[0]

    # Solve other points with nan point by point.
    cum_tmp = cum[~bool_pt_full, :].transpose()
    mask = (~xp.isnan(cum_tmp))
    cum_tmp[xp.isnan(cum_tmp)] = 0
    print(
        '  Next, solve {0} points including nan point-by-point...'.format(n_pt-n_pt_full), flush=True)

    result[:, ~bool_pt_full] = censored_lstsq_slow(
        G, cum_tmp, mask)  # (n_im+1, n_pt)

    vconst = result[0, :]
    vel = result[1, :]

    return vel, vconst


def calc_velsin(cum, dt_cum, imd0, gpu=False):
    """
    Calculate velocity and coeffcients of sin (annual) function.

    Inputs:
      cum    : cumulative phase block for each point (n_pt, n_im)
      dt_cum : Cumulative days for each image (n_im)
      imd0   : Date of first acquistion (str, yyyymmdd)

    Returns:
      vel    : Velocity (n_pt)
      vconst : Constant part of linear velocity (c of vt+c) (n_pt)
      amp    : Amplitude of sin function
      dt     : Time difference of sin function wrt Jan 1 (day)
    """
    global xp
    if gpu:
        import cupy as xp
    else:
        xp = np

    doy0 = (dt.datetime.strptime(imd0, '%Y%m%d') -
            dt.datetime.strptime(imd0[0:4]+'0101', '%Y%m%d')).days

    n_pt, n_im = cum.shape
    result = xp.zeros((4, n_pt), dtype=xp.float32) * \
        xp.nan  # [vconst, vel, coef_s, coef_c]

    sin = xp.sin(2*xp.pi*dt_cum)
    cos = xp.cos(2*xp.pi*dt_cum)
    G = xp.stack((xp.ones_like(dt_cum), dt_cum, sin, cos), axis=1)

    vconst = xp.zeros((n_pt), dtype=xp.float32)*xp.nan
    vel = xp.zeros((n_pt), dtype=xp.float32)*xp.nan
    amp = xp.zeros((n_pt), dtype=xp.float32)*xp.nan
    delta_t = xp.zeros((n_pt), dtype=xp.float32)*xp.nan

    bool_pt_full = xp.all(~xp.isnan(cum), axis=1)
    n_pt_full = bool_pt_full.sum()

    if n_pt_full != 0:
        print('  Solving {0:6}/{1:6}th points with full cum at a time...'.format(
            n_pt_full, n_pt), flush=True)

        # Sovle
        result[:, bool_pt_full] = xp.linalg.lstsq(
            G, cum[bool_pt_full, :].transpose(), rcond=-1)[0]

    # Solve other points with nan point by point.
    cum_tmp = cum[~bool_pt_full, :].transpose()
    mask = (~xp.isnan(cum_tmp))
    cum_tmp[xp.isnan(cum_tmp)] = 0
    print(
        '  Next, solve {0} points including nan point-by-point...'.format(n_pt-n_pt_full), flush=True)

    result[:, ~bool_pt_full] = censored_lstsq_slow(
        G, cum_tmp, mask)  # (n_im+1, n_pt)

    vconst = result[0, :]
    vel = result[1, :]
    coef_s = result[2, :]
    coef_c = result[3, :]

    amp = xp.sqrt(coef_s**2+coef_c**2)
    delta_t = xp.arctan2(-coef_c, coef_s)/2/xp.pi*365.25  # wrt 1st img
    delta_t = delta_t+doy0  # wrt Jan 1
    delta_t[delta_t < 0] = delta_t[delta_t < 0]+365.25  # 0-365.25
    delta_t[delta_t > 365.25] = delta_t[delta_t > 365.25]-365.25

    return vel, vconst, amp, delta_t


def calc_velstd_withnan(cum, dt_cum, gpu=False):
    """
    Calculate std of velocity by bootstrap for each point which may include nan.

    Inputs:
      cum    : Cumulative phase block for each point (n_pt, n_im)
               Can include nan.
      dt_cum : Cumulative days for each image (n_im)

    Returns:
      vstd   : Std of Velocity for each point (n_pt)
    """
    global bootcount, bootnum
    if gpu:
        import cupy as xp
    else:
        xp = np

    n_pt, n_im = cum.shape
    bootnum = 100
    bootcount = 0

    vstd = xp.zeros((n_pt), dtype=xp.float32)
    G = xp.stack((xp.ones_like(dt_cum), dt_cum), axis=1)

    data = cum.transpose().copy()
    ixs_day = xp.arange(n_im)
    mask = (~xp.isnan(data))
    data[xp.isnan(data)] = 0

    def velinv(x):
        return censored_lstsq2(G[x, :], data[x, :], mask[x, :], gpu)[1]

    with NumpyRNGContext(1):
        bootresult = bootstrap(ixs_day, bootnum, bootfunc=velinv, gpu=gpu)

    vstd = xp.nanstd(bootresult, axis=0)
    if gpu:
        vstd = xp.asnumpy(vstd)

    print('')

    return vstd


def censored_lstsq2(A, B, M, gpu=False):
    # http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/
    global bootcount, bootnum
    if gpu:
        import cupy as xp
    else:
        xp = np

    print('\r  Running {0:3}/{1:3}th bootstrap...'.format(bootcount,
                                                          bootnum), end='', flush=True)
    bootcount = bootcount+1

    # if B is a vector, simply drop out corresponding rows in A
    if B.ndim == 1 or B.shape[1] == 1:
        return xp.linalg.leastsq(A[M], B[M])[0]

    # else solve via tensor representation
    rhs = xp.dot(A.T, M * B).T[:, :, None]  # n x r x 1 tensor
    T = xp.matmul(A.T[None, :, :], M.T[:, :, None] *
                  A[None, :, :])  # n x r x r tensor
    try:
        X = xp.squeeze(xp.linalg.solve(T, rhs)).T  # transpose to get r x n
    except:  # In case Singular matrix
        X = xp.zeros((B.shape[1]), dtype=xp.float32) * xp.nan
    return X


# %%
def calc_stc(cum, gpu=False):
    """
    Calculate STC (spatio-temporal consistensy; Hanssen et al., 2008, Terrafirma) of time series of displacement.
    Note that isolated pixels (which have no surrounding pixel) have nan of STC.

    Input:
      cum  : Cumulative displacement (n_im, length, width)

    Return:
      stc  : STC (length, width)
    """
    if gpu:
        import cupy as xp
    else:
        xp = np
    n_im, length, width = cum.shape

    # Add 1 pixel margin to cum data filled with nan
    cum1 = xp.ones((n_im, length+2, width+2), dtype=xp.float32)*xp.nan
    cum1[:, 1:length+1, 1:width+1] = cum

    # Calc STC for surrounding 8 pixels
    _stc = xp.ones((length, width, 8), dtype=xp.float32)*xp.nan
    pixels = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 2]]
    # Left Top = [0, 0], Rigth Bottmon = [2, 2], Center = [1, 1]

    for i, pixel in enumerate(pixels):
        # Spatial difference (surrounding pixel-center)
        d_cum = cum1[:, pixel[0]:length + pixel[0], pixel[1]:width + pixel[1]] - \
            cum1[:, 1:length + 1, 1:width+1]

        # Temporal difference (double difference)
        dd_cum = d_cum[:-1, :, :]-d_cum[1:, :, :]

        # STC (i.e., RMS of DD)
        sumsq_dd_cum = xp.nansum(dd_cum**2, axis=0)
        n_dd_cum = (xp.sum(~xp.isnan(dd_cum), axis=0)
                    ).astype(xp.float32)  # nof non-nan
        n_dd_cum[n_dd_cum == 0] = xp.nan  # to avoid 0 division
        _stc[:, :, i] = xp.sqrt(sumsq_dd_cum/n_dd_cum)

    # Strange but some adjacent pixels can have identical time series,
    # resulting in 0 of stc. To avoid this, replace 0 with nan.
    _stc[_stc == 0] = xp.nan

    # Identify minimum value as final STC
    with warnings.catch_warnings():  # To silence warning by All-Nan slice
        warnings.simplefilter('ignore', RuntimeWarning)
        stc = xp.nanmin(_stc, axis=2)

    return stc


# %%
def censored_lstsq(A, B, M):
    # http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/
    # This is actually slow because matmul does not use multicore...
    # Need multiprocessing.
    # Precison is bad widh bad condition, so this is unfortunately useless for NSABS...
    # But maybe usable for vstd because its condition is good.
    """Solves least squares problem subject to missing data.

    Note: uses a broadcasted solve for speed.

    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)

    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    # Note: we should check A is full rank but we won't bother...

    # if B is a vector, simply drop out corresponding rows in A
    if B.ndim == 1 or B.shape[1] == 1:
        return np.linalg.leastsq(A[M], B[M])[0]

    # else solve via tensor representation
    rhs = np.dot(A.T, M * B).T[:, :, None]  # n x r x 1 tensor
    T = np.matmul(A.T[None, :, :], M.T[:, :, None] *
                  A[None, :, :])  # n x r x r tensor
    return np.squeeze(np.linalg.solve(T, rhs)).T  # transpose to get r x n


# %%
def censored_lstsq_slow(A, B, M):
    # http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/
    """Solves least squares problem subject to missing data.

    Note: uses a for loop over the columns of B, leading to a
    slower but more numerically stable algorithm

    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)

    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    X = xp.empty((A.shape[1], B.shape[1]))
    for i in range(B.shape[1]):
        if xp.mod(i, 1000) == 0:
            print('\r  Running {0:6}/{1:6}th point...'.format(i, B.shape[1]),
                  end='', flush=True)

        m = M[:, i]  # drop rows where mask is zero
        try:
            X[:, i] = xp.linalg.lstsq(A[m], B[m, i], rcond=-1)[0]
        except:
            X[:, i] = xp.nan

    print('')
    return X
