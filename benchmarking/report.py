# (c) David Novak 2024

# Helper functions for 04_report.ipynb
# We use code from funkyheatmappy to monkey-patch its `fh.funkyheatmappy.calculate_positions` function for our purposes.
# funkyheatmappy was written by Artuur Couckuyt and Louise Deconinck
# GitHub repo: https://github.com/funkyheatmap/funkyheatmappy

import os
import re
import copy
import pandas as pd
import funkyheatmappy as fh
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from adjustText import adjust_text
import warnings
warnings.filterwarnings('ignore')

def collect_dicts(
    datasets,
    methods,
    zdim=2,
    nruns=5
):
    """Collect RNX curve values, Local SP and Global SP per set-up in a dictionary"""
    def _collect_dict(denoised=False):
        rnx = {}
        sl = {}
        sg = {}
        for dataset in datasets:
            d_rnx = {}
            d_sl = {}
            d_sg = {}
            for method in methods:
                m_rnx = [np.load(f'results/{dataset}_{method}_z{zdim}_u{denoised}/rnx_curve_seed{seed}.npy', allow_pickle=True) for seed in range(1, nruns+1)]        
                m_sl = [np.load(f'results/{dataset}_{method}_z{zdim}_u{denoised}/sp_local_seed{seed}.npy', allow_pickle=True) for seed in range(1, nruns+1)]
                m_sg = [np.load(f'results/{dataset}_{method}_z{zdim}_u{denoised}/sp_global_seed{seed}.npy', allow_pickle=True) for seed in range(1, nruns+1)]
                d_rnx.update({method: m_rnx})
                d_sl.update({method: m_sl})
                d_sg.update({method: m_sg})
            rnx.update({dataset: d_rnx})
            sl.update({dataset: d_sl})
            sg.update({dataset: d_sg})
        return rnx, sl, sg
    rnx = {}
    sl = {}
    sg = {}
    rnx_uFalse, sl_uFalse, sg_uFalse = _collect_dict(denoised=False)
    rnx_uTrue, sl_uTrue, sg_uTrue = _collect_dict(denoised=True)
    rnx.update({'uFalse': rnx_uFalse})
    rnx.update({'uTrue':  rnx_uTrue})
    sl.update({'uFalse': sl_uFalse})
    sl.update({'uTrue':  sl_uTrue})
    sg.update({'uFalse': sg_uFalse})
    sg.update({'uTrue':  sg_uTrue})
    return rnx, sl, sg

def collect_df_times_avg(
    datasets,
    methods,
    nruns=5,
    zdim=2,
    wide=True # wide or long format
):
    """Collect running times means and standard deviations per set-up in a data frame"""
    res = []
    for dataset in datasets:
        for method in methods:
            for denoised in [False, True]:
                fpath    = os.path.join('results', f'{dataset}_{method}_z{zdim}_u{denoised}')

                times  = np.array([np.load(os.path.join(fpath, f'time_seed{seed}.npy'), allow_pickle=True) for seed in range(1, nruns+1)])
                if wide:
                    res.append([
                        dataset,
                        method,
                        zdim,
                        denoised,
                        np.mean(times), np.std(times)
                    ])
                else:
                    res.append([dataset, method, zdim, denoised, 'RunningTime', 'Mean', np.mean(times)])
                    res.append([dataset, method, zdim, denoised, 'RunningTime', 'SD', np.std(times)])

    if wide:
        df_avg = pd.DataFrame(columns=[
            'Dataset', 'Method', 'zdim', 'Denoised',
            'RunningTime_Mean', 'RunningTime_SD'
        ], data = res)
    else:
        df_avg = pd.DataFrame(columns=[
            'Dataset', 'Method', 'zdim', 'Denoised',
            'id', 'stat', 'value'
        ], data = res)

    return df_avg

def collect_df_avg(
    datasets,
    methods,
    nruns=5,
    zdim=2,
    balanced_measure='harmonic_mean',
    wide=True # wide or long format
): 
    """Collect mean Local, Global and Balanced Structure Preservation (SP) scores and standard deviations in a data frame

    Reports means and standard deviations across repeated runs.

    Here we need to decide between using geometric mean or harmonic mean (F-score) to compute the Balanced SP (specify in `balanced_measure` as 'harmonic_mean' or 'geometric_mean').
    We use the harmonic mean, but it is debatable which one is more valid.

    Also, the function returns the data frame in wide format by default (`wide=True`), but long format (suitable, for instance, for making ggplot figures in R) can be used instead.
    """
    res = []
    for dataset in datasets:
        for method in methods:
            for denoised in [False, True]:
                fpath    = os.path.join('results', f'{dataset}_{method}_z{zdim}_u{denoised}')

                localsp  = np.array([np.load(os.path.join(fpath, f'sp_local_seed{seed}.npy'), allow_pickle=True) for seed in range(1, nruns+1)])
                globalsp = np.array([np.load(os.path.join(fpath, f'sp_global_seed{seed}.npy'), allow_pickle=True) for seed in range(1, nruns+1)])
                if balanced_measure=='geometric_mean':
                    balsp = np.array([np.sqrt(localsp[i]*globalsp[i]) for i in range(nruns)])
                elif balanced_measure=='harmonic_mean':
                    balsp = np.array([2*(localsp[i]*globalsp[i])/(localsp[i]+globalsp[i]) for i in range(nruns)])
                if wide:
                    res.append([
                        dataset,
                        method,
                        zdim,
                        denoised,
                        np.mean(localsp), np.std(localsp),
                        np.mean(globalsp), np.std(globalsp), 
                        np.mean(balsp), np.std(balsp)
                    ])
                else:
                    res.append([dataset, method, zdim, denoised, 'LocalSP', 'Mean', np.mean(localsp)])
                    res.append([dataset, method, zdim, denoised, 'LocalSP', 'SD', np.std(localsp)])
                    res.append([dataset, method, zdim, denoised, 'GlobalSP', 'Mean', np.mean(globalsp)])
                    res.append([dataset, method, zdim, denoised, 'GlobalSP', 'SD', np.std(globalsp)])
                    res.append([dataset, method, zdim, denoised, 'BalancedSP', 'Mean', np.mean(balsp)])
                    res.append([dataset, method, zdim, denoised, 'BalancedSP', 'SD', np.std(balsp)])
    if wide:
        df_avg = pd.DataFrame(columns=[
            'Dataset', 'Method', 'zdim', 'Denoised',
            'LocalSP_Mean', 'LocalSP_SD',
            'GlobalSP_Mean', 'GlobalSP_SD',
            'BalancedSP_Mean', 'BalancedSP_SD',
        ], data = res)
    else:
        df_avg = pd.DataFrame(columns=[
            'Dataset', 'Method', 'zdim', 'Denoised',
            'id', 'stat', 'value'
        ], data = res)

    return df_avg

def collect_df_all(
    datasets,
    methods,
    nruns=5,
    zdim=2,
    balanced_measure='harmonic_mean',
    wide=True # wide or long format
): 
    """Collect all Local, Global and Balanced Structure Preservation (SP) scores in a data frame

    Reports all values, i.e., does not reduce to means and standard deviations across repeated runs.

    Here we need to decide between using geometric mean or harmonic mean (F-score) to compute the Balanced SP (specify in `balanced_measure` as 'harmonic_mean' or 'geometric_mean').
    We use the harmonic mean, but it is debatable which one is more valid.

    Also, the function returns the data frame in wide format by default (`wide=True`), but long format (suitable, for instance, for making ggplot figures in R) can be used instead.
    """
    res = []
    for dataset in datasets:
        for method in methods:
            for denoised in [False, True]:
                fpath    = os.path.join('results', f'{dataset}_{method}_z{zdim}_u{denoised}')

                localsp  = np.array([np.load(os.path.join(fpath, f'sp_local_seed{seed}.npy'), allow_pickle=True) for seed in range(1, nruns+1)])
                globalsp = np.array([np.load(os.path.join(fpath, f'sp_global_seed{seed}.npy'), allow_pickle=True) for seed in range(1, nruns+1)])
                if balanced_measure=='geometric_mean':
                    balsp = np.array([np.sqrt(localsp[i]*globalsp[i]) for i in range(nruns)])
                elif balanced_measure=='harmonic_mean':
                    balsp = np.array([2*(localsp[i]*globalsp[i])/(localsp[i]+globalsp[i]) for i in range(nruns)])

                if wide:
                    for i in range(len(localsp)):
                        res.append([dataset, method, zdim, denoised, localsp[i], globalsp[i], balsp[i]])#, xnpemean[i]])
                else:
                    for i in range(len(localsp)):
                        res.append([dataset, method, zdim, denoised, 'LocalSP', localsp[i]])
                    for i in range(len(globalsp)):
                        res.append([dataset, method, zdim, denoised, 'GlobalSP', globalsp[i]])
                    for i in range(len(balsp)):
                        res.append([dataset, method, zdim, denoised, 'BalancedSP', balsp[i]])
    if wide:
        df_all = pd.DataFrame(columns=[
            'Dataset', 'Method', 'zdim', 'Denoised', 'LocalSP', 'GlobalSP', 'BalancedSP'
        ], data = res)
    else:
        df_all = pd.DataFrame(columns=[
            'Dataset', 'Method', 'zdim', 'Denoised', 'id', 'value'
        ], data = res)
    return df_all

def get_denoised_mask(
    df,
    denoised
):
    """Get a Boolean mask for rows in a data frame based on data de-noising criterion

    The `df` argument is a data frame with scores obtained via `collect_df_avg or `collect_df_all`.
    The `denoised` argument is either False (get results from non-denoised), True (from de-noised) or 'ViVAE' (from de-noised for models starting with 'ViVAE', otherwise non-denoised).
    """
    if denoised==True or denoised==False:
        mask_denoised = np.array(df['Denoised']==denoised)
    elif denoised=='ViVAE':
        mask_denoised = np.logical_or.reduce([
            np.logical_and.reduce([df['Denoised']==True, pd.Series([x.startswith('ViVAE') for x in df['Method']])]),
            np.logical_and.reduce([df['Denoised']==False, pd.Series([not x.startswith('ViVAE') for x in df['Method']])])
        ])
    return mask_denoised

def prepare_denoising_data(datasets, methods, rnx, sl, sg):
    rnx_lims = {}
    l_vals0 = {}
    l_vals1 = {}
    l_diffs = {}
    g_vals0 = {}
    g_vals1 = {}
    g_diffs = {}

    l_diff_lims = [0, 0]
    g_diff_lims = [0, 0]

    for dataset in datasets:

        rnx_max = 0.

        this_l_vals0 = {}
        this_l_vals1 = {}
        this_l_diffs = {}

        this_g_vals0 = {}
        this_g_vals1 = {}
        this_g_diffs = {}

        for method in methods:

            this_rnx_max = np.max(rnx['uFalse'][dataset][method])
            if this_rnx_max>rnx_max:
                rnx_max = this_rnx_max

            this_l_vals0.update({method: np.array([x for x in sl['uFalse'][dataset][method]])})
            this_l_vals1.update({method: np.array([x for x in sl['uTrue'][dataset][method]])})
            this_l_diffs.update({method: np.mean(this_l_vals1[method])-np.mean(this_l_vals0[method])})

            this_g_vals0.update({method: np.array([x for x in sg['uFalse'][dataset][method]])})
            this_g_vals1.update({method: np.array([x for x in sg['uTrue'][dataset][method]])})
            this_g_diffs.update({method: np.mean(this_g_vals1[method])-np.mean(this_g_vals0[method])})

            l_diff_min = np.min(this_l_diffs[method])
            if l_diff_min<l_diff_lims[0]:
                l_diff_lims[0] = l_diff_min
            g_diff_min = np.min(this_g_diffs[method])
            if g_diff_min<g_diff_lims[0]:
                g_diff_lims[0] = g_diff_min

            l_diff_max = np.max(this_l_diffs[method])
            if l_diff_max>l_diff_lims[1]:
                l_diff_lims[1] = l_diff_max
            g_diff_max = np.max(this_g_diffs[method])
            if g_diff_max>g_diff_lims[1]:
                g_diff_lims[1] = g_diff_max
        
        rnx_lims.update({dataset: rnx_max})
        l_vals0.update({dataset: this_l_vals0})
        l_vals1.update({dataset: this_l_vals1})
        l_diffs.update({dataset: this_l_diffs})
        g_vals0.update({dataset: this_g_vals0})
        g_vals1.update({dataset: this_g_vals1})
        g_diffs.update({dataset: this_g_diffs})

    return rnx_lims, l_vals0, l_vals1, l_diffs, g_vals0, g_vals1, g_diffs, l_diff_lims, g_diff_lims



def plot_separate_sp(datasets, methods, df_all, palette, denoised='ViVAE'):
    """Plot Local, Global and Balanced Structure Preservation (SP) separately

    By default, we take results for ViVAE run on de-noised inputs and results for other methods on original inputs (`denoised='ViVAE'`).
    The resulting figure is exported as a PNG and SVG file: `report/01_sp_separate.png` and `report/01_sp_separate.svg`.

    Args:
        datasets (List): Names of datasets.
        methods (List): Names of evaluated methods.
        df_all (pandas.DataFrame): Corresponding data frame from `collect_df_all`.
        denoised (str or bool, optional): Whether to use de-noised (smoothed) inputs: either True, False or 'ViVAE' (use them for ViVAE). Defaults to 'ViVAE'.
        palette (List, optional): Colour palette. Defaults to `palette_methods` defined above.
    """
    mpl.rcParams['axes.linewidth'] = 0.1
    fig, ax = plt.subplots(nrows=len(datasets), ncols=3, figsize=(.65*len(datasets), .85*len(methods)), sharey=True, dpi=150)
    fig.subplots_adjust(hspace=.5)
    for i, dataset in enumerate(datasets):

        for j, method in enumerate(methods):

            mask_dataset = df_all['Dataset']==dataset
            mask_method = df_all['Method']==method
            mask_denoised = get_denoised_mask(df_all, denoised=denoised)
            mask = np.logical_and.reduce([mask_dataset, mask_method, mask_denoised])

            ## Local SP
            d = df_all['LocalSP'][mask]
            ax[i][0].errorbar(x=np.mean(d), y=j, xerr=np.std(d), label=method, color=palette[j], markersize=1.8, alpha=.9, fmt='o', linewidth=1.6)
            ax[i][0].grid(visible=True, axis='x', ls='--')

            ## Global SP
            d = df_all['GlobalSP'][mask]
            ax[i][1].errorbar(x=np.mean(d), y=j, xerr=np.std(d), label=method, color=palette[j], markersize=1.8, alpha=.9, fmt='o', linewidth=1.6)
            ax[i][1].grid(visible=True, axis='x', ls='--')

            ## Balanced SP
            d = df_all['BalancedSP'][mask]
            ax[i][2].errorbar(x=np.mean(d), y=j, xerr=np.std(d), label=method, color=palette[j], markersize=1.8, alpha=.9, fmt='o', linewidth=1.6)
            ax[i][2].grid(visible=True, axis='x', ls='--')

        ax[i][0].set_yticks(ticks=range(len(methods)), labels=methods)
        ax[i][1].yaxis.set_tick_params(left=False)
        ax[i][2].yaxis.set_tick_params(left=False)
        for j in [0, 1, 2]:
            ax[i][j].tick_params(axis='x', labelsize=5)
            ax[i][j].tick_params(axis='y', labelsize=4)
    pad = 5
    for a, row in zip(ax[:,0], datasets):
            a.annotate(row, xy=(0, .5), xytext=(0, 0), xycoords=a.yaxis.label, textcoords='offset points', size=6, weight='bold', ha='right', va='center')
    for a, col in zip(ax[0], ['Local SP', 'Global SP', 'Balanced SP']):
            a.annotate(col, xy=(.4, 1), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size=6, weight='bold', ha='left', rotation=45, va='baseline')
    
    fig.savefig('./report/01_sp_separate.png', bbox_inches='tight', dpi=300, transparent=True)
    fig.savefig('./report/01_sp_separate.svg', bbox_inches='tight', transparent=True)
    plt.close()

def plot_sp_tradeoffs(datasets, methods, df_avg, df_all, palette, markers, show_sd=False, text_labels=False, legend=True, denoised='ViVAE', pointsize=8):
    """Plot balance between Local and Global Structure Preservation (SP)

    By default, we take results for ViVAE run on de-noised inputs and results for other methods on original inputs (`denoised='ViVAE'`).
    The resulting figures per dataset are first exported as PNG and SVG files: report/01_sp_tradeoffs/01_sp_tradeoff_{dataset}.[svg|png]
    The resulting figure is exported as a PNG file: `report/01_sp_tradeoffs.png`.


    Args:
        datasets (List): Names of datasets.
        methods (List): Names of evaluated methods.
        df_all (pandas.DataFrame): Corresponding data frame from `collect_df_all`.
        denoised (str or bool, optional): Whether to use de-noised (smoothed) inputs: either True, False or 'ViVAE' (use them for ViVAE). Defaults to 'ViVAE'.
        palette (List, optional): Colour palette. Defaults to `palette_methods` defined above.
    """
    png_names = []
    fpath_plots = './report/01_sp_tradeoffs/'
    if not os.path.exists(fpath_plots):
            os.mkdir(fpath_plots)
    for i, dataset in enumerate(datasets):
        fig, ax = plt.subplots(figsize=(1.8,1.8), dpi=150)
        xlims = np.array([np.min(df_all['LocalSP'])-.05, np.max(df_all['LocalSP'])+.05])
        ylims = np.array([np.min(df_all['GlobalSP'])-.05, np.max(df_all['GlobalSP'])+.05])
        
        mask_dataset = df_avg['Dataset']==dataset
        xcoords = []
        ycoords = []
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        for j, method in enumerate(methods):
            mask_method = df_avg['Method']==method
            if denoised==True or denoised==False:
                mask_denoised = df_avg['Denoised']==denoised
            elif denoised=='ViVAE':
                mask_denoised = df_avg['Denoised']==True if method.startswith('ViVAE') else df_avg['Denoised']==False
            mask = np.logical_and.reduce([mask_dataset, mask_method, mask_denoised])

            mu_local     = df_avg['LocalSP_Mean'][mask]
            mu_global    = df_avg['GlobalSP_Mean'][mask]
            sigma_local  = df_avg['LocalSP_SD'][mask]
            sigma_global = df_avg['GlobalSP_SD'][mask]

            xcoords.append(mu_local)    
            ycoords.append(mu_global)

            if show_sd:
                ellipse = mpl.patches.Ellipse(xy=(mu_local, mu_global), width=sigma_local, height=sigma_global, color=palette[j], alpha=.6)
                ax.add_patch(ellipse)
        
            if i==(len(datasets)-1) and legend:
                ax.scatter(x=mu_local, y=mu_global, s=pointsize, marker=markers[j], linewidth=.8, facecolor='None', color=palette[j], alpha=.8, zorder=2, label=method)
            else:
                ax.scatter(x=mu_local, y=mu_global, s=pointsize, marker=markers[j], linewidth=.8, facecolor='None', color=palette[j], alpha=.8, zorder=2)

            ax.set_xlabel('Local SP', fontsize=5)
            ax.set_ylabel('Global SP', fontsize=5)

            if i>0:
                ax.get_yaxis().set_visible(False)

        method_labs = copy.deepcopy(methods)
    
        ## Pareto front
        x = np.array([1.-xc for xc in xcoords])
        y = np.array([1.-yc for yc in ycoords])
        pareto_idcs = []
        for idx in range(len(x)):
            is_pareto = True
            for jdx in range(len(x)):
                if idx != jdx and x[jdx] <= x[idx] and y[jdx] <= y[idx]:
                    is_pareto = False
                    break
            if is_pareto:
                pareto_idcs.append(idx)

        if i==(len(datasets)-1) and legend:
            fig.legend(bbox_to_anchor=(1.5, .9), prop={'size': 5})

        if len(pareto_idcs)>0:
            x = np.array(xcoords)[pareto_idcs].flatten()
            y = np.array(ycoords)[pareto_idcs].flatten()
            o = np.argsort(x)
            x=x[o]
            y=y[o]
            pareto_x = np.hstack((0., x, x[-1]))
            pareto_y = np.hstack((y[0], y, 0.))
            ax.plot(pareto_x.flatten(), pareto_y.flatten(), zorder=0, linewidth=.5, color='darkgrey', linestyle='dashed')

        ax.plot()

        if text_labels:
            texts = []
            for k, (x, y, s) in enumerate(zip(xcoords, ycoords, method_labs)):
                texts.append(plt.text(x, y, s, color=palette[k], size=4, path_effects=[pe.withStroke(linewidth=.2, foreground='black')]))
            adjust_text(texts, force_points=.0, force_text=.1, arrowprops=dict(arrowstyle='-', color='lightgray', alpha=.5, lw=.8))
        
        ax.set_title(dataset, fontsize=8, fontweight='bold')
        ax.tick_params(axis='both', labelsize=5)

        fig.savefig(f'./report/01_sp_tradeoffs/01_sp_tradeoff_{dataset}.svg', bbox_inches='tight', transparent=True, pad_inches=.05)
        png_name = f'./report/01_sp_tradeoffs/01_sp_tradeoff_{dataset}.png'
        fig.savefig(png_name, bbox_inches='tight', dpi=600, transparent=True, pad_inches=.05)
        png_names.append(png_name)
        
 
    imgs = [Image.open(path) for path in png_names]
    w = sum(img.width for img in imgs)
    h = max(img.height for img in imgs)
    merged_img = Image.new(mode=imgs[0].mode, size=(w, h))
    x_offset = 0
    for img in imgs:
        merged_img.paste(img, (x_offset, 0))
        x_offset += img.width
    merged_img.save('./report/01_sp_tradeoffs.png')

## Monkey-patching funkyheatmappy function

def calculate_positions(
    data,
    column_info,
    row_info,
    column_groups,
    row_groups,
    palettes,
    position_args,
    scale_column,
    add_abc,
):
    row_height = position_args["row_height"]
    row_space = position_args["row_space"]
    row_bigspace = position_args["row_bigspace"]
    col_width = position_args["col_width"]
    col_space = position_args["col_space"]
    col_bigspace = position_args["col_bigspace"]
    col_annot_offset = position_args["col_annot_offset"]-1.5
    col_annot_angle = position_args["col_annot_angle"]

    # Determine row positions
    if not "group" in row_info.columns or all(pd.isna(row_info["group"])):
        row_info["group"] = ""
        row_groups = pd.DataFrame({"group": [""]})
        plot_row_annotation = False
    else:
        plot_row_annotation = True

    row_pos = fh.calculate_row_positions.calculate_row_positions(
        row_info=row_info,
        row_height=row_height,
        row_space=row_space,
        row_bigspace=row_bigspace,
    )

    # Determine column positions
    if not "group" in column_info.columns or all(pd.isna(column_info["group"])):
        column_info["group"] = ""
        column_groups = pd.DataFrame({"group": [""]})
        plot_column_annotation = False
    else:
        plot_column_annotation = True

    column_pos = fh.calculate_column_positions.calculate_column_positions(
        column_info=column_info, col_space=col_space, col_bigspace=col_bigspace
    )

    # Process data
    data_processor = fh.make_data_processor.make_data_processor(
        data=data,
        column_pos=column_pos,
        row_pos=row_pos,
        scale_column=scale_column,
        palette_list=palettes,
    )

    def circle_fun(dat):
        dat = dat.assign(x0=dat["x"], y0=dat["y"], r=row_height / 2 * dat["value"])
        return dat

    circle_data = data_processor("circle", circle_fun)

    def rect_fun(dat):
        return dat

    rect_data = data_processor("rect", rect_fun)

    def funkyrect_fun(dat):
        result = pd.concat(
            [
                fh.score_to_funkyrectangle.score_to_funkyrectangle(
                    xmin=row["xmin"],
                    xmax=row["xmax"],
                    ymin=row["ymin"],
                    ymax=row["ymax"],
                    value=row["value"],
                    midpoint=0.8,
                )
                for _, row in dat[["xmin", "xmax", "ymin", "ymax", "value"]].iterrows()
            ]
        )
        return result

    funkyrect_data = data_processor("funkyrect", funkyrect_fun)

    def bar_fun(dat):
        dat = fh.add_column_if_missing.add_column_if_missing(dat, hjust=0)
        dat = dat.assign(
            xmin=dat["xmin"] + (1 - dat["value"]) * dat["xwidth"] * dat["hjust"],
            xmax=dat["xmax"] - (1 - dat["value"]) * dat["xwidth"] * (1 - dat["hjust"]),
        )
        return dat

    bar_data = data_processor("bar", bar_fun)

    def barguides_fun(dat):
        dat = ((dat.groupby("column_id").first())[["xmin", "xmax"]]).melt(
            var_name="col", value_name="x"
        )
        dat = dat.assign(xend=dat["x"])[["x", "xend"]]
        cols_to_add = pd.DataFrame({"y": row_pos["ymin"], "yend": row_pos["ymax"]})
        result = (
            pd.merge(dat.assign(key=1), cols_to_add.assign(key=1), on="key")
            .drop("key", axis=1)
            .sort_values(["x", "xend"])
            .reset_index(drop=True)
            .drop_duplicates()
            .assign(palette=np.nan, value=np.nan)
        )
        return result

    segment_data = data_processor("bar", barguides_fun).assign(
        colour="black", size=0.5, linestyle="dashed"
    )

    def text_fun(dat):
        dat = dat.assign(color="black")
        return dat

    text_data = data_processor("text", text_fun)

    def pie_fun(dat):
        result = pd.DataFrame()
        for _, row in dat.iterrows():
            value_df = pd.DataFrame(row["value"], index=["end_angle"]).transpose()
            pctgs = value_df["end_angle"] / value_df["end_angle"].sum()
            value_df = (value_df / value_df.sum()) * 360
            value_df = value_df.cumsum().fillna(0)
            value_df["start_angle"] = value_df["end_angle"].shift(1).fillna(0)
            value_df = value_df.loc[value_df["start_angle"] != value_df["end_angle"], :]

            end_angle = (-1 * value_df["start_angle"] + 90) % 360
            start_angle = (-1 * value_df["end_angle"] + 90) % 360
            value_df["start_angle"], value_df["end_angle"] = start_angle, end_angle

            value_df["height"] = row_height / 2
            value_df["x0"] = row["x"]
            value_df["y0"] = row["y"]
            value_df["row_id"] = row["row_id"]
            value_df["value"] = value_df.index
            value_df["pctgs"] = pctgs
            result = pd.concat([result, value_df])
        result = result.dropna(subset="value", axis=0)
        dat = result.merge(dat.drop("value", axis=1), on=["row_id"], how="left")
        return dat

    pie_data = data_processor("pie", pie_fun)

    def image_fun(dat):
        dat = dat.assign(y0=dat["y"] - row_height, height=row_height, width=row_height)
        return dat

    image_data = data_processor("image", image_fun)

    # Add Annotations
    if plot_row_annotation:
        row_annotation = row_groups.melt(
            id_vars="group", var_name="level", value_name="name"
        ).merge(row_pos[["group", "ymin", "ymax"]], how="left", on="group")

        row_annotation = pd.DataFrame(
            {
                "ymin": row_annotation.groupby("name").apply(lambda x: min(x["ymin"])),
                "ymax": row_annotation.groupby("name").apply(lambda x: max(x["ymax"])),
            }
        )
        row_annotation["y"] = (row_annotation["ymin"] + row_annotation["ymax"]) / 2
        row_annotation["xmin"] = -0.5
        row_annotation["xmax"] = 5
        row_annotation = row_annotation[
            (~pd.isna(row_annotation.index)) & (row_annotation.index != "")
        ]

        text_data_rows = pd.DataFrame(
            {
                "xmin": row_annotation["xmin"],
                "xmax": row_annotation["xmax"],
                "ymin": row_annotation["ymax"] + row_space,
                "label_value": [re.sub("\n", " ", x) for x in row_annotation.index],
                "ha": 0,
                "va": 0.5,
                "fontweight": "bold",
                "ymax": (row_annotation["ymax"] + row_space) + row_height,
            }
        )
        text_data = pd.concat([text_data, text_data_rows])

    if plot_column_annotation:
        col_join = column_groups.melt(
            id_vars=["group", "palette"], var_name="level", value_name="name"
        ).merge(column_pos[["group", "xmin", "xmax"]], how="left", on="group")
        text_pct = 0.9
        level_heights = pd.DataFrame(
            col_join.groupby("level").apply(lambda x: max(x["name"].str.count("\n"))),
            columns=["max_newlines"],
        )
        level_heights["height"] = (level_heights["max_newlines"] + 1) * text_pct + (
            1 - text_pct
        )
        level_heights["levelmatch"] = pd.Series(
            [column_groups.columns.tolist().index(x) for x in level_heights.index],
            index=level_heights.index,
            name="level",
        )
        level_heights = level_heights.sort_values(["levelmatch"], ascending=False)
        level_heights["ysep"] = row_space
        level_heights["ymax"] = (
            col_annot_offset
            + (level_heights["height"] + level_heights["ysep"]).cumsum()
            - level_heights["ysep"]
        )
        level_heights["ymin"] = level_heights["ymax"] - level_heights["height"]
        level_heights["y"] = (level_heights["ymin"] + level_heights["ymax"]) / 2
        palette_mids = {
            x: palettes[x][round(len(palettes[x]) / 2)]
            if isinstance(palettes[x], list)
            else list(palettes[x].values())[round(len(palettes[x]) / 2)]
            for x in palettes.keys()
        }
        max_newlines = (
            col_join.groupby("level")
            .apply(lambda x: x["name"].str.count("\n").max())
            .transpose()
        )
        column_annotation = col_join.merge(
            max_newlines.rename("max_newlines"), on="level", how="left"
        )
        xmin = column_annotation.groupby(
            ["level", "name", "palette"], dropna=False
        ).apply(lambda x: min(x["xmin"]))
        xmax = column_annotation.groupby(
            ["level", "name", "palette"], dropna=False
        ).apply(lambda x: max(x["xmax"]))
        column_annotation = (
            pd.concat(
                [
                    xmin.index.to_frame(),
                    xmin.rename("xmin"),
                    xmax.rename("xmax"),
                    ((xmin + xmax) / 2).rename("x"),
                ],
                axis=1,
            )
        ).reset_index(drop=True)

        column_annotation = column_annotation.merge(
            level_heights, on="level", how="left"
        )
        column_annotation = column_annotation[~pd.isna(column_annotation["name"])]
        column_annotation = column_annotation[
            column_annotation["name"].str.contains("[a-zA-Z]")
        ]
        column_annotation["colour"] = [
            palette_mids[col] for col in column_annotation["palette"]
        ]
        rect_data = pd.concat(
            [
                rect_data,
                pd.DataFrame(
                    {
                        "xmin": column_annotation["xmin"],
                        "xmax": column_annotation["xmax"],
                        "ymin": column_annotation["ymin"],
                        "ymax": column_annotation["ymax"],
                        "colour": column_annotation["colour"],
                        "alpha": [
                            1 if lm == 0 else 0.25
                            for lm in column_annotation["levelmatch"]
                        ],
                        "border": False,
                    }
                ),
            ]
        )
        text_data = pd.concat(
            [
                text_data,
                pd.DataFrame(
                    {
                        "xmin": column_annotation["xmin"] + col_space,
                        "xmax": column_annotation["xmax"] - col_space,
                        "ymin": column_annotation["ymin"],
                        "ymax": column_annotation["ymax"],
                        "va": 0.5,
                        "ha": 0.5,
                        "fontweight": [
                            "bold" if lm == 0 else np.nan
                            for lm in column_annotation["levelmatch"]
                        ],
                        "colour": [
                            "white" if lm == 0 else "black"
                            for lm in column_annotation["levelmatch"]
                        ],
                        "label_value": column_annotation["name"],
                        "size": 3. ##FLAG
                    }
                ),
            ]
        )

        if add_abc:
            alphabet = list(map(chr, range(97, 123)))
            c_a_df = (
                column_annotation[column_annotation["levelmatch"] == 0]
                .sort_values("x")
                .reset_index(drop=True)
            )
            text_data_abc = pd.DataFrame(
                {
                    "xmin": c_a_df["xmin"] + col_space,
                    "xmax": c_a_df["xmax"] - col_space,
                    "ymin": c_a_df["ymin"],
                    "ymax": c_a_df["ymax"],
                    "va": 0.5,
                    "ha": 0,
                    "fontweight": "bold",
                    "colour": "white",
                    "label_value": [alphabet[i] + ")" for i in c_a_df.index],
                }
            )
            text_data = pd.concat([text_data, text_data_abc])

    # Add column names
    df = column_pos[column_pos["name"] != ""]
    if df.shape[0] > 0:
        df_column_segments = pd.DataFrame(
            {"x": df["x"], "xend": df["x"], "y": -0.3, "yend": -0.1, "size": 4.}
        )
        segment_data = pd.concat([segment_data, df_column_segments])
        df_column_text = pd.DataFrame(
            {
                "xmin": df["xmin"],
                "xmax": df["xmax"],
                "ymin": 0,
                "ymax": col_annot_offset,
                "angle": col_annot_angle,
                "va": 0,
                "ha": 0,
                "label_value": df["name"],
                "size": 4.
            }
        )
        text_data = pd.concat([text_data, df_column_text])

    # Determine plotting window
    minimum_x = min(
        [
            min(lst, default=np.nan)
            for lst in [
                column_pos["xmin"],
                segment_data["x"],
                segment_data["xend"],
                rect_data["xmin"],
                circle_data["x"] - circle_data["r"],
                funkyrect_data["x"] - funkyrect_data["r"],
                pie_data["xmin"],
                text_data["xmin"],
            ]
        ]
    )

    maximum_x = max(
        [
            max(lst, default=np.nan)
            for lst in [
                column_pos["xmax"],
                segment_data["x"],
                segment_data["xend"],
                rect_data["xmax"],
                circle_data["x"] + circle_data["r"],
                funkyrect_data["x"] + funkyrect_data["r"],
                pie_data["xmax"],
                text_data["xmax"],
            ]
        ]
    )

    minimum_y = min(
        [
            min(lst, default=np.nan)
            for lst in [
                row_pos["ymin"],
                segment_data["y"],
                segment_data["yend"],
                rect_data["ymin"],
                circle_data["y"] - circle_data["r"],
                funkyrect_data["y"] - funkyrect_data["r"],
                pie_data["ymin"],
                text_data["ymin"],
            ]
        ]
    )

    maximum_y = max(
        [
            max(lst, default=np.nan)
            for lst in [
                row_pos["ymax"],
                segment_data["y"],
                segment_data["yend"],
                rect_data["ymax"],
                circle_data["y"] + circle_data["r"],
                funkyrect_data["y"] + funkyrect_data["r"],
                pie_data["ymax"],
                text_data["ymax"],
            ]
        ]
    )



    # Simplify certain geoms
    if funkyrect_data.shape[0] > 0:
        circle_data = pd.concat(
            [
                circle_data,
                funkyrect_data[
                    ~np.isnan(funkyrect_data["start"])
                    & (funkyrect_data["start"] < 1e-10)
                    & (2 * np.pi - 1e-10 < funkyrect_data["end"])
                ][["x", "y", "r", "colour"]],
            ]
        )
        funkyrect_data = funkyrect_data[
            ~(
                ~np.isnan(funkyrect_data["start"])
                & (funkyrect_data["start"] < 1e-10)
                & (2 * np.pi - 1e-10 < funkyrect_data["end"])
            )
        ]

    rect_data = pd.concat([rect_data, bar_data])
    return {
        "row_pos": row_pos,
        "column_pos": column_pos,
        "segment_data": segment_data,
        "rect_data": rect_data,
        "circle_data": circle_data,
        "funkyrect_data": funkyrect_data,
        "pie_data": pie_data,
        "text_data": text_data,
        "image_data": image_data,
        "bounds": {
            "minimum_x": minimum_x,
            "maximum_x": maximum_x,
            "minimum_y": minimum_y,
            "maximum_y": maximum_y,
        },
        "viz_params": row_space,
    }

fh.funkyheatmappy.calculate_positions = calculate_positions

def plot_funky_heatmap(df, datasets, methods, denoised='ViVAE', geom='funkyrect', scale_column=True):
    plt.figure(dpi=150)

    #def rescale_to_range(vals, lims):
    #    return vals-np.min(vals)/(np.max(vals)-np.min(vals))*(lims[1]-lims[0])+lims[0]

    ## Re-scale values for plotting

    mask_denoised = get_denoised_mask(df=df, denoised=denoised)

    dd = copy.deepcopy(df)
    dd = dd[mask_denoised]
    dd = dd[['Dataset', 'Method', 'LocalSP_Mean', 'GlobalSP_Mean', 'BalancedSP_Mean']]
    dd = dd.rename(columns={'LocalSP_Mean': 'LocalSP', 'GlobalSP_Mean': 'GlobalSP', 'BalancedSP_Mean': 'BalancedSP'})
    dd[['LocalSP', 'GlobalSP', 'BalancedSP']] = MinMaxScaler().fit_transform(dd[['LocalSP', 'GlobalSP', 'BalancedSP']])

    ## Gather data for funky heatmap

    dhf = pd.DataFrame.from_dict({'id': methods})
    dhf.index = methods

    col_groups = [pd.NA]
    col_names = ['']
    col_geoms = ['text']
    col_pals = [np.nan]
    col_opts = [{'ha': 1., 'width':4., 'size': 4}]
    col_legs = [False]

    for i, dataset in enumerate(datasets):
        mask = dd['Dataset']==dataset
        d = dd[mask]
        d = d[['LocalSP', 'GlobalSP', 'BalancedSP']]
        d = d.rename(columns={'LocalSP': f'LocalSP_{dataset}', 'GlobalSP': f'GlobalSP_{dataset}', 'BalancedSP': f'BalancedSP_{dataset}'})
        col_groups.extend([dataset, dataset, dataset])
        col_names.extend(['Local', 'Global', 'Balanced'])
        col_geoms.extend([geom, geom, geom])
        palette_to_use = f'palette{np.mod(i,2)+1}'
        col_pals.extend([palette_to_use, palette_to_use, palette_to_use])
        col_opts.extend([{}, {}, {}])
        col_legs.extend([False, False, False])
        d.index = list(methods)
        dhf = pd.concat([dhf, d], axis=1)

    col_info_df = pd.DataFrame({
        'id': list(dhf.columns),
        'group': col_groups,
        'name': col_names,
        'geom': col_geoms,
        'options': col_opts,
        'palette': col_pals,
        'legend': col_legs
    })
    col_info_df.index = col_info_df['id']

    group_pal = np.tile(['palette1', 'palette2'], np.ceil(len(datasets)/2).astype(int))
    col_groups_df = pd.DataFrame({
        'Category': datasets,
        'group': datasets,
        'palette': group_pal
    })
    row_info_df = pd.DataFrame({'id': dhf['id'], 'group': ''}, index=dhf['id'])
    row_groups_df = pd.DataFrame({'Group': [''], 'group': ['']})

    norm = mpl.colors.Normalize(vmin=0, vmax=101, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='Greys')
    colours_bg = [mapper.to_rgba(i) for i in range(0, 101)]

    palettes = pd.DataFrame.from_dict({
        'palettes': ['overall', 'palette1', 'palette2'],
        'colours': [colours_bg, 'Blues', 'Reds']
    })

    pos_args = {
        'row_height': 1,
        'row_space': 0.1,
        'row_bigspace': 0.5,
        'col_width': 1,
        'col_space': 0.1,
        'col_bigspace': 0.5,
        'col_annot_offset': 3,
        'col_annot_angle': 30,
        'expand_xmin': 0,
        'expand_xmax': 2,
        'expand_ymin': 0,
        'expand_ymax': 0,
        'font_size': 4
    }

    __annotations__ = fh.funky_heatmap(
        data=dhf,
        column_info=col_info_df,
        column_groups=col_groups_df,
        row_info=row_info_df,
        row_groups=row_groups_df,
        palettes=palettes,
        position_args=pos_args,
        add_abc=False,
        scale_column=scale_column
    )
    
    plt.title('A', x=.05, y=.88, size=36, weight='bold')

    plt.savefig('report/02_funky_heatmap.svg')
    plt.savefig('report/02_funky_heatmap.png', dpi=300, transparent=True)

def plot_rnx_curves(
    rnx,
    datasets,
    methods,
    denoised='ViVAE',
    fname=['./report/03_rnx_curves.svg', './report/03_rnx_curves.png']
):
    """Plot RNX curve approximations"""
    if denoised==True:
        this_rnx = copy.deepcopy(rnx['uTrue'])
    elif denoised==False:
        this_rnx = copy.deepcopy(rnx['uFalse'])
    elif denoised=='ViVAE': # use denoised inputs for ViVAE
        this_rnx = copy.deepcopy(rnx['uFalse'])
        k = list(this_rnx.keys())
        for i, key in enumerate(k):
            if key.startswith('ViVAE'):
                this_rnx[key] = rnx['uTrue'][key]

    mpl.rcParams['axes.linewidth'] = 0.2
    fig, ax = plt.subplots(nrows=len(datasets), ncols=len(methods), figsize=(.8*len(methods), .7*len(datasets)), sharey=True, sharex=False, dpi=150)
    fig.subplots_adjust(hspace=.5)
    for i, dataset in enumerate(datasets):
        for j, method in enumerate(methods):
            curves = this_rnx[dataset][method]
            for k, curve in enumerate(curves):
                d = curve
                ax[i][j].axhline(y=0, xmin=0, xmax=len(curve)+2, lw=.2, ls='--', color='gray')
                ax[i][j].plot(range(len(curve)), d, lw=.5)
                ax[i][j].xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: ('{:,.0f}'.format(x/1000) + 'K') if x>0 else '0'))
                ax[i][j].tick_params(axis='both', labelsize=5)

    pad = 5
    for a, col in zip(ax[0], methods):
        a.annotate(col, xy=(.4, 1), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size=6, weight='bold', ha='left', rotation=45, va='baseline')
    for a, row in zip(ax[:,0], datasets):
        a.annotate(row, xy=(0, .5), xytext=(0, 0), xycoords=a.yaxis.label, textcoords='offset points', size=6, weight='bold', ha='right', va='center')

    fig.show()
    if fname is not None:
        if isinstance(fname, list) and len(fname)>0:
            for f in fname:
                if f.endswith('.png'):
                    fig.savefig(f, bbox_inches='tight', dpi=300)
                else:
                    fig.savefig(f, bbox_inches='tight')
        elif isinstance(fname, str):
                fig.savefig(fname, bbox_inches='tight')

def plot_embeddings(datasets, methods, palette, denoised='ViVAE', seed=1,
        fname_embedding='./report/04_embeddings.png',
        fpath_legends='./report/04_legends'
    ):
    ## Plot embeddings
    fig, ax = plt.subplots(nrows=len(datasets), ncols=len(methods), figsize=(1.4*len(methods), 1.4*len(datasets)), dpi=300)
    fig.subplots_adjust(hspace=.1, wspace=.1)
    handles = []
    labels = []
    for i, dataset in enumerate(datasets):
        labs = np.load(f'./data/{dataset}_labels.npy', allow_pickle=True)
        unas = np.load(f'./data/{dataset}_unassigned.npy', allow_pickle=True)
        if len(unas.shape)==1 and len(unas)==1:
            unas = unas.item()
        else:
            unas = None
        idcs0 = np.array([])
        idcs1 = np.arange(len(labs))
        s = 5e-2 if len(labs)>30000 else 1e-1
        if unas is not None:
            idcs0 = np.where(labs == unas)[0]
            idcs1 = np.delete(idcs1, idcs0)
            labs = np.delete(labs, idcs0)
        for j, method in enumerate(methods):
            if denoised==False or denoised==True:
                emb = np.load(f'./results/{dataset}_{method}_z2_u{denoised}/emb_seed{seed}.npy', allow_pickle=True)
            elif denoised=='ViVAE':
                if method.startswith('ViVAE'):
                    emb = np.load(f'./results/{dataset}_{method}_z2_uTrue/emb_seed1.npy', allow_pickle=True)
                else:
                    emb = np.load(f'./results/{dataset}_{method}_z2_uFalse/emb_seed1.npy', allow_pickle=True)
            
            if len(idcs0)>0:
                ax[i][j].scatter(emb[idcs0,0], emb[idcs0,1], s=s, c='#bfbfbf', alpha=1., marker='o', linewidths=0)
            idx_pop = 0
            for pop in np.unique(labs):
                idcs = np.where(labs == pop)[0]
                ax[i][j].scatter(emb[idcs1[idcs],0], emb[idcs1[idcs],1], label=pop, s=s, c=palette[idx_pop], alpha=1., marker='o', linewidths=0)
                ax[i][j].tick_params(axis='both', labelbottom=False, labelleft=False, bottom=False, left=False)
                idx_pop += 1
            ax[i][j].axis('equal')
        this_handles, this_labels = ax[i][0].get_legend_handles_labels()
        handles.append(this_handles)
        labels.append(this_labels)
    pad = 5
    for a, col in zip(ax[0], methods):
        a.annotate(col, xy=(.5, 1), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size=6, weight='bold', ha='center', va='baseline')
    for a, row in zip(ax[:,0], datasets):
        a.annotate(row, xy=(0, .5), xytext=(0, 0), xycoords=a.yaxis.label, textcoords='offset points', size=6, weight='bold', ha='right', va='center')
    if fname_embedding is not None:
        if isinstance(fname_embedding, list) and len(fname_embedding)>0:
            for f in fname_embedding:
                if f.endswith('.png'):
                    fig.savefig(f, bbox_inches='tight', dpi=300)
                else:
                    fig.savefig(f, bbox_inches='tight')
        elif isinstance(fname_embedding, str):
                fig.savefig(fname_embedding, bbox_inches='tight')

        ## Plot legends
        if not os.path.exists(fpath_legends):
            os.mkdir(fpath_legends)
        for i, dataset in enumerate(datasets):
            plt.clf()
            plt.axis(False)
            l = plt.legend(handles[i], labels[i], loc='center', title=f'{dataset}', title_fontproperties={'weight': 'bold'})
            for lh in l.legend_handles: 
                lh.set_alpha(1.)
                lh._sizes = [50]
            l._legend_box.align = 'center'
            plt.savefig(os.path.join(fpath_legends, f'04_legend_{dataset}.png'), bbox_inches='tight', dpi=300, pad_inches=.05)
            plt.savefig(os.path.join(fpath_legends, f'04_legend_{dataset}.svg'), bbox_inches='tight', dpi=300, pad_inches=.05)
    plt.close()

def plot_denoising_sp_change(
    datasets,
    methods,
    l_diffs,
    g_diffs,
    l_diff_lims,
    g_diff_lims
):
    fig, ax = plt.subplots(ncols=len(methods), nrows=len(datasets), figsize=(len(methods)*.7, len(datasets)), dpi=150)
    fig.subplots_adjust(wspace=.4)
    mpl.rcParams['axes.linewidth'] = 0.1

    for i, dataset in enumerate(datasets):
        for j, method in enumerate(methods):
            
            ax[i][j].set_ylim((-np.max(np.abs(l_diff_lims)), np.max(np.abs(l_diff_lims))))
            ax[i][j].set_xlim((.0, .15))
            ax[i][j].hlines(y=0., xmin=.0, xmax=.2, color='grey', ls='--', lw=1.)
            if i==0 and j==0:
                ax[i][j].bar(x=.05, width=.04, height=l_diffs[dataset][method], color='purple', label='Local SP change')
            else:
                ax[i][j].bar(x=.05, width=.04, height=l_diffs[dataset][method], color='purple')
            ax[i][j].set_xticks([])
            ax[i][j].tick_params(axis='both', labelsize=5)
            ax2 = ax[i][j].twinx()

            if j>0:
                ax[i][j].set_yticks([])

            ax2.set_ylim((-np.max(np.abs(g_diff_lims)), np.max(np.abs(g_diff_lims))))
            ax2.hlines(y=0., xmin=.0, xmax=.2, color='grey', ls='--', lw=1.)
            if i==0 and j==0:
                ax2.bar(x=.1, width=.04, height=g_diffs[dataset][method], color='orange', label='Global SP change')
            else:
                ax2.bar(x=.1, width=.04, height=g_diffs[dataset][method], color='orange')
            ax2.set_xticks([])
            ax2.tick_params(axis='both', labelsize=5)
            ax2.yaxis.tick_right()

            if j<(len(methods)-1):
                ax2.set_yticks([])

    pad = 5
    for a, col in zip(ax[0], methods):
        a.annotate(col, xy=(.5, 1), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size=6, weight='bold', ha='left', va='baseline', rotation=45)
    for a, row in zip(ax[:,0], datasets):
        a.annotate(row, xy=(0, .5), xytext=(0, 0), xycoords=a.yaxis.label, textcoords='offset points', size=6, weight='bold', ha='right', va='center')

    fig.legend(bbox_to_anchor=(.9, 1.), prop={'size': 5})

    fig.savefig('./report/05_denoising_sp.png', bbox_inches='tight', dpi=300, transparent=True)
    fig.savefig('./report/05_denoising_sp.svg', bbox_inches='tight', transparent=True)

def plot_denoising_rnx_change(
    datasets,
    methods,
    rnx,
    nruns=5
):
    fig, ax = plt.subplots(ncols=len(methods), nrows=len(datasets), sharey=True, sharex=False, figsize=(.8*len(methods), .7*len(datasets)), dpi=150)
    fig.subplots_adjust(hspace=.5)
    mpl.rcParams['axes.linewidth'] = 0.2

    col0 = plt.cm.Blues(np.linspace(.5, 1., nruns))
    col1 = plt.cm.Reds(np.linspace(.5, 1., nruns))

    for i, dataset in enumerate(datasets):
        for j, method in enumerate(methods):
            curves0 = np.array(rnx['uFalse'][dataset][method])
            avg0 = np.mean(curves0, axis=0)
            curves1 = np.array(rnx['uTrue'][dataset][method])
            avg1 = np.mean(curves1, axis=0)

            for k, curve in enumerate(curves0):
                if i==0 and j==0 and k==0:
                    ax[i][j].plot(range(len(curve)), curve, lw=.6, color=col0[k], label='Trained on original inputs')
                else:
                    ax[i][j].plot(range(len(curve)), curve, lw=.6, color=col0[k])
            for k, curve in enumerate(curves1):
                if i==0 and j==0 and k==0:
                    ax[i][j].plot(range(len(curve)), curve, lw=.6, color=col1[k], label='Trained on denoised inputs')
                else:
                    ax[i][j].plot(range(len(curve)), curve, lw=.6, color=col1[k])
            ax[i][j].axhline(y=0, xmin=0, xmax=len(curve)+2, lw=.2, ls='--', color='gray')
            ax[i][j].xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: ('{:,.0f}'.format(x/1000) + 'K') if x>0 else '0'))
            ax[i][j].tick_params(axis='both', labelsize=5)
            ax[i][j].tick_params(axis='both', labelsize=5)
        
    pad = 5
    for a, col in zip(ax[0], methods):
        a.annotate(col, xy=(.4, 1), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size=6, weight='bold', ha='left', rotation=45, va='baseline')
    for a, row in zip(ax[:,0], datasets):
        a.annotate(row, xy=(0, .5), xytext=(0, 0), xycoords=a.yaxis.label, textcoords='offset points', size=6, weight='bold', ha='right', va='center')

    l = fig.legend(bbox_to_anchor=(.8, 1.05), prop={'size': 5})
    for line in l.get_lines():
        line.set_linewidth(2.)

    fig.savefig('./report/05_denoising_rnx_curves.png', bbox_inches='tight', dpi=300, transparent=True)
    fig.savefig('./report/05_denoising_rnx_curves.svg', bbox_inches='tight', transparent=True)