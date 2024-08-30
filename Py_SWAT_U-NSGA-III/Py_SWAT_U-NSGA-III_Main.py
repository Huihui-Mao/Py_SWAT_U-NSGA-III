# -*- coding: utf-8 -*-
"""
Created on 2023.10.28
@author: Mao Huihui
"""
import os
import re
import sys
import time
import math
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from itertools import chain
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from SALib.sample.sobol import sample
from SALib.sample import latin
from SALib.analyze import sobol
from SALib.analyze import pawn
from pymoo.optimize import minimize
from pymoo.decomposition.asf import ASF
from pymoo.operators.mutation.pm import PM
from pymoo.indicators.hv import Hypervolume
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.termination import get_termination
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.running_metric import RunningMetricAnimation
from SWAT_NSGA import SWAT_Execution, SWAT_Optimization_Problems


# Main Function
if __name__ == '__main__':
    stime = time.time()
    print('=' * 90)
    print('Py_SWAT_U-NSGA-III'.center(90, '='))
    print('=' * 90)
    print('\n')
    np.set_printoptions(suppress=True) # Cancel scientific notation

    # Project Path
    pop_size = 100
    SWAT_Model_Project_Path = os.getcwd()

    # SWAT Model Execution
    SWAT_Execution_Run = SWAT_Execution.SWAT_Run(SWAT_Model_Project_Path, pop_size)
    swat_parameter = SWAT_Execution_Run.swat_parameter
    para_names = [para_idx[0] for para_idx in swat_parameter]
    # print('para_names:', len(para_names), para_names)
    para_bounds = [[para_idx[1][0], para_idx[1][1]] for para_idx in swat_parameter]
    # print('para_bounds:', len(para_bounds))
    cal_scheme        = SWAT_Execution_Run.cal_scheme
    para_fix_mode     = SWAT_Execution_Run.para_fix_mode
    cal_val_state     = SWAT_Execution_Run.cal_val_state
    swat_model        = SWAT_Execution_Run.swat_model
    swat_parallel     = SWAT_Execution_Run.swat_parallel
    swat_TxtInOut     = SWAT_Execution_Run.swat_TxtInOut
    swat_nsga_out     = SWAT_Execution_Run.swat_nsga_out
    obj_func_num      = SWAT_Execution_Run.obj_func_num
    cal_vars_list     = SWAT_Execution_Run.cal_vars_list
    objective_funs    = SWAT_Execution_Run.objective_funs
    hydro_stas        = SWAT_Execution_Run.hydro_stas
    para_obj_val_dict = SWAT_Execution_Run.para_obj_val_dict

    # Sensitivity Analysis
    SA_flag = SWAT_Execution_Run.SA_flag
    if SA_flag:
        print('Sensitivity Analysis'.center(90, '*'))
        SA_start_time = time.time()
        TVSA           = SWAT_Execution_Run.TVSA
        half_win       = SWAT_Execution_Run.half_win
        cal_period     = SWAT_Execution_Run.cal_period
        SA_method      = SWAT_Execution_Run.SA_method
        cpu_worker_num = SWAT_Execution_Run.cpu_worker_num
        # The number of parameters
        D = len(swat_parameter)
        print('D:', D)

        # 1) Defining the Model Inputs
        problem = {
            'num_vars': D,
            'names': para_names,
            'bounds': para_bounds
        }

        # The number of samples to generate
        N = 30 # Sobol: Ideally a power of 2, requires a large number of sampling, N >= 1000

        # 2) Generate Samples
        sobol_param_values, latin_param_values, param_values = None, None, None
        if SA_method == 'Sobol':
            print('Sobol’ Sensitivity Analysis:')
            # Sobol: Generates model inputs using Saltelli’s extension of the Sobol’ sequence
            sobol_param_values = sample(problem, N, calc_second_order=False)
            print('sobol_param_values:', sobol_param_values.shape)
            print(sobol_param_values)
            param_values = sobol_param_values
            print('\n')
            # Model runs
            model_runs = N * (D + 2)
            print('model_runs:', model_runs)
        elif SA_method == 'PAWN':
            print('PAWN Sensitivity Analysis:')
            # PAWN: Latin hypercube sampling (LHS)
            latin_param_values = latin.sample(problem, N)
            print('latin_param_values:', latin_param_values.shape)
            print(latin_param_values)
            param_values = latin_param_values
            print('\n')
            # Model runs
            print('model_runs:', N)

        # 3) Run Model
        mod_run_obj_fun_list = []
        obj_fun_list_SF_NingDu, obj_fun_list_SF_ShiCheng, obj_fun_list_SF_FenKeng, obj_fun_list_SF, obj_fun_list_LAI, obj_fun_list_ET = [], [], [], [], [], []
        for mod_run_idx in range(0, param_values.shape[0], pop_size):
            print('mod_run_idx:', mod_run_idx)
            param_values_subset = param_values[mod_run_idx:mod_run_idx + pop_size]
            print(param_values_subset.shape, '\n', param_values_subset)
            obj_fun_list = SWAT_Execution_Run.SWAT_model_execution(param_values_subset)
            # print('obj_fun_list:', len(obj_fun_list))
            if cal_vars_list == ['Streamflow']:
                if TVSA:
                    if obj_func_num == 1:
                        mod_run_obj_fun_list.extend(obj_fun_list)
                    elif obj_func_num == 3:
                        obj_fun_list_SF_NingDu.extend(obj_fun_list[0])
                        obj_fun_list_SF_ShiCheng.extend(obj_fun_list[1])
                        obj_fun_list_SF_FenKeng.extend(obj_fun_list[2])
                else:
                    if obj_func_num == 1:
                        obj_fun_SA = obj_fun_list
                        mod_run_obj_fun_list.extend(obj_fun_SA)
                    else:
                        obj_fun_SA = np.array(obj_fun_list).mean(axis=0)
                        mod_run_obj_fun_list.extend(obj_fun_SA)
            elif cal_vars_list == ['LAI'] or cal_vars_list == ['BIOM']:
                mod_run_obj_fun_list.extend(obj_fun_list)
            elif cal_vars_list == ['Streamflow', 'ET']:
                if TVSA:
                    if len(obj_fun_list[0]) == 1:
                        obj_fun_list_SF.extend(obj_fun_list[0])
                        obj_fun_list_ET.extend(obj_fun_list[1])
                    else:
                        obj_fun_list_SF_NingDu.extend(obj_fun_list[0][0])
                        obj_fun_list_SF_ShiCheng.extend(obj_fun_list[0][1])
                        obj_fun_list_SF_FenKeng.extend(obj_fun_list[0][2])
                        obj_fun_list_ET.extend(obj_fun_list[1])
                else:
                    if len(obj_fun_list[0]) == 1:
                        obj_fun_rch_SA = obj_fun_list[0]
                        obj_fun_ET_SA  = obj_fun_list[1]
                        mod_run_obj_fun_list.extend(np.array([obj_fun_rch_SA, obj_fun_ET_SA]).mean(axis=0))
                    else:
                        obj_fun_rch_SA = np.array(obj_fun_list[0]).mean(axis=0)
                        obj_fun_ET_SA  = obj_fun_list[1]
                        mod_run_obj_fun_list.extend(np.array([obj_fun_rch_SA, obj_fun_ET_SA]).mean(axis=0))
            elif cal_vars_list == ['Streamflow', 'RZSW']:
                if len(obj_fun_list[0]) == 1:
                    obj_fun_rch_SA = obj_fun_list[0]
                    obj_fun_RZSW_SA = obj_fun_list[1]
                    mod_run_obj_fun_list.extend(np.array([obj_fun_rch_SA, obj_fun_RZSW_SA]).mean(axis=0))
                else:
                    obj_fun_rch_SA = np.array(obj_fun_list[0]).mean(axis=0)
                    obj_fun_RZSW_SA = obj_fun_list[1]
                    mod_run_obj_fun_list.extend(np.array([obj_fun_rch_SA, obj_fun_RZSW_SA]).mean(axis=0))
            elif cal_vars_list == ['Streamflow', 'LAI', 'ET']:
                if TVSA:
                    if len(obj_fun_list[0]) == 1:
                        obj_fun_list_SF.extend(obj_fun_list[0])
                        obj_fun_list_LAI.extend(obj_fun_list[1])
                        obj_fun_list_ET.extend(obj_fun_list[2])
                    else:
                        obj_fun_list_SF_NingDu.extend(obj_fun_list[0][0])
                        obj_fun_list_SF_ShiCheng.extend(obj_fun_list[0][1])
                        obj_fun_list_SF_FenKeng.extend(obj_fun_list[0][2])
                        obj_fun_list_LAI.extend(obj_fun_list[1])
                        obj_fun_list_ET.extend(obj_fun_list[2])
                else:
                    if len(obj_fun_list[0]) == 1:
                        obj_fun_rch_SA = obj_fun_list[0]
                        obj_fun_LAI_SA = obj_fun_list[1]
                        obj_fun_ET_SA  = obj_fun_list[2]
                        mod_run_obj_fun_list.extend(np.array([obj_fun_rch_SA, obj_fun_LAI_SA, obj_fun_ET_SA]).mean(axis=0))
                    else:
                        obj_fun_rch_SA = np.array(obj_fun_list[0]).mean(axis=0)
                        obj_fun_LAI_SA = obj_fun_list[1]
                        obj_fun_ET_SA  = obj_fun_list[2]
                        mod_run_obj_fun_list.extend(np.array([obj_fun_rch_SA, obj_fun_LAI_SA, obj_fun_ET_SA]).mean(axis=0))
            elif cal_vars_list == ['Streamflow', 'ET', 'RZSW']:
                if len(obj_fun_list[0]) == 1:
                    obj_fun_rch_SA  = obj_fun_list[0]
                    obj_fun_ET_SA   = obj_fun_list[1]
                    obj_fun_RZSW_SA = obj_fun_list[2]
                    mod_run_obj_fun_list.extend(np.array([obj_fun_rch_SA, obj_fun_ET_SA, obj_fun_RZSW_SA]).mean(axis=0))
                else:
                    obj_fun_rch_SA  = np.array(obj_fun_list[0]).mean(axis=0)
                    obj_fun_ET_SA   = obj_fun_list[1]
                    obj_fun_RZSW_SA = obj_fun_list[2]
                    mod_run_obj_fun_list.extend(np.array([obj_fun_rch_SA, obj_fun_ET_SA, obj_fun_RZSW_SA]).mean(axis=0))
        print('mod_run_obj_fun_list:', len(mod_run_obj_fun_list))
        if TVSA:
            if cal_vars_list == ['Streamflow']:
                obs_sf_data = SWAT_Execution_Run.obs_sf_data
                # 创建日期序列
                date_range = pd.date_range(start=f'{cal_period[0]}-01-01', end=f'{cal_period[1]}-12-31', freq='D').strftime('%Y-%m-%d')
                winsize = 2 * half_win + 1
                TVSA_date = [date_range[win_idx + half_win] if (win_idx + half_win) < len(date_range)
                             else date_range[int(win_idx + (len(date_range) - win_idx) / 2)] for win_idx in range(0, len(date_range), winsize)]
                print('TVSA_date:', len(TVSA_date))

                Y_NingDu_SF   = np.array(obj_fun_list_SF_NingDu)
                Y_ShiCheng_SF = np.array(obj_fun_list_SF_ShiCheng)
                Y_FenKeng_SF  = np.array(obj_fun_list_SF_FenKeng)
                print('Y_NingDu_SF:', Y_NingDu_SF.shape)
                print('Y_ShiCheng_SF:', Y_ShiCheng_SF.shape)
                print('Y_FenKeng_SF:', Y_FenKeng_SF.shape)

                # Sobol’ Sensitivity Analysis
                if SA_method == 'Sobol':
                    # Time-varying sensitivity analysis
                    Sobol_STi_NingDu_SF_df, Sobol_STi_ShiCheng_SF_df, Sobol_STi_FenKeng_SF_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                    for win_idx in tqdm(range(Y_NingDu_SF.shape[1]), total=Y_NingDu_SF.shape[1]):
                        # 4) Perform Analysis
                        Sobol_Si_NingDu_SF_win   = sobol.analyze(problem, Y_NingDu_SF[:, win_idx], calc_second_order=False, print_to_console=False,
                                                                 parallel=True, n_processors=cpu_worker_num)
                        Sobol_Si_ShiCheng_SF_win = sobol.analyze(problem, Y_ShiCheng_SF[:, win_idx], calc_second_order=False, print_to_console=False,
                                                                 parallel=True, n_processors=cpu_worker_num)
                        Sobol_Si_FenKeng_SF_win  = sobol.analyze(problem, Y_FenKeng_SF[:, win_idx], calc_second_order=False, print_to_console=False,
                                                                 parallel=True, n_processors=cpu_worker_num)
                        Sobol_STi_NingDu_SF_df_win   = Sobol_Si_NingDu_SF_win.to_df()[0].drop(columns={'ST_conf'}, axis=1).rename(columns={'ST': f'ST{win_idx + 1}'})
                        Sobol_STi_ShiCheng_SF_df_win = Sobol_Si_ShiCheng_SF_win.to_df()[0].drop(columns={'ST_conf'}, axis=1).rename(columns={'ST': f'ST{win_idx + 1}'})
                        Sobol_STi_FenKeng_SF_df_win  = Sobol_Si_FenKeng_SF_win.to_df()[0].drop(columns={'ST_conf'}, axis=1).rename(columns={'ST': f'ST{win_idx + 1}'})
                        Sobol_STi_NingDu_SF_df   = pd.concat([Sobol_STi_NingDu_SF_df, Sobol_STi_NingDu_SF_df_win], axis=1, ignore_index=False)
                        Sobol_STi_ShiCheng_SF_df = pd.concat([Sobol_STi_ShiCheng_SF_df, Sobol_STi_ShiCheng_SF_df_win], axis=1, ignore_index=False)
                        Sobol_STi_FenKeng_SF_df  = pd.concat([Sobol_STi_FenKeng_SF_df, Sobol_STi_FenKeng_SF_df_win], axis=1, ignore_index=False)
                    # Mean Sobol Sensitivity Index
                    Sobol_STi_NingDu_SF_df['ST_mean']   = Sobol_STi_NingDu_SF_df.mean(axis=1)
                    Sobol_STi_ShiCheng_SF_df['ST_mean'] = Sobol_STi_ShiCheng_SF_df.mean(axis=1)
                    Sobol_STi_FenKeng_SF_df['ST_mean']  = Sobol_STi_FenKeng_SF_df.mean(axis=1)

                    # 修改index列，去掉大括号
                    NingDu_SF_idx_list, ShiCheng_SF_idx_list, FenKeng_SF_idx_list = [], [], []
                    for NingDu_SF_idx, ShiCheng_SF_idx, FenKeng_SF_idx in zip(Sobol_STi_NingDu_SF_df.index, Sobol_STi_ShiCheng_SF_df.index,
                                                                              Sobol_STi_FenKeng_SF_df.index):
                        NingDu_SF_Si_idx, ShiCheng_SF_Si_idx, FenKeng_SF_Si_idx = (NingDu_SF_idx.split('{')[0], ShiCheng_SF_idx.split('{')[0],
                                                                                   FenKeng_SF_idx.split('{')[0])
                        NingDu_SF_idx_list.append(NingDu_SF_Si_idx)
                        ShiCheng_SF_idx_list.append(ShiCheng_SF_Si_idx)
                        FenKeng_SF_idx_list.append(FenKeng_SF_Si_idx)
                    Sobol_STi_NingDu_SF_df.index   = NingDu_SF_idx_list
                    Sobol_STi_ShiCheng_SF_df.index = ShiCheng_SF_idx_list
                    Sobol_STi_FenKeng_SF_df.index  = FenKeng_SF_idx_list

                    # index列添加列名
                    Sobol_STi_NingDu_SF_df.index.name   = 'Para'
                    Sobol_STi_ShiCheng_SF_df.index.name = 'Para'
                    Sobol_STi_FenKeng_SF_df.index.name  = 'Para'

                    # 使用groupby方法按Index列分组，并计算平均值，最后按列SA_mean值排序
                    Sobol_STi_NingDu_SF_df_group   = Sobol_STi_NingDu_SF_df.groupby('Para').mean().sort_values(by='ST_mean', ascending=False)
                    Sobol_STi_ShiCheng_SF_df_group = Sobol_STi_ShiCheng_SF_df.groupby('Para').mean().sort_values(by='ST_mean', ascending=False)
                    Sobol_STi_FenKeng_SF_df_group  = Sobol_STi_FenKeng_SF_df.groupby('Para').mean().sort_values(by='ST_mean', ascending=False)
                    print('Sobol_STi_NingDu_SF_df_group:', Sobol_STi_NingDu_SF_df_group.shape, '\n', Sobol_STi_NingDu_SF_df_group)
                    print('Sobol_STi_ShiCheng_SF_df_group:', Sobol_STi_ShiCheng_SF_df_group.shape, '\n', Sobol_STi_ShiCheng_SF_df_group)
                    print('Sobol_STi_FenKeng_SF_df_group:', Sobol_STi_FenKeng_SF_df_group.shape, '\n', Sobol_STi_FenKeng_SF_df_group)

                    # 5) Result Output
                    # 将数据框输出为Excel文件
                    Sobol_STi_NingDu_SF_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\Sobol_Sensitivity_TVSA_NingDu.xlsx', index=True)
                    Sobol_STi_ShiCheng_SF_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\Sobol_Sensitivity_TVSA_ShiCheng.xlsx', index=True)
                    Sobol_STi_FenKeng_SF_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\Sobol_Sensitivity_TVSA_FenKeng.xlsx', index=True)

                    # 6) TVSA Plotting
                    Sobol_STi_max = math.ceil(max(Sobol_STi_NingDu_SF_df_group.max(axis=0).max(), Sobol_STi_ShiCheng_SF_df_group.max(axis=0).max(),
                                                  Sobol_STi_FenKeng_SF_df_group.max(axis=0).max()) * 10) / 10.0
                    print('Sobol_STi_max: ', Sobol_STi_max)
                    for hydro_sta_idx in zip(['NingDu', 'ShiCheng', 'FenKeng'],
                                             [Sobol_STi_NingDu_SF_df_group, Sobol_STi_ShiCheng_SF_df_group, Sobol_STi_FenKeng_SF_df_group]):
                        print(hydro_sta_idx[0])
                        fig = plt.figure(figsize=(16, 10), dpi=500)
                        # 创建一个4行4列的GridSpec对象
                        grid = gridspec.GridSpec(nrows=13, ncols=18, wspace=0.2, hspace=2.8)
                        ax1 = plt.subplot(grid[2:-1, :-2])
                        ax2 = plt.subplot(grid[:2, :-2])
                        ax3 = plt.subplot(grid[2:-1, -2:])
                        ax4 = plt.subplot(grid[-1, :-2])
                        # ax1
                        ticks_interval = 22
                        sns.heatmap(hydro_sta_idx[1].iloc[:, :-1], vmin=0, vmax=Sobol_STi_max, cmap='gist_earth_r', cbar=True, cbar_ax=ax4,
                                    cbar_kws={'orientation': 'horizontal'}, xticklabels=ticks_interval, yticklabels='auto', ax=ax1)
                        xticks = ax1.get_xticks()
                        yticks = ax1.get_yticks()
                        ax1.set_title(label=f'Sobol Sensitivity Index (ST$_i$)-{hydro_sta_idx[0]}', pad=7, fontsize=12)
                        ax1.set_xticks(xticks, labels=TVSA_date[::ticks_interval], rotation=18, ha='right')
                        ax1.set_ylabel('')
                        # ax2
                        ax2.set_xticks([])
                        ax2.set_xticklabels([])
                        ax2.plot(date_range, obs_sf_data[hydro_stas[hydro_sta_idx[0]]], color='grey', linewidth=1.0)
                        ax2.set_xlim([-1, len(date_range) + 1])
                        ax2.set_ylabel('Streamflow\n(m$^3$/s)')
                        # ax3
                        ax3.sharey(ax1)  # 共享y轴
                        ax3.tick_params(axis='y', labelleft=False)  # 不显示y轴刻度标签
                        # 绘制横向柱状图
                        ax3.barh(yticks, hydro_sta_idx[1]['ST_mean'], color='grey')
                        ax3.set_xlabel(xlabel=r'Mean ST$_i$', fontsize=12)
                        plt.savefig(f'{SWAT_Execution_Run.swat_nsga_out}\\Sobol_Sensitivity_Index_TVSA_{hydro_sta_idx[0]}.jpg', bbox_inches='tight')
                        plt.show()
                    # Exit the current program after completing sensitivity analysis
                    print('Sobol’ Sensitivity Analysis Finished!')
                    sys.exit()
                # PAWN Sensitivity Analysis
                elif SA_method == 'PAWN':
                    # Time-varying sensitivity analysis
                    PAWN_Si_NingDu_SF_df, PAWN_Si_ShiCheng_SF_df, PAWN_Si_FenKeng_SF_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                    for win_idx in tqdm(range(Y_NingDu_SF.shape[1]), total=Y_NingDu_SF.shape[1]):
                        # 4) Perform Analysis
                        PAWN_Si_NingDu_SF_win   = pawn.analyze(problem, latin_param_values, Y_NingDu_SF[:, win_idx], S=20, print_to_console=False)
                        PAWN_Si_ShiCheng_SF_win = pawn.analyze(problem, latin_param_values, Y_ShiCheng_SF[:, win_idx], S=20, print_to_console=False)
                        PAWN_Si_FenKeng_SF_win  = pawn.analyze(problem, latin_param_values, Y_FenKeng_SF[:, win_idx], S=20, print_to_console=False)
                        PAWN_Si_NingDu_SF_df_win   = (PAWN_Si_NingDu_SF_win.to_df().drop(columns={'minimum', 'mean', 'maximum', 'CV'}, axis=1).
                                                      rename(columns={'median': f'median{win_idx+1}'}))
                        PAWN_Si_ShiCheng_SF_df_win = (PAWN_Si_ShiCheng_SF_win.to_df().drop(columns={'minimum', 'mean', 'maximum', 'CV'}, axis=1).
                                                      rename(columns={'median': f'median{win_idx+1}'}))
                        PAWN_Si_FenKeng_SF_df_win  = (PAWN_Si_FenKeng_SF_win.to_df().drop(columns={'minimum', 'mean', 'maximum', 'CV'}, axis=1).
                                                      rename(columns={'median': f'median{win_idx+1}'}))
                        PAWN_Si_NingDu_SF_df   = pd.concat([PAWN_Si_NingDu_SF_df, PAWN_Si_NingDu_SF_df_win], axis=1, ignore_index=False)
                        PAWN_Si_ShiCheng_SF_df = pd.concat([PAWN_Si_ShiCheng_SF_df, PAWN_Si_ShiCheng_SF_df_win], axis=1, ignore_index=False)
                        PAWN_Si_FenKeng_SF_df  = pd.concat([PAWN_Si_FenKeng_SF_df, PAWN_Si_FenKeng_SF_df_win], axis=1, ignore_index=False)
                    # Mean PAWN Sensitivity Index
                    PAWN_Si_NingDu_SF_df['median_mean']   = PAWN_Si_NingDu_SF_df.mean(axis=1)
                    PAWN_Si_ShiCheng_SF_df['median_mean'] = PAWN_Si_ShiCheng_SF_df.mean(axis=1)
                    PAWN_Si_FenKeng_SF_df['median_mean']  = PAWN_Si_FenKeng_SF_df.mean(axis=1)

                    # 修改index列，去掉大括号
                    NingDu_SF_idx_list, ShiCheng_SF_idx_list, FenKeng_SF_idx_list = [], [], []
                    for NingDu_SF_idx, ShiCheng_SF_idx, FenKeng_SF_idx in zip(PAWN_Si_NingDu_SF_df.index, PAWN_Si_ShiCheng_SF_df.index,
                                                                              PAWN_Si_FenKeng_SF_df.index):
                        NingDu_SF_Si_idx, ShiCheng_SF_Si_idx, FenKeng_SF_Si_idx = (NingDu_SF_idx.split('{')[0], ShiCheng_SF_idx.split('{')[0],
                                                                                   FenKeng_SF_idx.split('{')[0])
                        NingDu_SF_idx_list.append(NingDu_SF_Si_idx)
                        ShiCheng_SF_idx_list.append(ShiCheng_SF_Si_idx)
                        FenKeng_SF_idx_list.append(FenKeng_SF_Si_idx)
                    PAWN_Si_NingDu_SF_df.index   = NingDu_SF_idx_list
                    PAWN_Si_ShiCheng_SF_df.index = ShiCheng_SF_idx_list
                    PAWN_Si_FenKeng_SF_df.index  = FenKeng_SF_idx_list

                    # index列添加列名
                    PAWN_Si_NingDu_SF_df.index.name   = 'Para'
                    PAWN_Si_ShiCheng_SF_df.index.name = 'Para'
                    PAWN_Si_FenKeng_SF_df.index.name  = 'Para'

                    # 使用groupby方法按Index列分组，并计算平均值，最后按列SA_mean值排序
                    PAWN_Si_NingDu_SF_df_group   = PAWN_Si_NingDu_SF_df.groupby('Para').mean().sort_values(by='median_mean', ascending=False)
                    PAWN_Si_ShiCheng_SF_df_group = PAWN_Si_ShiCheng_SF_df.groupby('Para').mean().sort_values(by='median_mean', ascending=False)
                    PAWN_Si_FenKeng_SF_df_group  = PAWN_Si_FenKeng_SF_df.groupby('Para').mean().sort_values(by='median_mean', ascending=False)
                    print('PAWN_Si_NingDu_SF_df_group:', PAWN_Si_NingDu_SF_df_group.shape, '\n', PAWN_Si_NingDu_SF_df_group)
                    print('PAWN_Si_ShiCheng_SF_df_group:', PAWN_Si_ShiCheng_SF_df_group.shape, '\n', PAWN_Si_ShiCheng_SF_df_group)
                    print('PAWN_Si_FenKeng_SF_df_group:', PAWN_Si_FenKeng_SF_df_group.shape, '\n', PAWN_Si_FenKeng_SF_df_group)

                    # 5) Result Output
                    # 将数据框输出为Excel文件
                    PAWN_Si_NingDu_SF_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\PAWN_Sensitivity_TVSA_NingDu.xlsx', index=True)
                    PAWN_Si_ShiCheng_SF_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\PAWN_Sensitivity_TVSA_ShiCheng.xlsx', index=True)
                    PAWN_Si_FenKeng_SF_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\PAWN_Sensitivity_TVSA_FenKeng.xlsx', index=True)

                    # 6) TVSA Plotting
                    PAWN_Si_max = math.ceil(max(PAWN_Si_NingDu_SF_df_group.max(axis=0).max(), PAWN_Si_ShiCheng_SF_df_group.max(axis=0).max(),
                                                PAWN_Si_FenKeng_SF_df_group.max(axis=0).max()) * 10) / 10.0
                    print('PAWN_Si_max: ', PAWN_Si_max)
                    for hydro_sta_idx in zip(['NingDu', 'ShiCheng', 'FenKeng'],
                                             [PAWN_Si_NingDu_SF_df_group, PAWN_Si_ShiCheng_SF_df_group, PAWN_Si_FenKeng_SF_df_group]):
                        print(hydro_sta_idx[0])
                        fig = plt.figure(figsize=(16, 10), dpi=500)
                        # 创建一个4行4列的GridSpec对象
                        grid = gridspec.GridSpec(nrows=13, ncols=18, wspace=0.2, hspace=2.8)
                        ax1 = plt.subplot(grid[2:-1,:-2])
                        ax2 = plt.subplot(grid[:2,:-2])
                        ax3 = plt.subplot(grid[2:-1,-2:])
                        ax4 = plt.subplot(grid[-1,:-2])
                        # ax1
                        ticks_interval = 22
                        sns.heatmap(hydro_sta_idx[1].iloc[:, :-1], vmin=0, vmax=PAWN_Si_max, cmap='gist_earth_r', cbar=True, cbar_ax=ax4,
                                    cbar_kws={'orientation':'horizontal'}, xticklabels=ticks_interval, yticklabels='auto', ax=ax1)
                        xticks = ax1.get_xticks()
                        yticks = ax1.get_yticks()
                        ax1.set_title(label=f'PAWN Sensitivity Index (S$_i$)-{hydro_sta_idx[0]}', pad=7, fontsize=12)
                        ax1.set_xticks(xticks, labels=TVSA_date[::ticks_interval], rotation=18, ha='right')
                        ax1.set_ylabel('')
                        # ax2
                        ax2.set_xticks([])
                        ax2.set_xticklabels([])
                        ax2.plot(date_range, obs_sf_data[hydro_stas[hydro_sta_idx[0]]], color='grey', linewidth=1.0)
                        ax2.set_xlim([-1, len(date_range) + 1])
                        ax2.set_ylabel('Streamflow\n(m$^3$/s)')
                        # ax3
                        ax3.sharey(ax1)  # 共享y轴
                        ax3.tick_params(axis='y', labelleft=False)  # 不显示y轴刻度标签
                        # 绘制横向柱状图
                        ax3.barh(yticks, hydro_sta_idx[1]['median_mean'], color='grey')
                        ax3.set_xlabel(xlabel=r'Mean S$_i$', fontsize=12)
                        plt.savefig(f'{SWAT_Execution_Run.swat_nsga_out}\\PAWN_Sensitivity_Index_TVSA_{hydro_sta_idx[0]}.jpg', bbox_inches='tight')
                        plt.show()
                    # Exit the current program after completing sensitivity analysis
                    print('PAWN Sensitivity Analysis Finished!')
                    sys.exit()
            elif cal_vars_list == ['LAI']:
                obs_lai_w_data = SWAT_Execution_Run.LAI_obs_data_dict_area_w
                # 创建日期序列
                date_range = pd.date_range(start=f'{cal_period[0]}-01-01', end=f'{cal_period[1]}-12-31', freq='D').strftime('%Y-%m-%d')
                winsize = 2 * half_win + 1
                TVSA_date = [date_range[win_idx + half_win] if (win_idx + half_win) < len(date_range)
                             else date_range[int(win_idx + (len(date_range) - win_idx) / 2)]
                             for win_idx in range(0, len(date_range), winsize)]
                print('TVSA_date:', len(TVSA_date))

                Y_LAI = np.array(mod_run_obj_fun_list)
                print('Y_LAI:', Y_LAI.shape)

                # Sobol’ Sensitivity Analysis
                if SA_method == 'Sobol':
                    # Time-varying sensitivity analysis
                    Sobol_STi_LAI_df = pd.DataFrame()
                    for win_idx in tqdm(range(Y_LAI.shape[1]), total=Y_LAI.shape[1]):
                        # 4) Perform Analysis
                        Sobol_Si_LAI_win = sobol.analyze(problem, Y_LAI[:, win_idx], calc_second_order=False, print_to_console=False, parallel=True,
                                                         n_processors=cpu_worker_num)
                        Sobol_STi_LAI_df_win = Sobol_Si_LAI_win.to_df()[0].drop(columns={'ST_conf'}, axis=1).rename(columns={'ST': f'ST{win_idx + 1}'})
                        Sobol_STi_LAI_df = pd.concat([Sobol_STi_LAI_df, Sobol_STi_LAI_df_win], axis=1, ignore_index=False)

                    # Mean Sobol Sensitivity Index
                    Sobol_STi_LAI_df['ST_mean'] = Sobol_STi_LAI_df.mean(axis=1)

                    # 修改index列，去掉大括号
                    LAI_idx_list = []
                    for LAI_idx in Sobol_STi_LAI_df.index:
                        LAI_Si_idx = LAI_idx.split('{')[0]
                        LAI_idx_list.append(LAI_Si_idx)
                    Sobol_STi_LAI_df.index = LAI_idx_list

                    # index列添加列名
                    Sobol_STi_LAI_df.index.name = 'Para'

                    # 使用groupby方法按Index列分组，并计算平均值，最后按列SA_mean值排序
                    Sobol_STi_LAI_df_group = Sobol_STi_LAI_df.groupby('Para').mean().sort_values(by='ST_mean', ascending=False)
                    print('Sobol_STi_LAI_df_group:', Sobol_STi_LAI_df_group.shape, '\n', Sobol_STi_LAI_df_group)

                    # 5) Result Output
                    Sobol_STi_LAI_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'Sobol_Sensitivity_TVSA_LAI.xlsx', index=True)

                    # 6) TVSA Plotting
                    Sobol_STi_max = math.ceil(Sobol_STi_LAI_df_group.max(axis=0).max() * 10) / 10.0
                    print('Sobol_STi_max: ', Sobol_STi_max)
                    #
                    fig = plt.figure(figsize=(16, 10), dpi=500)
                    # 创建一个4行4列的GridSpec对象
                    grid = gridspec.GridSpec(nrows=13, ncols=18, wspace=0.2, hspace=2.8)
                    ax1 = plt.subplot(grid[2:-1, :-2])
                    ax2 = plt.subplot(grid[:2, :-2])
                    ax3 = plt.subplot(grid[2:-1, -2:])
                    ax4 = plt.subplot(grid[-1, :-2])
                    # ax1
                    ticks_interval = 22
                    sns.heatmap(Sobol_STi_LAI_df_group.iloc[:, :-1], vmin=0, vmax=Sobol_STi_max, cmap='gist_earth_r', cbar=True, cbar_ax=ax4,
                                cbar_kws={'orientation': 'horizontal'}, xticklabels=ticks_interval, yticklabels='auto', ax=ax1)
                    xticks = ax1.get_xticks()
                    yticks = ax1.get_yticks()
                    ax1.set_title(label=f'Sobol Sensitivity Index (S$_i$)', pad=7, fontsize=12)
                    ax1.set_xticks(xticks, labels=TVSA_date[::ticks_interval], rotation=18, ha='right')
                    ax1.set_ylabel('')
                    # ax2
                    ax2.set_xticks([])
                    ax2.set_xticklabels([])
                    ax2.plot(date_range, obs_lai_w_data, color='grey', linewidth=1.0)
                    ax2.set_xlim([-1, len(date_range) + 1])
                    ax2.set_ylabel('LAI\n(m$^2$/m$^2$)')
                    # ax3
                    ax3.sharey(ax1)  # 共享y轴
                    ax3.tick_params(axis='y', labelleft=False)  # 不显示y轴刻度标签
                    # 绘制横向柱状图
                    ax3.barh(yticks, Sobol_STi_LAI_df_group['ST_mean'], color='grey')
                    ax3.set_xlabel(xlabel=r'Mean ST$_i$', fontsize=12)
                    plt.savefig(f'{SWAT_Execution_Run.swat_nsga_out}\\Sobol_Sensitivity_Index_TVSA_LAI.jpg', bbox_inches='tight')
                    plt.show()
                    # Exit the current program after completing sensitivity analysis
                    print('Sobol’ Sensitivity Analysis Finished!')
                    sys.exit()
                # PAWN Sensitivity Analysis
                elif SA_method == 'PAWN':
                    # Time-varying sensitivity analysis
                    PAWN_Si_LAI_df = pd.DataFrame()
                    for win_idx in tqdm(range(Y_LAI.shape[1]), total=Y_LAI.shape[1]):
                        # 4) Perform Analysis
                        PAWN_Si_LAI_win = pawn.analyze(problem, latin_param_values, Y_LAI[:, win_idx], S=20, print_to_console=False)
                        PAWN_Si_LAI_df_win = (PAWN_Si_LAI_win.to_df().drop(columns={'minimum', 'mean', 'maximum', 'CV'}, axis=1).
                                              rename(columns={'median': f'median{win_idx + 1}'}))
                        PAWN_Si_LAI_df = pd.concat([PAWN_Si_LAI_df, PAWN_Si_LAI_df_win], axis=1, ignore_index=False)

                    # Mean PAWN Sensitivity Index
                    PAWN_Si_LAI_df['median_mean'] = PAWN_Si_LAI_df.mean(axis=1)

                    # 修改index列，去掉大括号
                    LAI_idx_list = []
                    for LAI_idx in PAWN_Si_LAI_df.index:
                        LAI_Si_idx = LAI_idx.split('{')[0]
                        LAI_idx_list.append(LAI_Si_idx)
                    PAWN_Si_LAI_df.index = LAI_idx_list

                    # index列添加列名
                    PAWN_Si_LAI_df.index.name = 'Para'

                    # 使用groupby方法按Index列分组，并计算平均值，最后按列SA_mean值排序
                    PAWN_Si_LAI_df_group = PAWN_Si_LAI_df.groupby('Para').mean().sort_values(by='median_mean', ascending=False)
                    print('PAWN_Si_LAI_df_group:', PAWN_Si_LAI_df_group.shape, '\n', PAWN_Si_LAI_df_group)

                    # 5) Result Output
                    PAWN_Si_LAI_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'PAWN_Sensitivity_TVSA_LAI.xlsx', index=True)

                    # 6) TVSA Plotting
                    PAWN_Si_max = math.ceil(PAWN_Si_LAI_df_group.max(axis=0).max() * 10) / 10.0
                    print('PAWN_Si_max: ', PAWN_Si_max)
                    #
                    fig = plt.figure(figsize=(16, 10), dpi=500)
                    # 创建一个4行4列的GridSpec对象
                    grid = gridspec.GridSpec(nrows=13, ncols=18, wspace=0.2, hspace=2.8)
                    ax1 = plt.subplot(grid[2:-1, :-2])
                    ax2 = plt.subplot(grid[:2, :-2])
                    ax3 = plt.subplot(grid[2:-1, -2:])
                    ax4 = plt.subplot(grid[-1, :-2])
                    # ax1
                    ticks_interval = 22
                    sns.heatmap(PAWN_Si_LAI_df_group.iloc[:, :-1], vmin=0, vmax=PAWN_Si_max, cmap='gist_earth_r', cbar=True, cbar_ax=ax4,
                                cbar_kws={'orientation': 'horizontal'}, xticklabels=ticks_interval, yticklabels='auto', ax=ax1)
                    xticks = ax1.get_xticks()
                    yticks = ax1.get_yticks()
                    ax1.set_title(label=f'PAWN Sensitivity Index (S$_i$)', pad=7, fontsize=12)
                    ax1.set_xticks(xticks, labels=TVSA_date[::ticks_interval], rotation=18, ha='right')
                    ax1.set_ylabel('')
                    # ax2
                    ax2.set_xticks([])
                    ax2.set_xticklabels([])
                    ax2.plot(date_range, obs_lai_w_data, color='grey', linewidth=1.0)
                    ax2.set_xlim([-1, len(date_range) + 1])
                    ax2.set_ylabel('LAI\n(m$^2$/m$^2$)')
                    # ax3
                    ax3.sharey(ax1)  # 共享y轴
                    ax3.tick_params(axis='y', labelleft=False)  # 不显示y轴刻度标签
                    # 绘制横向柱状图
                    ax3.barh(yticks, PAWN_Si_LAI_df_group['median_mean'], color='grey')
                    ax3.set_xlabel(xlabel=r'Mean S$_i$', fontsize=12)
                    plt.savefig(f'{SWAT_Execution_Run.swat_nsga_out}\\PAWN_Sensitivity_Index_TVSA_LAI.jpg', bbox_inches='tight')
                    plt.show()
                    # Exit the current program after completing sensitivity analysis
                    print('PAWN Sensitivity Analysis Finished!')
                    sys.exit()
            elif cal_vars_list == ['Streamflow', 'ET']:
                obs_sf_data    = SWAT_Execution_Run.obs_sf_data
                obs_et_w_data  = SWAT_Execution_Run.ET_obs_data_dict_area_w
                # 创建日期序列
                date_range = pd.date_range(start=f'{cal_period[0]}-01-01', end=f'{cal_period[1]}-12-31', freq='D').strftime('%Y-%m-%d')
                winsize = 2 * half_win + 1
                TVSA_date = [date_range[win_idx + half_win] if (win_idx + half_win) < len(date_range)
                             else date_range[int(win_idx + (len(date_range) - win_idx) / 2)]
                             for win_idx in range(0, len(date_range), winsize)]
                print('TVSA_date:', len(TVSA_date))

                Y_NingDu_SF   = np.array(obj_fun_list_SF_NingDu)
                Y_ShiCheng_SF = np.array(obj_fun_list_SF_ShiCheng)
                Y_FenKeng_SF  = np.array(obj_fun_list_SF_FenKeng)
                Y_ET          = np.array(obj_fun_list_ET)
                print('Y_NingDu_SF:', Y_NingDu_SF.shape)
                print('Y_ShiCheng_SF:', Y_ShiCheng_SF.shape)
                print('Y_FenKeng_SF:', Y_FenKeng_SF.shape)
                print('Y_ET:', Y_ET.shape)

                # Sobol’ Sensitivity Analysis
                if SA_method == 'Sobol':
                    # Time-varying sensitivity analysis
                    Sobol_STi_NingDu_SF_df, Sobol_STi_ShiCheng_SF_df, Sobol_STi_FenKeng_SF_df, Sobol_STi_ET_df \
                        = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                    for win_idx in tqdm(range(Y_NingDu_SF.shape[1]), total=Y_NingDu_SF.shape[1]):
                        # 4) Perform Analysis
                        Sobol_Si_NingDu_SF_win   = sobol.analyze(problem, Y_NingDu_SF[:, win_idx], calc_second_order=False, print_to_console=False,
                                                                 parallel=True, n_processors=cpu_worker_num)
                        Sobol_Si_ShiCheng_SF_win = sobol.analyze(problem, Y_ShiCheng_SF[:, win_idx], calc_second_order=False, print_to_console=False,
                                                                 parallel=True, n_processors=cpu_worker_num)
                        Sobol_Si_FenKeng_SF_win  = sobol.analyze(problem, Y_FenKeng_SF[:, win_idx], calc_second_order=False, print_to_console=False,
                                                                 parallel=True, n_processors=cpu_worker_num)
                        Sobol_Si_ET_win          = sobol.analyze(problem, Y_ET[:, win_idx], calc_second_order=False, print_to_console=False,
                                                                 parallel=True, n_processors=cpu_worker_num)

                        Sobol_STi_NingDu_SF_df_win   = (Sobol_Si_NingDu_SF_win.to_df()[0].drop(columns={'ST_conf'}, axis=1).
                                                        rename(columns={'ST': f'ST{win_idx + 1}'}))
                        Sobol_STi_ShiCheng_SF_df_win = (Sobol_Si_ShiCheng_SF_win.to_df()[0].drop(columns={'ST_conf'}, axis=1).
                                                        rename(columns={'ST': f'ST{win_idx + 1}'}))
                        Sobol_STi_FenKeng_SF_df_win  = (Sobol_Si_FenKeng_SF_win.to_df()[0].drop(columns={'ST_conf'}, axis=1).
                                                        rename(columns={'ST': f'ST{win_idx + 1}'}))
                        Sobol_STi_ET_df_win          = (Sobol_Si_ET_win.to_df()[0].drop(columns={'ST_conf'}, axis=1).
                                                        rename(columns={'ST': f'ST{win_idx + 1}'}))

                        Sobol_STi_NingDu_SF_df   = pd.concat([Sobol_STi_NingDu_SF_df, Sobol_STi_NingDu_SF_df_win], axis=1, ignore_index=False)
                        Sobol_STi_ShiCheng_SF_df = pd.concat([Sobol_STi_ShiCheng_SF_df, Sobol_STi_ShiCheng_SF_df_win], axis=1, ignore_index=False)
                        Sobol_STi_FenKeng_SF_df  = pd.concat([Sobol_STi_FenKeng_SF_df, Sobol_STi_FenKeng_SF_df_win], axis=1, ignore_index=False)
                        Sobol_STi_ET_df          = pd.concat([Sobol_STi_ET_df, Sobol_STi_ET_df_win], axis=1, ignore_index=False)

                    # Mean Sobol Sensitivity Index
                    Sobol_STi_NingDu_SF_df['ST_mean']   = Sobol_STi_NingDu_SF_df.mean(axis=1)
                    Sobol_STi_ShiCheng_SF_df['ST_mean'] = Sobol_STi_ShiCheng_SF_df.mean(axis=1)
                    Sobol_STi_FenKeng_SF_df['ST_mean']  = Sobol_STi_FenKeng_SF_df.mean(axis=1)
                    Sobol_STi_ET_df['ST_mean']          = Sobol_STi_ET_df.mean(axis=1)

                    # 修改index列，去掉大括号
                    NingDu_SF_idx_list, ShiCheng_SF_idx_list, FenKeng_SF_idx_list, ET_idx_list = [], [], [], []
                    for NingDu_SF_idx, ShiCheng_SF_idx, FenKeng_SF_idx, ET_idx in zip(Sobol_STi_NingDu_SF_df.index, Sobol_STi_ShiCheng_SF_df.index,
                                                                                      Sobol_STi_FenKeng_SF_df.index, Sobol_STi_ET_df.index):
                        NingDu_SF_Si_idx, ShiCheng_SF_Si_idx, FenKeng_SF_Si_idx, ET_Si_idx \
                            = (NingDu_SF_idx.split('{')[0], ShiCheng_SF_idx.split('{')[0], FenKeng_SF_idx.split('{')[0], ET_idx.split('{')[0])
                        NingDu_SF_idx_list.append(NingDu_SF_Si_idx)
                        ShiCheng_SF_idx_list.append(ShiCheng_SF_Si_idx)
                        FenKeng_SF_idx_list.append(FenKeng_SF_Si_idx)
                        ET_idx_list.append(ET_Si_idx)
                    Sobol_STi_NingDu_SF_df.index   = NingDu_SF_idx_list
                    Sobol_STi_ShiCheng_SF_df.index = ShiCheng_SF_idx_list
                    Sobol_STi_FenKeng_SF_df.index  = FenKeng_SF_idx_list
                    Sobol_STi_ET_df.index          = ET_idx_list

                    # index列添加列名
                    Sobol_STi_NingDu_SF_df.index.name   = 'Para'
                    Sobol_STi_ShiCheng_SF_df.index.name = 'Para'
                    Sobol_STi_FenKeng_SF_df.index.name  = 'Para'
                    Sobol_STi_ET_df.index.name          = 'Para'

                    # 使用groupby方法按Index列分组，并计算平均值，最后按列SA_mean值排序
                    Sobol_STi_NingDu_SF_df_group   = Sobol_STi_NingDu_SF_df.groupby('Para').mean().sort_values(by='ST_mean', ascending=False)
                    Sobol_STi_ShiCheng_SF_df_group = Sobol_STi_ShiCheng_SF_df.groupby('Para').mean().sort_values(by='ST_mean', ascending=False)
                    Sobol_STi_FenKeng_SF_df_group  = Sobol_STi_FenKeng_SF_df.groupby('Para').mean().sort_values(by='ST_mean', ascending=False)
                    Sobol_STi_ET_df_group          = Sobol_STi_ET_df.groupby('Para').mean().sort_values(by='ST_mean', ascending=False)
                    print('Sobol_STi_NingDu_SF_df_group:', Sobol_STi_NingDu_SF_df_group.shape, '\n', Sobol_STi_NingDu_SF_df_group)
                    print('Sobol_STi_ShiCheng_SF_df_group:', Sobol_STi_ShiCheng_SF_df_group.shape, '\n', Sobol_STi_ShiCheng_SF_df_group)
                    print('Sobol_STi_FenKeng_SF_df_group:', Sobol_STi_FenKeng_SF_df_group.shape, '\n', Sobol_STi_FenKeng_SF_df_group)
                    print('Sobol_STi_ET_df_group:', Sobol_STi_ET_df_group.shape, '\n', Sobol_STi_ET_df_group)

                    # 5) Result Output
                    Sobol_STi_NingDu_SF_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'Sobol_Sensitivity_TVSA_NingDu.xlsx', index=True)
                    Sobol_STi_ShiCheng_SF_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'Sobol_Sensitivity_TVSA_ShiCheng.xlsx', index=True)
                    Sobol_STi_FenKeng_SF_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'Sobol_Sensitivity_TVSA_FenKeng.xlsx', index=True)
                    Sobol_STi_ET_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'Sobol_Sensitivity_TVSA_ET.xlsx', index=True)

                    # 6) TVSA Plotting
                    Sobol_STi_max = math.ceil(max(Sobol_STi_NingDu_SF_df_group.max(axis=0).max(), Sobol_STi_ShiCheng_SF_df_group.max(axis=0).max(),
                                                  Sobol_STi_FenKeng_SF_df_group.max(axis=0).max(), Sobol_STi_ET_df_group.max(axis=0).max()) * 10) / 10.0
                    print('Sobol_STi_max: ', Sobol_STi_max)
                    for hydro_sta_idx in zip(['NingDu_SF', 'ShiCheng_SF', 'FenKeng_SF', 'ET'],
                                             [Sobol_STi_NingDu_SF_df_group, Sobol_STi_ShiCheng_SF_df_group,
                                              Sobol_STi_FenKeng_SF_df_group, Sobol_STi_ET_df_group]):
                        print(hydro_sta_idx[0])
                        fig = plt.figure(figsize=(16, 10), dpi=500)
                        # 创建一个4行4列的GridSpec对象
                        grid = gridspec.GridSpec(nrows=13, ncols=18, wspace=0.2, hspace=2.8)
                        ax1 = plt.subplot(grid[2:-1, :-2])
                        ax2 = plt.subplot(grid[:2, :-2])
                        ax3 = plt.subplot(grid[2:-1, -2:])
                        ax4 = plt.subplot(grid[-1, :-2])
                        # ax1
                        ticks_interval = 22
                        sns.heatmap(hydro_sta_idx[1].iloc[:, :-1], vmin=0, vmax=Sobol_STi_max, cmap='gist_earth_r', cbar=True, cbar_ax=ax4,
                                    cbar_kws={'orientation': 'horizontal'}, xticklabels=ticks_interval, yticklabels='auto', ax=ax1)
                        xticks = ax1.get_xticks()
                        yticks = ax1.get_yticks()
                        #
                        if hydro_sta_idx[0] == 'ET':
                            ax1.set_title(label=f'Sobol Sensitivity Index (S$_i$)', pad=7, fontsize=12)
                        else:
                            ax1.set_title(label=f'Sobol Sensitivity Index (S$_i$)-{hydro_sta_idx[0].split("_")[0]} Station', pad=7, fontsize=12)
                        ax1.set_xticks(xticks, labels=TVSA_date[::ticks_interval], rotation=18, ha='right')
                        ax1.set_ylabel('')
                        # ax2
                        ax2.set_xticks([])
                        ax2.set_xticklabels([])
                        #
                        if hydro_sta_idx[0] == 'ET':
                            ax2.plot(date_range, obs_et_w_data, color='grey', linewidth=1.0)
                        else:
                            ax2.plot(date_range, obs_sf_data[hydro_stas[hydro_sta_idx[0].split('_')[0]]], color='grey', linewidth=1.0)
                        ax2.set_xlim([-1, len(date_range) + 1])
                        #
                        if hydro_sta_idx[0] == 'ET':
                            ax2.set_ylabel('ET\n(mm)')
                        else:
                            ax2.set_ylabel('Streamflow\n(m$^3$/s)')
                        # ax3
                        ax3.sharey(ax1)  # 共享y轴
                        ax3.tick_params(axis='y', labelleft=False)  # 不显示y轴刻度标签
                        # 绘制横向柱状图
                        ax3.barh(yticks, hydro_sta_idx[1]['ST_mean'], color='grey')
                        ax3.set_xlabel(xlabel=r'Mean ST$_i$', fontsize=12)
                        plt.savefig(f'{SWAT_Execution_Run.swat_nsga_out}\\Sobol_Sensitivity_Index_TVSA_{hydro_sta_idx[0]}.jpg', bbox_inches='tight')
                        plt.show()
                    # Exit the current program after completing sensitivity analysis
                    print('Sobol’ Sensitivity Analysis Finished!')
                    sys.exit()
                # PAWN Sensitivity Analysis
                elif SA_method == 'PAWN':
                    # Time-varying sensitivity analysis
                    PAWN_Si_NingDu_SF_df, PAWN_Si_ShiCheng_SF_df, PAWN_Si_FenKeng_SF_df, PAWN_Si_ET_df \
                        = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                    for win_idx in tqdm(range(Y_NingDu_SF.shape[1]), total=Y_NingDu_SF.shape[1]):
                        # 4) Perform Analysis
                        PAWN_Si_NingDu_SF_win   = pawn.analyze(problem, latin_param_values, Y_NingDu_SF[:, win_idx], S=20, print_to_console=False)
                        PAWN_Si_ShiCheng_SF_win = pawn.analyze(problem, latin_param_values, Y_ShiCheng_SF[:, win_idx], S=20, print_to_console=False)
                        PAWN_Si_FenKeng_SF_win  = pawn.analyze(problem, latin_param_values, Y_FenKeng_SF[:, win_idx], S=20, print_to_console=False)
                        PAWN_Si_ET_win          = pawn.analyze(problem, latin_param_values, Y_ET[:, win_idx], S=20, print_to_console=False)

                        PAWN_Si_NingDu_SF_df_win   = (PAWN_Si_NingDu_SF_win.to_df().drop(columns={'minimum', 'mean', 'maximum', 'CV'}, axis=1).
                                                      rename(columns={'median': f'median{win_idx + 1}'}))
                        PAWN_Si_ShiCheng_SF_df_win = (PAWN_Si_ShiCheng_SF_win.to_df().drop(columns={'minimum', 'mean', 'maximum', 'CV'}, axis=1).
                                                      rename(columns={'median': f'median{win_idx + 1}'}))
                        PAWN_Si_FenKeng_SF_df_win  = (PAWN_Si_FenKeng_SF_win.to_df().drop(columns={'minimum', 'mean', 'maximum', 'CV'}, axis=1).
                                                      rename(columns={'median': f'median{win_idx + 1}'}))
                        PAWN_Si_ET_df_win          = (PAWN_Si_ET_win.to_df().drop(columns={'minimum', 'mean', 'maximum', 'CV'}, axis=1).
                                                      rename(columns={'median': f'median{win_idx + 1}'}))

                        PAWN_Si_NingDu_SF_df   = pd.concat([PAWN_Si_NingDu_SF_df, PAWN_Si_NingDu_SF_df_win], axis=1, ignore_index=False)
                        PAWN_Si_ShiCheng_SF_df = pd.concat([PAWN_Si_ShiCheng_SF_df, PAWN_Si_ShiCheng_SF_df_win], axis=1, ignore_index=False)
                        PAWN_Si_FenKeng_SF_df  = pd.concat([PAWN_Si_FenKeng_SF_df, PAWN_Si_FenKeng_SF_df_win], axis=1, ignore_index=False)
                        PAWN_Si_ET_df          = pd.concat([PAWN_Si_ET_df, PAWN_Si_ET_df_win], axis=1, ignore_index=False)

                    # Mean PAWN Sensitivity Index
                    PAWN_Si_NingDu_SF_df['median_mean']   = PAWN_Si_NingDu_SF_df.mean(axis=1)
                    PAWN_Si_ShiCheng_SF_df['median_mean'] = PAWN_Si_ShiCheng_SF_df.mean(axis=1)
                    PAWN_Si_FenKeng_SF_df['median_mean']  = PAWN_Si_FenKeng_SF_df.mean(axis=1)
                    PAWN_Si_ET_df['median_mean']          = PAWN_Si_ET_df.mean(axis=1)

                    # 修改index列，去掉大括号
                    NingDu_SF_idx_list, ShiCheng_SF_idx_list, FenKeng_SF_idx_list, ET_idx_list = [], [], [], []
                    for NingDu_SF_idx, ShiCheng_SF_idx, FenKeng_SF_idx, ET_idx in zip(PAWN_Si_NingDu_SF_df.index, PAWN_Si_ShiCheng_SF_df.index,
                                                                                      PAWN_Si_FenKeng_SF_df.index, PAWN_Si_ET_df.index):
                        NingDu_SF_Si_idx, ShiCheng_SF_Si_idx, FenKeng_SF_Si_idx, ET_Si_idx \
                            = (NingDu_SF_idx.split('{')[0], ShiCheng_SF_idx.split('{')[0], FenKeng_SF_idx.split('{')[0], ET_idx.split('{')[0])
                        NingDu_SF_idx_list.append(NingDu_SF_Si_idx)
                        ShiCheng_SF_idx_list.append(ShiCheng_SF_Si_idx)
                        FenKeng_SF_idx_list.append(FenKeng_SF_Si_idx)
                        ET_idx_list.append(ET_Si_idx)
                    PAWN_Si_NingDu_SF_df.index   = NingDu_SF_idx_list
                    PAWN_Si_ShiCheng_SF_df.index = ShiCheng_SF_idx_list
                    PAWN_Si_FenKeng_SF_df.index  = FenKeng_SF_idx_list
                    PAWN_Si_ET_df.index          = ET_idx_list

                    # index列添加列名
                    PAWN_Si_NingDu_SF_df.index.name   = 'Para'
                    PAWN_Si_ShiCheng_SF_df.index.name = 'Para'
                    PAWN_Si_FenKeng_SF_df.index.name  = 'Para'
                    PAWN_Si_ET_df.index.name          = 'Para'

                    # 使用groupby方法按Index列分组，并计算平均值，最后按列SA_mean值排序
                    PAWN_Si_NingDu_SF_df_group   = PAWN_Si_NingDu_SF_df.groupby('Para').mean().sort_values(by='median_mean', ascending=False)
                    PAWN_Si_ShiCheng_SF_df_group = PAWN_Si_ShiCheng_SF_df.groupby('Para').mean().sort_values(by='median_mean', ascending=False)
                    PAWN_Si_FenKeng_SF_df_group  = PAWN_Si_FenKeng_SF_df.groupby('Para').mean().sort_values(by='median_mean', ascending=False)
                    PAWN_Si_ET_df_group          = PAWN_Si_ET_df.groupby('Para').mean().sort_values(by='median_mean', ascending=False)
                    print('PAWN_Si_NingDu_SF_df_group:', PAWN_Si_NingDu_SF_df_group.shape, '\n', PAWN_Si_NingDu_SF_df_group)
                    print('PAWN_Si_ShiCheng_SF_df_group:', PAWN_Si_ShiCheng_SF_df_group.shape, '\n', PAWN_Si_ShiCheng_SF_df_group)
                    print('PAWN_Si_FenKeng_SF_df_group:', PAWN_Si_FenKeng_SF_df_group.shape, '\n', PAWN_Si_FenKeng_SF_df_group)
                    print('PAWN_Si_ET_df_group:', PAWN_Si_ET_df_group.shape, '\n', PAWN_Si_ET_df_group)

                    # 5) Result Output
                    PAWN_Si_NingDu_SF_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'PAWN_Sensitivity_TVSA_NingDu.xlsx', index=True)
                    PAWN_Si_ShiCheng_SF_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'PAWN_Sensitivity_TVSA_ShiCheng.xlsx', index=True)
                    PAWN_Si_FenKeng_SF_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'PAWN_Sensitivity_TVSA_FenKeng.xlsx', index=True)
                    PAWN_Si_ET_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'PAWN_Sensitivity_TVSA_ET.xlsx', index=True)

                    # 6) TVSA Plotting
                    PAWN_Si_max = math.ceil(max(PAWN_Si_NingDu_SF_df_group.max(axis=0).max(), PAWN_Si_ShiCheng_SF_df_group.max(axis=0).max(),
                                                PAWN_Si_FenKeng_SF_df_group.max(axis=0).max(), PAWN_Si_ET_df_group.max(axis=0).max()) * 10) / 10.0
                    print('PAWN_Si_max: ', PAWN_Si_max)
                    for hydro_sta_idx in zip(['NingDu_SF', 'ShiCheng_SF', 'FenKeng_SF', 'ET'],
                                             [PAWN_Si_NingDu_SF_df_group, PAWN_Si_ShiCheng_SF_df_group,
                                              PAWN_Si_FenKeng_SF_df_group, PAWN_Si_ET_df_group]):
                        print(hydro_sta_idx[0])
                        fig = plt.figure(figsize=(16, 10), dpi=500)
                        # 创建一个4行4列的GridSpec对象
                        grid = gridspec.GridSpec(nrows=13, ncols=18, wspace=0.2, hspace=2.8)
                        ax1 = plt.subplot(grid[2:-1, :-2])
                        ax2 = plt.subplot(grid[:2, :-2])
                        ax3 = plt.subplot(grid[2:-1, -2:])
                        ax4 = plt.subplot(grid[-1, :-2])
                        # ax1
                        ticks_interval = 22
                        sns.heatmap(hydro_sta_idx[1].iloc[:, :-1], vmin=0, vmax=PAWN_Si_max, cmap='gist_earth_r', cbar=True, cbar_ax=ax4,
                                    cbar_kws={'orientation': 'horizontal'}, xticklabels=ticks_interval, yticklabels='auto', ax=ax1)
                        xticks = ax1.get_xticks()
                        yticks = ax1.get_yticks()
                        #
                        if hydro_sta_idx[0] == 'ET':
                            ax1.set_title(label=f'PAWN Sensitivity Index (S$_i$)', pad=7, fontsize=12)
                        else:
                            ax1.set_title(label=f'PAWN Sensitivity Index (S$_i$)-{hydro_sta_idx[0].split("_")[0]} Station', pad=7, fontsize=12)
                        ax1.set_xticks(xticks, labels=TVSA_date[::ticks_interval], rotation=18, ha='right')
                        ax1.set_ylabel('')
                        # ax2
                        ax2.set_xticks([])
                        ax2.set_xticklabels([])
                        #
                        if hydro_sta_idx[0] == 'ET':
                            ax2.plot(date_range, obs_et_w_data, color='grey', linewidth=1.0)
                        else:
                            ax2.plot(date_range, obs_sf_data[hydro_stas[hydro_sta_idx[0].split('_')[0]]], color='grey', linewidth=1.0)
                        ax2.set_xlim([-1, len(date_range) + 1])
                        #
                        if hydro_sta_idx[0] == 'ET':
                            ax2.set_ylabel('ET\n(mm)')
                        else:
                            ax2.set_ylabel('Streamflow\n(m$^3$/s)')
                        # ax3
                        ax3.sharey(ax1)  # 共享y轴
                        ax3.tick_params(axis='y', labelleft=False)  # 不显示y轴刻度标签
                        # 绘制横向柱状图
                        ax3.barh(yticks, hydro_sta_idx[1]['median_mean'], color='grey')
                        ax3.set_xlabel(xlabel=r'Mean S$_i$', fontsize=12)
                        plt.savefig(f'{SWAT_Execution_Run.swat_nsga_out}\\PAWN_Sensitivity_Index_TVSA_{hydro_sta_idx[0]}.jpg', bbox_inches='tight')
                        plt.show()
                    # Exit the current program after completing sensitivity analysis
                    print('PAWN Sensitivity Analysis Finished!')
                    sys.exit()
            elif cal_vars_list == ['Streamflow', 'LAI', 'ET']:
                obs_sf_data    = SWAT_Execution_Run.obs_sf_data
                obs_lai_w_data = SWAT_Execution_Run.LAI_obs_data_dict_area_w
                obs_et_w_data  = SWAT_Execution_Run.ET_obs_data_dict_area_w
                # 创建日期序列
                date_range = pd.date_range(start=f'{cal_period[0]}-01-01', end=f'{cal_period[1]}-12-31', freq='D').strftime('%Y-%m-%d')
                winsize = 2 * half_win + 1
                TVSA_date = [date_range[win_idx + half_win] if (win_idx + half_win) < len(date_range)
                             else date_range[int(win_idx + (len(date_range) - win_idx) / 2)]
                             for win_idx in range(0, len(date_range), winsize)]
                print('TVSA_date:', len(TVSA_date))

                Y_NingDu_SF   = np.array(obj_fun_list_SF_NingDu)
                Y_ShiCheng_SF = np.array(obj_fun_list_SF_ShiCheng)
                Y_FenKeng_SF  = np.array(obj_fun_list_SF_FenKeng)
                Y_LAI         = np.array(obj_fun_list_LAI)
                Y_ET          = np.array(obj_fun_list_ET)
                print('Y_NingDu_SF:', Y_NingDu_SF.shape)
                print('Y_ShiCheng_SF:', Y_ShiCheng_SF.shape)
                print('Y_FenKeng_SF:', Y_FenKeng_SF.shape)
                print('Y_LAI:', Y_LAI.shape)
                print('Y_ET:', Y_ET.shape)

                # Sobol’ Sensitivity Analysis
                if SA_method == 'Sobol':
                    # Time-varying sensitivity analysis
                    Sobol_STi_NingDu_SF_df, Sobol_STi_ShiCheng_SF_df, Sobol_STi_FenKeng_SF_df, Sobol_STi_LAI_df, Sobol_STi_ET_df \
                        = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                    for win_idx in tqdm(range(Y_NingDu_SF.shape[1]), total=Y_NingDu_SF.shape[1]):
                        # 4) Perform Analysis
                        Sobol_Si_NingDu_SF_win   = sobol.analyze(problem, Y_NingDu_SF[:, win_idx], calc_second_order=False, print_to_console=False,
                                                                 parallel=True, n_processors=cpu_worker_num)
                        Sobol_Si_ShiCheng_SF_win = sobol.analyze(problem, Y_ShiCheng_SF[:, win_idx], calc_second_order=False, print_to_console=False,
                                                                 parallel=True, n_processors=cpu_worker_num)
                        Sobol_Si_FenKeng_SF_win  = sobol.analyze(problem, Y_FenKeng_SF[:, win_idx], calc_second_order=False, print_to_console=False,
                                                                 parallel=True, n_processors=cpu_worker_num)
                        Sobol_Si_LAI_win         = sobol.analyze(problem, Y_LAI[:, win_idx], calc_second_order=False, print_to_console=False,
                                                                 parallel=True, n_processors=cpu_worker_num)
                        Sobol_Si_ET_win          = sobol.analyze(problem, Y_ET[:, win_idx], calc_second_order=False, print_to_console=False,
                                                                 parallel=True, n_processors=cpu_worker_num)

                        Sobol_STi_NingDu_SF_df_win   = (Sobol_Si_NingDu_SF_win.to_df()[0].drop(columns={'ST_conf'}, axis=1).
                                                        rename(columns={'ST': f'ST{win_idx + 1}'}))
                        Sobol_STi_ShiCheng_SF_df_win = (Sobol_Si_ShiCheng_SF_win.to_df()[0].drop(columns={'ST_conf'}, axis=1).
                                                        rename(columns={'ST': f'ST{win_idx + 1}'}))
                        Sobol_STi_FenKeng_SF_df_win  = (Sobol_Si_FenKeng_SF_win.to_df()[0].drop(columns={'ST_conf'}, axis=1).
                                                        rename(columns={'ST': f'ST{win_idx + 1}'}))
                        Sobol_STi_LAI_df_win         = (Sobol_Si_LAI_win.to_df()[0].drop(columns={'ST_conf'}, axis=1).
                                                        rename(columns={'ST': f'ST{win_idx + 1}'}))
                        Sobol_STi_ET_df_win          = (Sobol_Si_ET_win.to_df()[0].drop(columns={'ST_conf'}, axis=1).
                                                        rename(columns={'ST': f'ST{win_idx + 1}'}))

                        Sobol_STi_NingDu_SF_df   = pd.concat([Sobol_STi_NingDu_SF_df, Sobol_STi_NingDu_SF_df_win], axis=1, ignore_index=False)
                        Sobol_STi_ShiCheng_SF_df = pd.concat([Sobol_STi_ShiCheng_SF_df, Sobol_STi_ShiCheng_SF_df_win], axis=1, ignore_index=False)
                        Sobol_STi_FenKeng_SF_df  = pd.concat([Sobol_STi_FenKeng_SF_df, Sobol_STi_FenKeng_SF_df_win], axis=1, ignore_index=False)
                        Sobol_STi_LAI_df         = pd.concat([Sobol_STi_LAI_df, Sobol_STi_LAI_df_win], axis=1, ignore_index=False)
                        Sobol_STi_ET_df          = pd.concat([Sobol_STi_ET_df, Sobol_STi_ET_df_win], axis=1, ignore_index=False)

                    # Mean Sobol Sensitivity Index
                    Sobol_STi_NingDu_SF_df['ST_mean']   = Sobol_STi_NingDu_SF_df.mean(axis=1)
                    Sobol_STi_ShiCheng_SF_df['ST_mean'] = Sobol_STi_ShiCheng_SF_df.mean(axis=1)
                    Sobol_STi_FenKeng_SF_df['ST_mean']  = Sobol_STi_FenKeng_SF_df.mean(axis=1)
                    Sobol_STi_LAI_df['ST_mean']         = Sobol_STi_LAI_df.mean(axis=1)
                    Sobol_STi_ET_df['ST_mean']          = Sobol_STi_ET_df.mean(axis=1)

                    # 修改index列，去掉大括号
                    NingDu_SF_idx_list, ShiCheng_SF_idx_list, FenKeng_SF_idx_list, LAI_idx_list, ET_idx_list = [], [], [], [], []
                    for NingDu_SF_idx, ShiCheng_SF_idx, FenKeng_SF_idx, LAI_idx, ET_idx in zip(Sobol_STi_NingDu_SF_df.index, Sobol_STi_ShiCheng_SF_df.index,
                                                                                               Sobol_STi_FenKeng_SF_df.index, Sobol_STi_LAI_df.index,
                                                                                               Sobol_STi_ET_df.index):
                        NingDu_SF_Si_idx, ShiCheng_SF_Si_idx, FenKeng_SF_Si_idx, LAI_Si_idx, ET_Si_idx \
                            = (NingDu_SF_idx.split('{')[0], ShiCheng_SF_idx.split('{')[0], FenKeng_SF_idx.split('{')[0], LAI_idx.split('{')[0], ET_idx.split('{')[0])
                        NingDu_SF_idx_list.append(NingDu_SF_Si_idx)
                        ShiCheng_SF_idx_list.append(ShiCheng_SF_Si_idx)
                        FenKeng_SF_idx_list.append(FenKeng_SF_Si_idx)
                        LAI_idx_list.append(LAI_Si_idx)
                        ET_idx_list.append(ET_Si_idx)
                    Sobol_STi_NingDu_SF_df.index   = NingDu_SF_idx_list
                    Sobol_STi_ShiCheng_SF_df.index = ShiCheng_SF_idx_list
                    Sobol_STi_FenKeng_SF_df.index  = FenKeng_SF_idx_list
                    Sobol_STi_LAI_df.index         = LAI_idx_list
                    Sobol_STi_ET_df.index          = ET_idx_list

                    # index列添加列名
                    Sobol_STi_NingDu_SF_df.index.name   = 'Para'
                    Sobol_STi_ShiCheng_SF_df.index.name = 'Para'
                    Sobol_STi_FenKeng_SF_df.index.name  = 'Para'
                    Sobol_STi_LAI_df.index.name         = 'Para'
                    Sobol_STi_ET_df.index.name          = 'Para'

                    # 使用groupby方法按Index列分组，并计算平均值，最后按列SA_mean值排序
                    Sobol_STi_NingDu_SF_df_group   = Sobol_STi_NingDu_SF_df.groupby('Para').mean().sort_values(by='ST_mean', ascending=False)
                    Sobol_STi_ShiCheng_SF_df_group = Sobol_STi_ShiCheng_SF_df.groupby('Para').mean().sort_values(by='ST_mean', ascending=False)
                    Sobol_STi_FenKeng_SF_df_group  = Sobol_STi_FenKeng_SF_df.groupby('Para').mean().sort_values(by='ST_mean', ascending=False)
                    Sobol_STi_LAI_df_group         = Sobol_STi_LAI_df.groupby('Para').mean().sort_values(by='ST_mean', ascending=False)
                    Sobol_STi_ET_df_group          = Sobol_STi_ET_df.groupby('Para').mean().sort_values(by='ST_mean', ascending=False)
                    print('Sobol_STi_NingDu_SF_df_group:', Sobol_STi_NingDu_SF_df_group.shape, '\n', Sobol_STi_NingDu_SF_df_group)
                    print('Sobol_STi_ShiCheng_SF_df_group:', Sobol_STi_ShiCheng_SF_df_group.shape, '\n', Sobol_STi_ShiCheng_SF_df_group)
                    print('Sobol_STi_FenKeng_SF_df_group:', Sobol_STi_FenKeng_SF_df_group.shape, '\n', Sobol_STi_FenKeng_SF_df_group)
                    print('Sobol_STi_LAI_df_group:', Sobol_STi_LAI_df_group.shape, '\n', Sobol_STi_LAI_df_group)
                    print('Sobol_STi_ET_df_group:', Sobol_STi_ET_df_group.shape, '\n', Sobol_STi_ET_df_group)

                    # 5) Result Output
                    Sobol_STi_NingDu_SF_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'Sobol_Sensitivity_TVSA_NingDu.xlsx', index=True)
                    Sobol_STi_ShiCheng_SF_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'Sobol_Sensitivity_TVSA_ShiCheng.xlsx', index=True)
                    Sobol_STi_FenKeng_SF_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'Sobol_Sensitivity_TVSA_FenKeng.xlsx', index=True)
                    Sobol_STi_LAI_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'Sobol_Sensitivity_TVSA_LAI.xlsx', index=True)
                    Sobol_STi_ET_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'Sobol_Sensitivity_TVSA_ET.xlsx', index=True)

                    # 6) TVSA Plotting
                    Sobol_STi_max = math.ceil(max(Sobol_STi_NingDu_SF_df_group.max(axis=0).max(), Sobol_STi_ShiCheng_SF_df_group.max(axis=0).max(),
                                                  Sobol_STi_FenKeng_SF_df_group.max(axis=0).max(), Sobol_STi_LAI_df_group.max(axis=0).max(),
                                                  Sobol_STi_ET_df_group.max(axis=0).max()) * 10) / 10.0
                    print('Sobol_STi_max: ', Sobol_STi_max)
                    for hydro_sta_idx in zip(['NingDu_SF', 'ShiCheng_SF', 'FenKeng_SF', 'LAI', 'ET'],
                                             [Sobol_STi_NingDu_SF_df_group, Sobol_STi_ShiCheng_SF_df_group,
                                              Sobol_STi_FenKeng_SF_df_group, Sobol_STi_LAI_df_group, Sobol_STi_ET_df_group]):
                        print(hydro_sta_idx[0])
                        fig = plt.figure(figsize=(16, 10), dpi=500)
                        # 创建一个4行4列的GridSpec对象
                        grid = gridspec.GridSpec(nrows=13, ncols=18, wspace=0.2, hspace=2.8)
                        ax1 = plt.subplot(grid[2:-1, :-2])
                        ax2 = plt.subplot(grid[:2, :-2])
                        ax3 = plt.subplot(grid[2:-1, -2:])
                        ax4 = plt.subplot(grid[-1, :-2])
                        # ax1
                        ticks_interval = 22
                        sns.heatmap(hydro_sta_idx[1].iloc[:, :-1], vmin=0, vmax=Sobol_STi_max, cmap='gist_earth_r', cbar=True, cbar_ax=ax4,
                                    cbar_kws={'orientation': 'horizontal'}, xticklabels=ticks_interval, yticklabels='auto', ax=ax1)
                        xticks = ax1.get_xticks()
                        yticks = ax1.get_yticks()
                        #
                        if hydro_sta_idx[0] == 'LAI' or hydro_sta_idx[0] == 'ET':
                            ax1.set_title(label=f'Sobol Sensitivity Index (S$_i$)', pad=7, fontsize=12)
                        else:
                            ax1.set_title(label=f'Sobol Sensitivity Index (S$_i$)-{hydro_sta_idx[0].split("_")[0]} Station', pad=7, fontsize=12)
                        ax1.set_xticks(xticks, labels=TVSA_date[::ticks_interval], rotation=18, ha='right')
                        ax1.set_ylabel('')
                        # ax2
                        ax2.set_xticks([])
                        ax2.set_xticklabels([])
                        #
                        if hydro_sta_idx[0] == 'LAI':
                            ax2.plot(date_range, obs_lai_w_data, color='grey', linewidth=1.0)
                        elif hydro_sta_idx[0] == 'ET':
                            ax2.plot(date_range, obs_et_w_data, color='grey', linewidth=1.0)
                        else:
                            ax2.plot(date_range, obs_sf_data[hydro_stas[hydro_sta_idx[0].split('_')[0]]], color='grey',
                                     linewidth=1.0)
                        ax2.set_xlim([-1, len(date_range) + 1])
                        #
                        if hydro_sta_idx[0] == 'LAI':
                            ax2.set_ylabel('LAI\n(m$^2$/m$^2$)')
                        elif hydro_sta_idx[0] == 'ET':
                            ax2.set_ylabel('ET\n(mm)')
                        else:
                            ax2.set_ylabel('Streamflow\n(m$^3$/s)')
                        # ax3
                        ax3.sharey(ax1)  # 共享y轴
                        ax3.tick_params(axis='y', labelleft=False)  # 不显示y轴刻度标签
                        # 绘制横向柱状图
                        ax3.barh(yticks, hydro_sta_idx[1]['ST_mean'], color='grey')
                        ax3.set_xlabel(xlabel=r'Mean ST$_i$', fontsize=12)
                        plt.savefig(f'{SWAT_Execution_Run.swat_nsga_out}\\Sobol_Sensitivity_Index_TVSA_{hydro_sta_idx[0]}.jpg', bbox_inches='tight')
                        plt.show()
                    # Exit the current program after completing sensitivity analysis
                    print('Sobol’ Sensitivity Analysis Finished!')
                    sys.exit()
                # PAWN Sensitivity Analysis
                elif SA_method == 'PAWN':
                    # Time-varying sensitivity analysis
                    PAWN_Si_NingDu_SF_df, PAWN_Si_ShiCheng_SF_df, PAWN_Si_FenKeng_SF_df, PAWN_Si_LAI_df, PAWN_Si_ET_df \
                        = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                    for win_idx in tqdm(range(Y_NingDu_SF.shape[1]), total=Y_NingDu_SF.shape[1]):
                        # 4) Perform Analysis
                        PAWN_Si_NingDu_SF_win   = pawn.analyze(problem, latin_param_values, Y_NingDu_SF[:, win_idx], S=20, print_to_console=False)
                        PAWN_Si_ShiCheng_SF_win = pawn.analyze(problem, latin_param_values, Y_ShiCheng_SF[:, win_idx], S=20, print_to_console=False)
                        PAWN_Si_FenKeng_SF_win  = pawn.analyze(problem, latin_param_values, Y_FenKeng_SF[:, win_idx], S=20, print_to_console=False)
                        PAWN_Si_LAI_win         = pawn.analyze(problem, latin_param_values, Y_LAI[:, win_idx], S=20, print_to_console=False)
                        PAWN_Si_ET_win          = pawn.analyze(problem, latin_param_values, Y_ET[:, win_idx], S=20, print_to_console=False)

                        PAWN_Si_NingDu_SF_df_win   = (PAWN_Si_NingDu_SF_win.to_df().drop(columns={'minimum', 'mean', 'maximum', 'CV'}, axis=1).
                                                      rename(columns={'median': f'median{win_idx + 1}'}))
                        PAWN_Si_ShiCheng_SF_df_win = (PAWN_Si_ShiCheng_SF_win.to_df().drop(columns={'minimum', 'mean', 'maximum', 'CV'}, axis=1).
                                                      rename(columns={'median': f'median{win_idx + 1}'}))
                        PAWN_Si_FenKeng_SF_df_win  = (PAWN_Si_FenKeng_SF_win.to_df().drop(columns={'minimum', 'mean', 'maximum', 'CV'}, axis=1).
                                                      rename(columns={'median': f'median{win_idx + 1}'}))
                        PAWN_Si_LAI_df_win         = (PAWN_Si_LAI_win.to_df().drop(columns={'minimum', 'mean', 'maximum', 'CV'}, axis=1).
                                                      rename(columns={'median': f'median{win_idx + 1}'}))
                        PAWN_Si_ET_df_win          = (PAWN_Si_ET_win.to_df().drop(columns={'minimum', 'mean', 'maximum', 'CV'}, axis=1).
                                                      rename(columns={'median': f'median{win_idx + 1}'}))

                        PAWN_Si_NingDu_SF_df   = pd.concat([PAWN_Si_NingDu_SF_df, PAWN_Si_NingDu_SF_df_win], axis=1, ignore_index=False)
                        PAWN_Si_ShiCheng_SF_df = pd.concat([PAWN_Si_ShiCheng_SF_df, PAWN_Si_ShiCheng_SF_df_win], axis=1, ignore_index=False)
                        PAWN_Si_FenKeng_SF_df  = pd.concat([PAWN_Si_FenKeng_SF_df, PAWN_Si_FenKeng_SF_df_win], axis=1, ignore_index=False)
                        PAWN_Si_LAI_df         = pd.concat([PAWN_Si_LAI_df, PAWN_Si_LAI_df_win], axis=1, ignore_index=False)
                        PAWN_Si_ET_df          = pd.concat([PAWN_Si_ET_df, PAWN_Si_ET_df_win], axis=1, ignore_index=False)

                    # Mean PAWN Sensitivity Index
                    PAWN_Si_NingDu_SF_df['median_mean']   = PAWN_Si_NingDu_SF_df.mean(axis=1)
                    PAWN_Si_ShiCheng_SF_df['median_mean'] = PAWN_Si_ShiCheng_SF_df.mean(axis=1)
                    PAWN_Si_FenKeng_SF_df['median_mean']  = PAWN_Si_FenKeng_SF_df.mean(axis=1)
                    PAWN_Si_LAI_df['median_mean']         = PAWN_Si_LAI_df.mean(axis=1)
                    PAWN_Si_ET_df['median_mean']          = PAWN_Si_ET_df.mean(axis=1)

                    # 修改index列，去掉大括号
                    NingDu_SF_idx_list, ShiCheng_SF_idx_list, FenKeng_SF_idx_list, LAI_idx_list, ET_idx_list = [], [], [], [], []
                    for NingDu_SF_idx, ShiCheng_SF_idx, FenKeng_SF_idx, LAI_idx, ET_idx in zip(PAWN_Si_NingDu_SF_df.index, PAWN_Si_ShiCheng_SF_df.index,
                                                                                               PAWN_Si_FenKeng_SF_df.index, PAWN_Si_LAI_df.index,
                                                                                               PAWN_Si_ET_df.index):
                        NingDu_SF_Si_idx, ShiCheng_SF_Si_idx, FenKeng_SF_Si_idx, LAI_Si_idx, ET_Si_idx \
                            = (NingDu_SF_idx.split('{')[0], ShiCheng_SF_idx.split('{')[0], FenKeng_SF_idx.split('{')[0], LAI_idx.split('{')[0], ET_idx.split('{')[0])
                        NingDu_SF_idx_list.append(NingDu_SF_Si_idx)
                        ShiCheng_SF_idx_list.append(ShiCheng_SF_Si_idx)
                        FenKeng_SF_idx_list.append(FenKeng_SF_Si_idx)
                        LAI_idx_list.append(LAI_Si_idx)
                        ET_idx_list.append(ET_Si_idx)
                    PAWN_Si_NingDu_SF_df.index   = NingDu_SF_idx_list
                    PAWN_Si_ShiCheng_SF_df.index = ShiCheng_SF_idx_list
                    PAWN_Si_FenKeng_SF_df.index  = FenKeng_SF_idx_list
                    PAWN_Si_LAI_df.index         = LAI_idx_list
                    PAWN_Si_ET_df.index          = ET_idx_list

                    # index列添加列名
                    PAWN_Si_NingDu_SF_df.index.name   = 'Para'
                    PAWN_Si_ShiCheng_SF_df.index.name = 'Para'
                    PAWN_Si_FenKeng_SF_df.index.name  = 'Para'
                    PAWN_Si_LAI_df.index.name         = 'Para'
                    PAWN_Si_ET_df.index.name          = 'Para'

                    # 使用groupby方法按Index列分组，并计算平均值，最后按列SA_mean值排序
                    PAWN_Si_NingDu_SF_df_group   = PAWN_Si_NingDu_SF_df.groupby('Para').mean().sort_values(by='median_mean', ascending=False)
                    PAWN_Si_ShiCheng_SF_df_group = PAWN_Si_ShiCheng_SF_df.groupby('Para').mean().sort_values(by='median_mean', ascending=False)
                    PAWN_Si_FenKeng_SF_df_group  = PAWN_Si_FenKeng_SF_df.groupby('Para').mean().sort_values(by='median_mean', ascending=False)
                    PAWN_Si_LAI_df_group         = PAWN_Si_LAI_df.groupby('Para').mean().sort_values(by='median_mean', ascending=False)
                    PAWN_Si_ET_df_group          = PAWN_Si_ET_df.groupby('Para').mean().sort_values(by='median_mean', ascending=False)
                    print('PAWN_Si_NingDu_SF_df_group:', PAWN_Si_NingDu_SF_df_group.shape, '\n', PAWN_Si_NingDu_SF_df_group)
                    print('PAWN_Si_ShiCheng_SF_df_group:', PAWN_Si_ShiCheng_SF_df_group.shape, '\n', PAWN_Si_ShiCheng_SF_df_group)
                    print('PAWN_Si_FenKeng_SF_df_group:', PAWN_Si_FenKeng_SF_df_group.shape, '\n', PAWN_Si_FenKeng_SF_df_group)
                    print('PAWN_Si_LAI_df_group:', PAWN_Si_LAI_df_group.shape, '\n', PAWN_Si_LAI_df_group)
                    print('PAWN_Si_ET_df_group:', PAWN_Si_ET_df_group.shape, '\n', PAWN_Si_ET_df_group)

                    # 5) Result Output
                    PAWN_Si_NingDu_SF_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'PAWN_Sensitivity_TVSA_NingDu.xlsx', index=True)
                    PAWN_Si_ShiCheng_SF_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'PAWN_Sensitivity_TVSA_ShiCheng.xlsx', index=True)
                    PAWN_Si_FenKeng_SF_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'PAWN_Sensitivity_TVSA_FenKeng.xlsx', index=True)
                    PAWN_Si_LAI_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'PAWN_Sensitivity_TVSA_LAI.xlsx', index=True)
                    PAWN_Si_ET_df_group.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\'f'PAWN_Sensitivity_TVSA_ET.xlsx', index=True)

                    # 6) TVSA Plotting
                    PAWN_Si_max = math.ceil(max(PAWN_Si_NingDu_SF_df_group.max(axis=0).max(), PAWN_Si_ShiCheng_SF_df_group.max(axis=0).max(),
                                                PAWN_Si_FenKeng_SF_df_group.max(axis=0).max(), PAWN_Si_LAI_df_group.max(axis=0).max(),
                                                PAWN_Si_ET_df_group.max(axis=0).max()) * 10) / 10.0
                    print('PAWN_Si_max: ', PAWN_Si_max)
                    for hydro_sta_idx in zip(['NingDu_SF', 'ShiCheng_SF', 'FenKeng_SF', 'LAI', 'ET'],
                                             [PAWN_Si_NingDu_SF_df_group, PAWN_Si_ShiCheng_SF_df_group,
                                              PAWN_Si_FenKeng_SF_df_group, PAWN_Si_LAI_df_group, PAWN_Si_ET_df_group]):
                        print(hydro_sta_idx[0])
                        fig = plt.figure(figsize=(16, 10), dpi=500)
                        # 创建一个4行4列的GridSpec对象
                        grid = gridspec.GridSpec(nrows=13, ncols=18, wspace=0.2, hspace=2.8)
                        ax1 = plt.subplot(grid[2:-1, :-2])
                        ax2 = plt.subplot(grid[:2, :-2])
                        ax3 = plt.subplot(grid[2:-1, -2:])
                        ax4 = plt.subplot(grid[-1, :-2])
                        # ax1
                        ticks_interval = 22
                        sns.heatmap(hydro_sta_idx[1].iloc[:, :-1], vmin=0, vmax=PAWN_Si_max, cmap='gist_earth_r', cbar=True, cbar_ax=ax4,
                                    cbar_kws={'orientation': 'horizontal'}, xticklabels=ticks_interval, yticklabels='auto', ax=ax1)
                        xticks = ax1.get_xticks()
                        yticks = ax1.get_yticks()
                        #
                        if hydro_sta_idx[0] == 'LAI' or hydro_sta_idx[0] == 'ET':
                            ax1.set_title(label=f'PAWN Sensitivity Index (S$_i$)', pad=7, fontsize=12)
                        else:
                            ax1.set_title(label=f'PAWN Sensitivity Index (S$_i$)-{hydro_sta_idx[0].split("_")[0]} Station', pad=7, fontsize=12)
                        ax1.set_xticks(xticks, labels=TVSA_date[::ticks_interval], rotation=18, ha='right')
                        ax1.set_ylabel('')
                        # ax2
                        ax2.set_xticks([])
                        ax2.set_xticklabels([])
                        #
                        if hydro_sta_idx[0] == 'LAI':
                            ax2.plot(date_range, obs_lai_w_data, color='grey', linewidth=1.0)
                        elif hydro_sta_idx[0] == 'ET':
                            ax2.plot(date_range, obs_et_w_data, color='grey', linewidth=1.0)
                        else:
                            ax2.plot(date_range, obs_sf_data[hydro_stas[hydro_sta_idx[0].split('_')[0]]], color='grey', linewidth=1.0)
                        ax2.set_xlim([-1, len(date_range) + 1])
                        #
                        if hydro_sta_idx[0] == 'LAI':
                            ax2.set_ylabel('LAI\n(m$^2$/m$^2$)')
                        elif hydro_sta_idx[0] == 'ET':
                            ax2.set_ylabel('ET\n(mm)')
                        else:
                            ax2.set_ylabel('Streamflow\n(m$^3$/s)')
                        # ax3
                        ax3.sharey(ax1)  # 共享y轴
                        ax3.tick_params(axis='y', labelleft=False)  # 不显示y轴刻度标签
                        # 绘制横向柱状图
                        ax3.barh(yticks, hydro_sta_idx[1]['median_mean'], color='grey')
                        ax3.set_xlabel(xlabel=r'Mean S$_i$', fontsize=12)
                        plt.savefig(f'{SWAT_Execution_Run.swat_nsga_out}\\PAWN_Sensitivity_Index_TVSA_{hydro_sta_idx[0]}.jpg', bbox_inches='tight')
                        plt.show()
                    # Exit the current program after completing sensitivity analysis
                    print('PAWN Sensitivity Analysis Finished!')
                    sys.exit()
        else:
            SA_var = ''
            if cal_vars_list == ['Streamflow']:
                SA_var = 'Streamflow'
            elif cal_vars_list == ['LAI']:
                SA_var = 'LAI'
            elif cal_vars_list == ['BIOM']:
                SA_var = 'BIOM'
            # Sobol’ Sensitivity Analysis
            if SA_method == 'Sobol':
                Y = np.array(mod_run_obj_fun_list)
                print('Y:', Y.shape)
                print(Y)

                # 4) Perform Analysis
                Sobol_Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False, parallel=True, n_processors=cpu_worker_num)
                Sobol_total_Si, Sobol_first_Si = Sobol_Si.to_df()
                print('Sobol_total_Si:\n', Sobol_total_Si)
                print('Sobol_first_Si:\n', Sobol_first_Si)

                # 使用merge方法合并数据框，通过指定连接键（例如，使用index）来连接两个数据框
                Sobol_total_first_Si = Sobol_total_Si.merge(Sobol_first_Si, left_index=True, right_index=True)
                print('Sobol_total_first_Si:\n', Sobol_total_first_Si)
                print('\n')


                # 修改index列，去掉大括号
                total_var_idx_list, first_var_idx_list, total_first_var_idx_list = [], [], []
                for total_var_idx, first_var_idx, total_first_var_idx in zip(Sobol_total_Si.index, Sobol_first_Si.index, Sobol_total_first_Si.index):
                    total_var_Si_idx, first_var_Si_idx, total_first_var_Si_idx =\
                        total_var_idx.split('{')[0], first_var_idx.split('{')[0], total_first_var_idx.split('{')[0]
                    total_var_idx_list.append(total_var_Si_idx)
                    first_var_idx_list.append(first_var_Si_idx)
                    total_first_var_idx_list.append(total_first_var_Si_idx)
                Sobol_total_Si.index = total_var_idx_list
                Sobol_first_Si.index = first_var_idx_list
                Sobol_total_first_Si.index = total_first_var_idx_list

                # index列添加列名
                Sobol_total_Si.index.name = 'Para'
                Sobol_first_Si.index.name = 'Para'
                Sobol_total_first_Si.index.name = 'Para'

                # 使用groupby方法按Index列分组，并计算平均值，最后按列SA_mean值排序
                Sobol_total_Si_df_sort = Sobol_total_Si.groupby('Para').mean().sort_values(by='ST', ascending=False)
                Sobol_first_Si_df_sort = Sobol_first_Si.groupby('Para').mean().sort_values(by='S1', ascending=False)
                Sobol_total_first_Si_df_sort = Sobol_total_first_Si.groupby('Para').mean().sort_values(by='ST', ascending=False)
                print('Sobol_total_Si_df_sort:', Sobol_total_Si_df_sort.shape, '\n', Sobol_total_Si_df_sort)
                print('Sobol_first_Si_df_sort:', Sobol_first_Si_df_sort.shape, '\n', Sobol_first_Si_df_sort)
                print('Sobol_total_first_Si_df_sort:', Sobol_total_first_Si_df_sort.shape, '\n', Sobol_total_first_Si_df_sort)
                print('\n')

                # 访问索引列
                Sobol_total_Si_index_col = Sobol_total_Si_df_sort.index
                Sobol_first_Si_index_col = Sobol_first_Si_df_sort.index
                Sobol_total_first_Si_index_col = Sobol_total_first_Si_df_sort.index

                # 5) Result Output
                # 将数据框输出为Excel文件
                Sobol_total_first_Si_df_sort.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\Sobol_Sensitivity_{SA_var}.xlsx', index=True)

                # 6) Plotting
                # Total-order index
                plt.figure(figsize=(12, 12), dpi=300)
                axes = plt.axes()
                axes.spines['top'].set_visible(False)
                axes.spines['right'].set_visible(False)
                plt.barh(Sobol_total_Si_index_col, Sobol_total_Si_df_sort['ST'], height=0.5, xerr=Sobol_total_Si_df_sort['ST_conf'], align='center',
                         color='grey', capsize=4)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.xlabel('Total-order index', fontsize=16)
                plt.tight_layout()
                plt.savefig(f'{SWAT_Execution_Run.swat_nsga_out}\\Sobol_Total_order_index_{SA_var}.jpg')
                plt.show()

                # First-order index
                plt.figure(figsize=(12, 12), dpi=300)
                axes = plt.axes()
                axes.spines['top'].set_visible(False)
                axes.spines['right'].set_visible(False)
                plt.barh(Sobol_first_Si_index_col, Sobol_first_Si_df_sort['S1'], height=0.5, xerr=Sobol_first_Si_df_sort['S1_conf'], align='center',
                         color='grey', capsize=4)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.xlabel('Sobol First-order index', fontsize=16)
                plt.tight_layout()
                plt.savefig(f'{SWAT_Execution_Run.swat_nsga_out}\\Sobol_First_order_index_{SA_var}.jpg')
                plt.show()

                # Total-order index/First-order index
                Sobol_total_first_index = {'Total-order index': (Sobol_total_first_Si_df_sort['ST'], Sobol_total_first_Si_df_sort['ST_conf'], '#A1D99B'),
                                           'First-order index': (Sobol_total_first_Si_df_sort['S1'], Sobol_total_first_Si_df_sort['S1_conf'], '#9DC9E1')}
                y = np.arange(len(Sobol_total_first_Si_index_col))  # The label locations
                height = 0.35  # The height of the bars
                multiplier = 0
                plt.figure(figsize=(12, 12), dpi=300)
                axes = plt.axes()
                axes.spines['top'].set_visible(False)
                axes.spines['right'].set_visible(False)
                for order_idx, order_val in Sobol_total_first_index.items():
                    offset = height * multiplier
                    plt.barh(y + offset, order_val[0], align='center', height=height, xerr=order_val[1], capsize=3, color=order_val[2], ecolor='grey',
                             label=order_idx)
                    multiplier += 1
                plt.xticks(fontsize=16)
                plt.yticks(y + height / 2, Sobol_total_first_Si_index_col, fontsize=16)
                plt.xlabel('Sobol’ Sensitivity Index', fontsize=16)
                plt.legend(loc='best', fontsize=16)
                plt.tight_layout()
                plt.savefig(f'{SWAT_Execution_Run.swat_nsga_out}\\Sobol’_Sensitivity_Index_{SA_var}.jpg')
                plt.show()

                # Exit the current program after completing sensitivity analysis
                print('Sobol’ Sensitivity Analysis Finished!')
                sys.exit()
            # PAWN Sensitivity Analysis
            elif SA_method == 'PAWN':
                Y = np.array(mod_run_obj_fun_list)
                print('Y:', Y.shape)
                print(Y)

                # 4) Perform Analysis
                PAWN_Si = pawn.analyze(problem, latin_param_values, Y, S=20, print_to_console=False)
                PAWN_Si_df = PAWN_Si.to_df()
                print('PAWN_Si_df:\n', PAWN_Si_df)

                # 修改index列，去掉大括号
                Var_idx_list = []
                for Var_idx in PAWN_Si_df.index:
                    Var_Si_idx = Var_idx.split('{')[0]
                    Var_idx_list.append(Var_Si_idx)
                PAWN_Si_df.index = Var_idx_list

                # index列添加列名
                PAWN_Si_df.index.name = 'Para'

                # 使用groupby方法按Index列分组，并计算平均值，最后按列SA_mean值排序
                PAWN_Si_df_group_sort = PAWN_Si_df.groupby('Para').mean().sort_values(by='median', ascending=False)
                print('PAWN_Si_df_group_sort:', PAWN_Si_df_group_sort.shape, '\n', PAWN_Si_df_group_sort)

                # 访问索引列
                PAWN_Si_index_col = PAWN_Si_df_group_sort.index

                # 5) Result Output
                # 将数据框输出为Excel文件
                PAWN_Si_df_group_sort.to_excel(f'{SWAT_Execution_Run.swat_nsga_out}\\PAWN_Sensitivity_{SA_var}.xlsx', index=True)

                # 6) Plotting
                plt.figure(figsize=(12, 12), dpi=300)
                axes = plt.axes()
                axes.spines['top'].set_visible(False)
                axes.spines['right'].set_visible(False)
                plt.barh(PAWN_Si_index_col, PAWN_Si_df_group_sort['median'], height=0.8, align='center', color='grey', capsize=4)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.xlabel('PAWN Sensitivity Index', fontsize=16)
                plt.ylim(-1, len(PAWN_Si_df_group_sort.index))
                plt.tight_layout()
                plt.savefig(f'{SWAT_Execution_Run.swat_nsga_out}\\PAWN_Sensitivity_Index_{SA_var}.jpg')
                plt.show()

                # Exit the current program after completing sensitivity analysis
                print('PAWN Sensitivity Analysis Finished!')
                sys.exit()
        SA_end_time = time.time()
        print(f'Sensitivity Analysis Time: {round((SA_end_time - SA_start_time) / 60.0, 2)} min!')

    # Parameter Fix (Compromise solution)
    if para_fix_mode:
        print('Para Fix:')
        # Read Parameters
        compromise_txt = r'Compromise_solution_Streamflow.txt'
        if cal_vars_list == ['Streamflow']:
            if cal_scheme == 'Multi-site':
                compromise_txt = r'Compromise_solution_Streamflow.txt'
        elif cal_vars_list == ['LAI']:
            if cal_scheme == 'Multi-objective':
                compromise_txt = r'Compromise_solution_LAI_Multi-objective.txt'
        elif cal_vars_list == ['BIOM']:
            if cal_scheme == 'Multi-objective':
                compromise_txt = r'Compromise_solution_BIOM_Multi-objective.txt'
        elif cal_vars_list == ['Streamflow', 'ET']:
            if cal_scheme == 'Multi-variable':
                compromise_txt = r'Compromise_solution_Streamflow_ET.txt'
        print('compromise_txt:', compromise_txt)
        with open(f'{swat_nsga_out}\\{compromise_txt}', 'r') as file:
            param_val_arr = np.array([float(val) for val in file.readlines()[1].split()][:len(swat_parameter)])
        print('param_val_arr:', param_val_arr.shape, '\n', param_val_arr)

        # 如果目标文件夹存在，则删除它
        if os.path.exists(f'{swat_model}\\Compro_Sol'):
            shutil.rmtree(f'{swat_model}\\Compro_Sol')
            print('Delete Compro_sol folder finished!')
        # Copy TxtInOut Folder
        shutil.copytree(f'{swat_TxtInOut}', f'{swat_model}\\Compro_sol')
        print('Copy to Compro_sol folder finished!')

        # SWAT Model Run
        print('SWAT Model Running:')
        SWAT_Execution_Run.SWAT_model_execution(param_val_arr)

        # Delete ParallelProcessing
        shutil.rmtree(f'{swat_parallel}')
        print('Delete ParallelProcessing folder finished!')

        # Exit
        sys.exit()

    # Validation
    if cal_val_state == 'Validation':
        print(f'{cal_vars_list}-Validation:')
        # Read Parameters
        pareto_f_txt = r'Pareto-front_solutions_Streamflow.txt'
        if cal_vars_list == ['Streamflow']:
            if cal_scheme == 'Multi-site':
                pareto_f_txt = r'Pareto-front_solutions_Streamflow_Multi-site.txt'
            elif cal_scheme == 'Multi-objective':
                pareto_f_txt = r'Pareto-front_solutions_Streamflow_Multi-objective.txt'
        elif cal_vars_list == ['LAI']:
            if cal_scheme == 'Multi-objective':
                pareto_f_txt = r'Pareto-front_solutions_LAI_Multi-objective.txt'
        elif cal_vars_list == ['BIOM']:
            if cal_scheme == 'Multi-objective':
                pareto_f_txt = r'Pareto-front_solutions_BIOM_Multi-objective.txt'
        elif cal_vars_list == ['Streamflow', 'ET']:
            if cal_scheme == 'Multi-variable':
                pareto_f_txt = r'Pareto-front_solutions_Streamflow_ET.txt'
        print('pareto_f_txt:', pareto_f_txt)
        param_val_arr = np.loadtxt(f'{swat_nsga_out}\\{pareto_f_txt}',
                                   skiprows=1, usecols=range(len(swat_parameter)), dtype=float)
        print('param_val_arr:', param_val_arr.shape, '\n', param_val_arr)

        # 如果目标文件夹存在，则删除它
        if os.path.exists(f'{swat_model}\\Validation_Pareto_Sol'):
            print('Deleting:')
            shutil.rmtree(f'{swat_model}\\Validation_Pareto_Sol')
            print('Delete Validation_Pareto_Sol folder finished!')
        # Copy Folder
        pop_num = 1 if param_val_arr.ndim == 1 else param_val_arr.shape[0]
        for pop_idx in range(1, pop_num + 1):
            print('pop_idx:', pop_idx)
            shutil.copytree(f'{swat_TxtInOut}', f'{swat_model}\\Validation_Pareto_Sol\\Population_{pop_idx}')
        print('Copy to Validation_Pareto_Sol folder finished!')

        # SWAT Model Run
        print('SWAT Model Running:')
        sim_dict, mean_season_dict, eva_metrics = SWAT_Execution_Run.SWAT_model_execution(param_val_arr)

        # Plot
        print('Plotting:')
        cal_period = SWAT_Execution_Run.cal_period
        val_period = SWAT_Execution_Run.val_period
        water_budget = SWAT_Execution_Run.water_budget
        ori_SWAT_path = r'C:\NSGA\Results_Backup\NSGA.OUT_Original SWAT'
        ori_SWAT = True
        obs_date_day_val = pd.date_range(start=f'{val_period[0]}-01-01', end=f'{val_period[1]}-12-31', freq='D')
        obs_date_senson  = pd.date_range(start=f'2001-01-01', end=f'2001-12-31', freq='D')
        if cal_vars_list == ['Streamflow']:
            obs_sf_data = SWAT_Execution_Run.obs_sf_data
            # Time Series
            print('Streamflow Time Series:')
            if len(hydro_stas) == 1:
                ## 创建一个3x2的子图布局，宽高比为3:1
                fig_sf, axes_sf = plt.subplots(1, 2, figsize=(12, 3), sharex='col', gridspec_kw={'width_ratios': [3, 1]}, dpi=500)
                ## ax1_sf
                ax1_sf = axes_sf[0]
                obs,  = ax1_sf.plot(obs_date_day_val, obs_sf_data[16], linestyle='-', linewidth=0.5, color='Red', label='Observation')
                comp, = ax1_sf.plot(obs_date_day_val, sim_dict[16][2], linestyle='-', linewidth=0.5, color='Blue', label='Compromise solution')
                env   = ax1_sf.fill_between(obs_date_day_val, sim_dict[16][0], sim_dict[16][1], color='#BDBDBD', label='Envelope')
                ax1_sf.set_title('Fenkeng Station', fontsize=12)
                ax1_sf.set_xticks(obs_date_day_val[::180])
                ax1_sf.set_xticklabels(obs_date_day_val[::180], fontsize=12, rotation=20, ha='right')
                for label in ax1_sf.get_yticklabels():
                    label.set_fontsize(12)
                ax1_sf.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax1_sf.axvline(x=obs_date_day_val[obs_date_day_val.get_loc(f'{cal_period[1]}-12-31')], color='Grey', linestyle='--', linewidth=1)
                ax1_sf.text(0.40, 0.90,  # 0.43, 0.77
                            f'Calibration\n',
                            # f'NSE={eva_metrics_rch_dict[hydro_idx[2]][1][0]:.2f}\n'
                            # f'PBIAS={eva_metrics_rch_dict[hydro_idx[2]][2][0]:.2f}',
                            fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax1_sf.transAxes)
                ax1_sf.text(0.60, 0.90,  # 0.565, 0.77
                            f'Validation\n',
                            # f'NSE={eva_metrics_rch_dict[hydro_idx[2]][1][1]:.2f}\n'
                            # f'PBIAS={eva_metrics_rch_dict[hydro_idx[2]][2][1]:.2f}',
                            fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax1_sf.transAxes)

                # ax2_sf
                ax2_sf = axes_sf[1]
                ax2_sf.plot(obs_date_senson, mean_season_dict[16][0], linestyle='-', linewidth=0.5, color='Red', label='Observation')
                ax2_sf.plot(obs_date_senson, mean_season_dict[16][-1], linestyle='-', linewidth=0.5, color='Blue', label='Compromise solution')
                ax2_sf.fill_between(obs_date_senson, mean_season_dict[16][1], mean_season_dict[16][2], color='#BDBDBD',
                                    label='Envelope')
                ax2_sf.set_xticks(obs_date_senson[::60])
                ax2_sf.set_xticklabels(obs_date_senson[::60], fontsize=12, rotation=20, ha='center')
                ax2_sf.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                # ax2_sf.text(0.02, 0.79,
                #             f'KGE={eva_metrics_rch_dict[hydro_idx[2]][0][2]:.2f}\n'
                #             f'NSE={eva_metrics_rch_dict[hydro_idx[2]][1][2]:.2f}\n'
                #             f'PBIAS={eva_metrics_rch_dict[hydro_idx[2]][2][2]:.2f}',
                #             fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax2_sf.transAxes)
                axes_sf[0].set_xlim(obs_date_day_val[0], obs_date_day_val[-1])
                axes_sf[1].set_xlim(obs_date_senson[0], obs_date_senson[-1])
                fig_sf.text(0.02, 0.5, 'Streamflow (m$^3$/s)', va='center', rotation='vertical', fontsize=12)
                fig_sf.legend(handles=[obs, comp, env], loc='lower center', ncol=3, columnspacing=3,
                              bbox_to_anchor=(0.5, -0.05), fontsize=12, frameon=False)
                plt.tight_layout()
                plt.savefig(f'{swat_nsga_out}\\Streamflow_Time_Series_Stage1.jpg')
                # plt.show()

                # Flow Duration Curve (FDC)
                print('Streamflow Flow Duration Curve (FDC):')
                fig_fdc, axes_fdc = plt.subplots(1, 1, figsize=(12, 6), sharex=True, dpi=500)
                ## 遍历所有子图并进行绘图
                plot_fdc_list = []

                ## FDC
                ## 1) Sort
                obs_sf_data_sort = np.sort(np.log(obs_sf_data[16]))[::-1]
                sim_sf_data_sort = np.sort(np.log(sim_dict[16][2]))[::-1]
                ## 2) Assign rank
                sf_data_sort_rank = np.arange(1, obs_sf_data_sort.shape[0] + 1)
                ## 3) Calculate exceedence probability (P)
                sf_data_exce_prob = 100 * (sf_data_sort_rank / (obs_sf_data_sort.shape[0] + 1))
                #
                ax_fdc = axes_fdc
                obs, = ax_fdc.plot(sf_data_exce_prob, obs_sf_data_sort, linestyle='-', linewidth=1.0, color='Red', label='Observation')
                comp, = ax_fdc.plot(sf_data_exce_prob, sim_sf_data_sort, linestyle='-', linewidth=1.0, color='Blue', label='Compromise solution')
                ax_fdc.set_title('Fenkeng Station', fontsize=16)
                for label in (ax_fdc.get_xticklabels() + ax_fdc.get_yticklabels()):
                    label.set_fontsize(16)
                ax_fdc.axvline(x=2, color='Grey', linestyle='--', linewidth=1)
                ax_fdc.axvline(x=20, color='Grey', linestyle='--', linewidth=1)
                ax_fdc.axvline(x=70, color='Grey', linestyle='--', linewidth=1)
                ax_fdc.text(0.015, 0.12, 'Peak flow', ha='center', va='center', rotation='vertical', fontsize=16, transform=ax_fdc.transAxes)
                ax_fdc.text(0.12, 0.12, 'High flow', ha='center', va='center', fontsize=16, transform=ax_fdc.transAxes)
                ax_fdc.text(0.46, 0.12, 'Mid flow', ha='center', va='center', fontsize=16, transform=ax_fdc.transAxes)
                ax_fdc.text(0.86, 0.12, 'Low flow', ha='center', va='center', fontsize=16, transform=ax_fdc.transAxes)
                plot_fdc_list.append([obs, comp])
                axes_fdc.set_xlim(-0.3, 100.3)
                axes_fdc.set_xlabel('Exceedance Probability (%)', fontsize=18)
                axes_fdc.set_ylabel('Log Streamflow (m$^3$/s)', fontsize=18)
                axes_fdc.legend(handles=plot_fdc_list[0], loc='upper right', fontsize=16, frameon=False)
                plt.tight_layout()
                plt.savefig(f'{swat_nsga_out}\\Streamflow_FDC_Stage1.jpg')
                # plt.show()
            elif len(hydro_stas) == 3:
                fig, axes = plt.subplots(3, 2, figsize=(12, 6), sharex='col', gridspec_kw={'width_ratios': [3, 1]}, dpi=500)
                ## 关闭除了最后一行之外的所有子图的x轴刻度标签
                for ax in axes[:-1, :].ravel():
                    ax.tick_params(labelbottom=False)
                ## 遍历所有子图并进行绘图
                plot_list = []
                for hydro_idx in zip(range(3), ['NingDu', 'ShiCheng', 'FenKeng'], sorted(hydro_stas.values())):
                    print('hydro_idx:', hydro_idx[1])
                    # ax1
                    ax1 = axes[hydro_idx[0], 0]
                    obs,  = ax1.plot(obs_date_day_val, obs_sf_data[hydro_idx[2]], linestyle='-', linewidth=0.5, color='Red', label='Observation')
                    comp, = ax1.plot(obs_date_day_val, sim_dict[hydro_idx[2]][2], linestyle='-', linewidth=0.5, color='Blue', label='Compromise solution')
                    env   = ax1.fill_between(obs_date_day_val, sim_dict[hydro_idx[2]][0], sim_dict[hydro_idx[2]][1], color='#BDBDBD', label='Envelope')
                    ax1.set_title(f'{hydro_idx[1]} Station', fontsize=12)
                    if hydro_idx[1] == 'FenKeng':
                        ax1.set_xticks(obs_date_day_val[::180])
                        ax1.set_xticklabels(obs_date_day_val[::180], fontsize=12, rotation=20, ha='right')
                    for label in ax1.get_yticklabels():
                        label.set_fontsize(12)
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax1.axvline(x=obs_date_day_val[obs_date_day_val.get_loc(f'{cal_period[1]}-12-31')], color='Grey', linestyle='--', linewidth=1)
                    ax1.text(0.40, 0.85,
                             f'Calibration\n',
                             # f'NSE={eva_metrics[hydro_idx[2]][1][0]:.2f}\n'
                             # f'PBIAS={eva_metrics[hydro_idx[2]][2][0]:.2f}',
                             fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax1.transAxes)
                    ax1.text(0.60, 0.85,
                             f'Validation\n',
                             # f'NSE={eva_metrics[hydro_idx[2]][1][1]:.2f}\n'
                             # f'PBIAS={eva_metrics[hydro_idx[2]][2][1]:.2f}',
                             fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax1.transAxes)
                    # ax2
                    ax2 = axes[hydro_idx[0], 1]
                    ax2.plot(obs_date_senson, mean_season_dict[hydro_idx[2]][0], linestyle='-', linewidth=0.5, color='Red', label='Observation')
                    ax2.plot(obs_date_senson, mean_season_dict[hydro_idx[2]][-1], linestyle='-', linewidth=0.5, color='Blue', label='Compromise solution')
                    ax2.fill_between(obs_date_senson, mean_season_dict[hydro_idx[2]][1], mean_season_dict[hydro_idx[2]][2], color='#BDBDBD', label='Envelope')
                    if hydro_idx[1] == 'FenKeng':
                        ax2.set_xticks(obs_date_senson[::60])
                        ax2.set_xticklabels(obs_date_senson[::60], fontsize=12, rotation=20, ha='center')
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                    # ax2.text(0.02, 0.79,
                    #          f'KGE={eva_metrics[hydro_idx[2]][0][2]:.2f}\n'
                    #          f'NSE={eva_metrics[hydro_idx[2]][1][2]:.2f}\n'
                    #          f'PBIAS={eva_metrics[hydro_idx[2]][2][2]:.2f}',
                    #          fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax2.transAxes)
                    plot_list.append([obs, comp, env])
                axes[0, 0].set_xlim(obs_date_day_val[0], obs_date_day_val[-1])
                axes[0, 1].set_xlim(obs_date_senson[0], obs_date_senson[-1])
                fig.text(0.02, 0.5, 'Streamflow (m$^3$/s)', va='center', rotation='vertical', fontsize=12)
                fig.legend(handles=plot_list[0], loc='lower center', ncol=3, columnspacing=3,
                           bbox_to_anchor=(0.5, -0.025), fontsize=12, frameon=False)
                plt.tight_layout()
                plt.savefig(f'{swat_nsga_out}\\Streamflow_Time_Series_Stage1.jpg')
                # plt.show()
                print('\n')

                # Flow Duration Curve (FDC)
                print('Streamflow Flow Duration Curve (FDC):')
                fig_fdc, axes_fdc = plt.subplots(3, 1, figsize=(12, 16), sharex=True, dpi=500)
                for ax in axes_fdc[:-1]:
                    ax.tick_params(labelbottom=False)
                ## 遍历所有子图并进行绘图
                plot_fdc_list = []
                for hydro_idx in zip(range(3), ['NingDu', 'ShiCheng', 'FenKeng'], sorted(hydro_stas.values())):
                    print('hydro_idx:', hydro_idx[1])
                    ## FDC
                    ## 1) Sort log
                    obs_sf_data_sort = np.sort(np.log(obs_sf_data[hydro_idx[2]]))[::-1]
                    sim_sf_data_sort = np.sort(np.log(sim_dict[hydro_idx[2]][2]))[::-1]
                    ## 2) Assign rank
                    sf_data_sort_rank = np.arange(1, obs_sf_data_sort.shape[0] + 1)
                    ## 3) Calculate exceedence probability (P)
                    sf_data_exce_prob = 100 * (sf_data_sort_rank / (obs_sf_data_sort.shape[0] + 1))
                    #
                    ax_fdc = axes_fdc[hydro_idx[0]]
                    obs,  = ax_fdc.plot(sf_data_exce_prob, obs_sf_data_sort, linestyle='-', linewidth=1.0, color='Red', label='Observation')
                    comp, = ax_fdc.plot(sf_data_exce_prob, sim_sf_data_sort, linestyle='-', linewidth=1.0, color='Blue', label='Compromise solution')
                    ax_fdc.set_title(f'{hydro_idx[1]} Station', fontsize=16)
                    for label in (ax_fdc.get_xticklabels() + ax_fdc.get_yticklabels()):
                        label.set_fontsize(16)
                    ax_fdc.axvline(x=2, color='Grey', linestyle='--', linewidth=1)
                    ax_fdc.axvline(x=20, color='Grey', linestyle='--', linewidth=1)
                    ax_fdc.axvline(x=70, color='Grey', linestyle='--', linewidth=1)
                    if hydro_idx[1] == 'NingDu':
                        ax_fdc.text(0.015, 0.12, 'Peak flow', ha='center', va='center', rotation='vertical', fontsize=16, transform=ax_fdc.transAxes)
                        ax_fdc.text(0.12, 0.12, 'High flow', ha='center', va='center', fontsize=16, transform=ax_fdc.transAxes)
                        ax_fdc.text(0.46, 0.12, 'Mid flow', ha='center', va='center', fontsize=16, transform=ax_fdc.transAxes)
                        ax_fdc.text(0.86, 0.12, 'Low flow', ha='center', va='center', fontsize=16, transform=ax_fdc.transAxes)
                    plot_fdc_list.append([obs, comp])
                axes_fdc[2].set_xlim(-0.3, 100.3)
                axes_fdc[2].set_xlabel('Exceedance Probability (%)', fontsize=18)
                axes_fdc[1].set_ylabel('Log Streamflow (m$^3$/s)', fontsize=18)
                axes_fdc[0].legend(handles=plot_fdc_list[0], loc='upper right', fontsize=16, frameon=False)
                plt.tight_layout()
                plt.savefig(f'{swat_nsga_out}\\Streamflow_FDC_Stage1.jpg')
                # plt.show()
        elif cal_vars_list == ['LAI']:
            LAI_obs_data_dict_area_w = SWAT_Execution_Run.LAI_obs_data_dict_area_w
            # Original SWAT Data
            ori_SWAT_data, ori_SWAT_mean_season = None, None
            if ori_SWAT == True:
                # ori_SWAT_data = np.loadtxt(fname=f'{ori_SWAT_path}\\HRU_Default.txt', skiprows=1, usecols=6)[:len(obs_date_day_val)]
                ori_SWAT_data = np.loadtxt(fname=f'{ori_SWAT_path}\\HRU_Default_Stage2.txt', skiprows=1, usecols=6)
                print('ori_SWAT_data:', ori_SWAT_data.shape, ori_SWAT_data)
                ori_SWAT_mean_season = SWAT_Execution_Run.mean_seasonal_cycle(obs_date_day_val, ori_SWAT_data)

            # 创建一个3x2的子图布局，宽高比为3:1
            fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharex='col', gridspec_kw={'width_ratios': [3, 1]}, dpi=500)
            # ax1 (Time Series)
            ax1 = axes[0]
            obs,  = ax1.plot(obs_date_day_val, LAI_obs_data_dict_area_w, linestyle='-', linewidth=0.5, color='Red', label='Observation')
            ori,  = ax1.plot(obs_date_day_val, ori_SWAT_data, linestyle='-', linewidth=0.5, color='Green', label='Original SWAT')
            comp, = ax1.plot(obs_date_day_val, sim_dict[2], linestyle='-', linewidth=0.5, color='Blue', label='Improved SWAT(Compromise solution)')
            env = ax1.fill_between(obs_date_day_val, sim_dict[0], sim_dict[1], color='#BDBDBD', label='Envelope')
            ax1.set_xticks(obs_date_day_val[::180])
            ax1.set_xticklabels(obs_date_day_val[::180], fontsize=12, rotation=20, ha='right')
            for label in ax1.get_yticklabels():
                label.set_fontsize(12)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.axvline(x=obs_date_day_val[obs_date_day_val.get_loc(f'{cal_period[1]}-12-31')], color='Grey', linestyle='--', linewidth=1)
            # ax1.text(0.44, 0.85, f'Calibration\nNSE={eva_metrics[1][0]:.2f}\nPBIAS={eva_metrics[2][0]:.2f}',
            #          fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax1.transAxes)
            # ax1.text(0.565, 0.85, f'Validation\nNSE={eva_metrics[1][1]:.2f}\nPBIAS={eva_metrics[2][1]:.2f}',
            #          fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax1.transAxes)

            # ax2 (Mean Season)
            ax2 = axes[1]
            ax2.plot(obs_date_senson, mean_season_dict[0], linestyle='-', linewidth=0.5, color='Red', label='Observation')
            ax2.plot(obs_date_senson, ori_SWAT_mean_season, linestyle='-', linewidth=0.5, color='Green', label='Original SWAT')
            ax2.plot(obs_date_senson, mean_season_dict[-1], linestyle='-', linewidth=0.5, color='Blue', label='Improved SWAT(Compromise solution)')
            ax2.fill_between(obs_date_senson, mean_season_dict[1], mean_season_dict[2], color='#BDBDBD', label='Envelope')
            ax2.set_xticks(obs_date_senson[::60])
            ax2.set_xticklabels(obs_date_senson[::60], fontsize=12, rotation=20, ha='center')
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            # ax2.text(0.02, 0.88, f'NSE={eva_metrics[1][2]:.2f}\nPBIAS={eva_metrics[2][2]:.2f}',
            #          fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax2.transAxes)
            axes[0].set_xlim(obs_date_day_val[0], obs_date_day_val[-1])
            axes[1].set_xlim(obs_date_senson[0], obs_date_senson[-1])
            fig.text(0.02, 0.5, 'LAI (m$^2$/m$^2$)', va='center', rotation='vertical', fontsize=12)
            fig.legend(handles=[obs, ori, comp, env], loc='lower center', ncol=4, columnspacing=3,
                       bbox_to_anchor=(0.5, -0.05), fontsize=12, frameon=False)
            plt.tight_layout()
            plt.savefig(f'{swat_nsga_out}\\LAI_Time_Series_Stage1.jpg')
            plt.show()
        elif cal_vars_list == ['BIOM']:
            BIOM_obs_data_dict_area_w_FRST = SWAT_Execution_Run.BIOM_obs_data_dict_area_w_FRST
            BIOM_obs_data_dict_area_w_FRSE = SWAT_Execution_Run.BIOM_obs_data_dict_area_w_FRSE
            # Original SWAT Data
            ori_FRST, ori_FRST_std, ori_FRSE, ori_FRSE_std = None, None, None, None
            if ori_SWAT == True:
                # ori_SWAT_data = np.loadtxt(fname=f'{ori_SWAT_path}\\BIOM_Default.txt', skiprows=1, usecols=(1, 2))[:9]
                ori_SWAT_data = np.loadtxt(fname=f'{ori_SWAT_path}\\BIOM_Default_Stage2.txt', skiprows=1, usecols=(1, 2))
                print('ori_SWAT_data:', ori_SWAT_data.shape, ori_SWAT_data)
                ori_FRST = ori_SWAT_data[:, 0]
                ori_FRSE = ori_SWAT_data[:, 1]
                print('ori_FRST:', ori_FRST.shape, ori_FRST)
                print('ori_FRSE:', ori_FRSE.shape, ori_FRSE)
                ori_FRST_std = np.std(ori_FRST)
                ori_FRSE_std = np.std(ori_FRSE)
            BIOM_sim_arr_FRST_comp, BIOM_sim_arr_FRSE_comp = sim_dict
            (obs_std_FRST, obs_std_FRSE), (sim_std_FRST, sim_std_FRSE) = mean_season_dict
            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 12), dpi=300)

            # 定义误差棒的样式
            error_attributes = {
                'elinewidth': 1.5,  # 增加误差线的宽度
                'ecolor': 'black',  # 设置误差线的颜色为黑色
                'capsize': 4,       # 增加误差线帽的大小
                'capthick': 1.5,    # 增加误差线帽的厚度
            }

            # 绘制柱状图及误差棒
            group = ['Observation', 'Original SWAT', 'Improved SWAT']
            # x轴的位置（用于柱状图的中心位置）
            x = np.arange(len(group))
            # 柱状图的宽度（每个组占据的宽度）
            width = 0.2
            bar_FRST = ax.bar(x - width / 2, [np.mean(BIOM_obs_data_dict_area_w_FRST), np.mean(ori_FRST), np.mean(BIOM_sim_arr_FRST_comp)],
                              yerr=[obs_std_FRST, ori_FRST_std, sim_std_FRST], error_kw=error_attributes, color='#318F0F', width=width, label='FRST')
            bar_FRSE = ax.bar(x + width / 2, [np.mean(BIOM_obs_data_dict_area_w_FRSE), np.mean(ori_FRSE), np.mean(BIOM_sim_arr_FRSE_comp)],
                              yerr=[obs_std_FRSE, ori_FRSE_std, sim_std_FRSE], error_kw=error_attributes, color='#9CD00D', width=width, label='FRSE')
            ax.set_ylabel('Total Biomass (t/ha)', fontsize=16)
            ax.set_xticks(x, labels=group, fontsize=16)
            ax.tick_params(axis='y', labelsize=16)
            plt.legend(loc='upper right', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'{swat_nsga_out}\\BIOM_Bar_Stage2.jpg')
            plt.show()
        elif cal_vars_list == ['Streamflow', 'ET']:
            if water_budget:
                rch_sta_sim_dict, ET_sim_arr = sim_dict
                mean_season_rch_dict, mean_season_ET_dict, [mean_season_WB_FRST_dict, mean_season_WB_FRSE_dict] = mean_season_dict
                eva_metrics_rch_dict, eva_metrics_ET_dict = eva_metrics
                #
                mean_season_FRST_ET_comp, mean_season_FRST_PRECIP_comp, mean_season_FRST_SURQ_comp, mean_season_FRST_LATQ_comp, mean_season_FRST_GWQ_comp = (
                    mean_season_WB_FRST_dict)
                mean_season_FRSE_ET_comp, mean_season_FRSE_PRECIP_comp, mean_season_FRSE_SURQ_comp, mean_season_FRSE_LATQ_comp, mean_season_FRSE_GWQ_comp = (
                    mean_season_WB_FRSE_dict)
                #
                mean_season_FRST_comp_sum = np.sum([mean_season_FRST_ET_comp, mean_season_FRST_SURQ_comp, mean_season_FRST_LATQ_comp, mean_season_FRST_GWQ_comp])
                mean_season_FRSE_comp_sum = np.sum([mean_season_FRSE_ET_comp, mean_season_FRSE_SURQ_comp, mean_season_FRSE_LATQ_comp, mean_season_FRSE_GWQ_comp])
                #
                mean_season_FRST_ET_comp_pct   = 100 * np.sum(mean_season_FRST_ET_comp) / mean_season_FRST_comp_sum
                mean_season_FRST_SURQ_comp_pct = 100 * np.sum(mean_season_FRST_SURQ_comp) / mean_season_FRST_comp_sum
                mean_season_FRST_LATQ_comp_pct = 100 * np.sum(mean_season_FRST_LATQ_comp) / mean_season_FRST_comp_sum
                mean_season_FRST_GWQ_comp_pct  = 100 * np.sum(mean_season_FRST_GWQ_comp) / mean_season_FRST_comp_sum
                FRST_pct = [mean_season_FRST_GWQ_comp_pct, mean_season_FRST_LATQ_comp_pct, mean_season_FRST_SURQ_comp_pct, mean_season_FRST_ET_comp_pct]
                #
                mean_season_FRSE_ET_comp_pct   = 100 * np.sum(mean_season_FRSE_ET_comp) / mean_season_FRSE_comp_sum
                mean_season_FRSE_SURQ_comp_pct = 100 * np.sum(mean_season_FRSE_SURQ_comp) / mean_season_FRSE_comp_sum
                mean_season_FRSE_LATQ_comp_pct = 100 * np.sum(mean_season_FRSE_LATQ_comp) / mean_season_FRSE_comp_sum
                mean_season_FRSE_GWQ_comp_pct  = 100 * np.sum(mean_season_FRSE_GWQ_comp) / mean_season_FRSE_comp_sum
                FRSE_pct = [mean_season_FRSE_GWQ_comp_pct, mean_season_FRSE_LATQ_comp_pct, mean_season_FRSE_SURQ_comp_pct, mean_season_FRSE_ET_comp_pct]

                # Annual Water Budget Plot
                print('Annual Water Budget:')
                ## FRST
                fig_stack, axes_stack = plt.subplots(2, 1, sharex=True, figsize=(12, 12), dpi=500)
                stack_FRST = [mean_season_FRST_GWQ_comp, mean_season_FRST_LATQ_comp, mean_season_FRST_SURQ_comp, mean_season_FRST_ET_comp]
                stack_FRSE = [mean_season_FRSE_GWQ_comp, mean_season_FRSE_LATQ_comp, mean_season_FRSE_SURQ_comp, mean_season_FRSE_ET_comp]
                mean_season_FRST_max = np.max(np.sum(stack_FRST, axis=0))
                mean_season_FRSE_max = np.max(np.sum(stack_FRSE, axis=0))
                colors = ['#934BC9', '#04AFF2', '#FF7F74', '#00AF50']
                labels = ['GWQ', 'LATQ', 'SURQ', 'ET']
                plot_list_stack = []
                for stack_idx in zip(range(2), ['FRST', 'FRSE'], [stack_FRST, stack_FRSE], [mean_season_FRST_PRECIP_comp, mean_season_FRSE_PRECIP_comp],
                                     [mean_season_FRST_max, mean_season_FRSE_max], [FRST_pct, FRSE_pct]):
                    print(stack_idx[1])
                    # Stackplot
                    ax_stack = axes_stack[stack_idx[0]]
                    stack_plot = ax_stack.stackplot(obs_date_senson, stack_idx[2], colors=colors, labels=labels)
                    ax_stack.set_title(stack_idx[1], fontsize=16)
                    ax_stack.set_xticks(obs_date_senson[::30])
                    ax_stack.set_xticklabels(obs_date_senson[::30], fontsize=14, ha='center')
                    ax_stack.set_xlim(obs_date_senson[0], obs_date_senson[-1])
                    ax_stack.set_ylim(0, stack_idx[4] * 2)
                    ax_stack.tick_params(axis='y', labelsize=14)
                    ax_stack.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                    if stack_idx[0] == 0:
                        ax_stack.tick_params(labelbottom=False)
                    # Bar
                    ax_bar = ax_stack.twinx()
                    bar_plot = ax_bar.bar(obs_date_senson, stack_idx[3], color='gray', label='Precipitation')
                    ax_bar.set_ylim(0, np.max(stack_idx[3]) * 2)
                    ax_bar.tick_params(axis='y', labelsize=14)
                    ax_bar.invert_yaxis()
                    # Pie
                    if stack_idx[0] == 0:
                        ax_pie = fig_stack.add_axes([0.65, 0.65, 0.25, 0.25])
                        ax_pie.pie(stack_idx[5], colors=colors, autopct='%1.1f%%', startangle=-90)
                    elif stack_idx[0] == 1:
                        ax_pie = fig_stack.add_axes([0.65, 0.15, 0.25, 0.25])
                        ax_pie.pie(stack_idx[5], colors=colors, autopct='%1.1f%%', startangle=-90)
                    plot_list_stack.append(stack_plot)
                fig_stack.text(0, 0.5, 'Mean Daily Water Budget (mm)', va='center', rotation='vertical', fontsize=14)
                fig_stack.text(0.985, 0.5, 'Precipitation (mm)', va='center', rotation=270, fontsize=14)
                handles, labels = axes_stack[0].get_legend_handles_labels()
                fig_stack.legend(handles[::-1], labels[::-1], loc='lower center', ncol=5, columnspacing=3,
                                 bbox_to_anchor=(0.5, -0.017), fontsize=14, frameon=False)
                plt.tight_layout()
                plt.savefig(f'{swat_nsga_out}\\Annual_Water_Budget_Plot_Stage1.jpg')
                # plt.show()

                # Streamflow Time Series
                print('Streamflow Time Series:')
                obs_sf_data = SWAT_Execution_Run.obs_sf_data
                ## 创建一个3x2的子图布局，宽高比为3:1
                fig_sf, axes_sf = plt.subplots(3, 2, figsize=(12, 6), sharex='col', gridspec_kw={'width_ratios': [3, 1]}, dpi=500)
                ## 关闭除了最后一行之外的所有子图的x轴刻度标签
                for ax_sf in axes_sf[:-1, :].ravel():
                    ax_sf.tick_params(labelbottom=False)
                ## 遍历所有子图并进行绘图
                plot_list_sf = []
                for hydro_idx in zip(range(3), ['NingDu', 'ShiCheng', 'FenKeng'], sorted(hydro_stas.values())):
                    print('hydro_idx:', hydro_idx[1])
                    # ax1_sf
                    ax1_sf = axes_sf[hydro_idx[0], 0]
                    obs_sf,  = ax1_sf.plot(obs_date_day_val, obs_sf_data[hydro_idx[2]], linestyle='-', linewidth=0.5, color='Red',
                                           label='Observation')
                    comp_sf, = ax1_sf.plot(obs_date_day_val, rch_sta_sim_dict[hydro_idx[2]][2], linestyle='-', linewidth=0.5,
                                           color='Blue', label='Compromise solution')
                    env_sf = ax1_sf.fill_between(obs_date_day_val, rch_sta_sim_dict[hydro_idx[2]][0], rch_sta_sim_dict[hydro_idx[2]][1],
                                                 color='#BDBDBD', label='Envelope')
                    ax1_sf.set_title(f'{hydro_idx[1]} Station', fontsize=12)
                    if hydro_idx[1] == 'FenKeng':
                        ax1_sf.set_xticks(obs_date_day_val[::180])
                        ax1_sf.set_xticklabels(obs_date_day_val[::180], fontsize=12, rotation=20, ha='right')
                    for label in ax1_sf.get_yticklabels():
                        label.set_fontsize(12)
                    ax1_sf.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax1_sf.axvline(x=obs_date_day_val[obs_date_day_val.get_loc(f'{cal_period[1]}-12-31')], color='Grey', linestyle='--', linewidth=1)
                    ax1_sf.text(0.43, 0.77,
                                f'Calibration\nKGE={eva_metrics_rch_dict[hydro_idx[2]][0][0]:.2f}\n'
                                f'NSE={eva_metrics_rch_dict[hydro_idx[2]][1][0]:.2f}\n'
                                f'PBIAS={eva_metrics_rch_dict[hydro_idx[2]][2][0]:.2f}',
                                fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax1_sf.transAxes)
                    ax1_sf.text(0.565, 0.77,
                                f'Validation\nKGE={eva_metrics_rch_dict[hydro_idx[2]][0][1]:.2f}\n'
                                f'NSE={eva_metrics_rch_dict[hydro_idx[2]][1][1]:.2f}\n'
                                f'PBIAS={eva_metrics_rch_dict[hydro_idx[2]][2][1]:.2f}',
                                fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax1_sf.transAxes)
                    # ax2_sf
                    ax2_sf = axes_sf[hydro_idx[0], 1]
                    ax2_sf.plot(obs_date_senson, mean_season_rch_dict[hydro_idx[2]][0], linestyle='-', linewidth=0.5, color='Red', label='Observation')
                    ax2_sf.plot(obs_date_senson, mean_season_rch_dict[hydro_idx[2]][-1], linestyle='-', linewidth=0.5, color='Blue', label='Compromise solution')
                    ax2_sf.fill_between(obs_date_senson, mean_season_rch_dict[hydro_idx[2]][1], mean_season_rch_dict[hydro_idx[2]][2], color='#BDBDBD',
                                        label='Envelope')
                    if hydro_idx[1] == 'FenKeng':
                        ax2_sf.set_xticks(obs_date_senson[::60])
                        ax2_sf.set_xticklabels(obs_date_senson[::60], fontsize=12, rotation=20, ha='center')
                    ax2_sf.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                    ax2_sf.text(0.02, 0.79,
                                f'KGE={eva_metrics_rch_dict[hydro_idx[2]][0][2]:.2f}\n'
                                f'NSE={eva_metrics_rch_dict[hydro_idx[2]][1][2]:.2f}\n'
                                f'PBIAS={eva_metrics_rch_dict[hydro_idx[2]][2][2]:.2f}',
                                fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax2_sf.transAxes)
                    plot_list_sf.append([obs_sf, comp_sf, env_sf])
                axes_sf[0, 0].set_xlim(obs_date_day_val[0], obs_date_day_val[-1])
                axes_sf[0, 1].set_xlim(obs_date_senson[0], obs_date_senson[-1])
                fig_sf.text(0.02, 0.5, 'Streamflow (m$^3$/s)', va='center', rotation='vertical', fontsize=12)
                fig_sf.legend(handles=plot_list_sf[0], loc='lower center', ncol=3, columnspacing=3,
                              bbox_to_anchor=(0.5, -0.025), fontsize=12, frameon=False)
                plt.tight_layout()
                plt.savefig(f'{swat_nsga_out}\\Streamflow_Time_Series_Stage1.jpg')
                # plt.show()

                # ET Time Series
                print('ET Time Series:')
                ET_obs_data_dict_area_w = SWAT_Execution_Run.ET_obs_data_dict_area_w
                ## 创建一个3x2的子图布局，宽高比为3:1
                fig_et, axes_et = plt.subplots(1, 2, figsize=(12, 3), sharex='col', gridspec_kw={'width_ratios': [3, 1]}, dpi=500)
                ## ax1_et
                ax1_et = axes_et[0]
                obs_et,  = ax1_et.plot(obs_date_day_val, ET_obs_data_dict_area_w, linestyle='-', linewidth=0.5, color='Red', label='Observation')
                comp_et, = ax1_et.plot(obs_date_day_val, ET_sim_arr[2], linestyle='-', linewidth=0.5, color='Blue', label='Compromise solution')
                env_et = ax1_et.fill_between(obs_date_day_val, ET_sim_arr[0], ET_sim_arr[1], color='#BDBDBD', label='Envelope')
                ax1_et.set_xticks(obs_date_day_val[::180])
                ax1_et.set_xticklabels(obs_date_day_val[::180], fontsize=12, rotation=20, ha='right')
                for label in ax1_et.get_yticklabels():
                    label.set_fontsize(12)
                ax1_et.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax1_et.axvline(x=obs_date_day_val[obs_date_day_val.get_loc(f'{cal_period[1]}-12-31')], color='Grey', linestyle='--', linewidth=1)
                ax1_et.text(0.43, 0.85,
                            f'Calibration\nKGE={eva_metrics_ET_dict[0][0]:.2f}\nNSE={eva_metrics_ET_dict[1][0]:.2f}\nPBIAS={eva_metrics_ET_dict[2][0]:.2f}',
                            fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax1_et.transAxes)
                ax1_et.text(0.565, 0.85,
                            f'Validation\nKGE={eva_metrics_ET_dict[0][1]:.2f}\nNSE={eva_metrics_ET_dict[1][1]:.2f}\nPBIAS={eva_metrics_ET_dict[2][1]:.2f}',
                            fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax1_et.transAxes)
                ## ax2_et
                ax2_et = axes_et[1]
                ax2_et.plot(obs_date_senson, mean_season_ET_dict[0], linestyle='-', linewidth=0.5, color='Red', label='Observation')
                ax2_et.plot(obs_date_senson, mean_season_ET_dict[-1], linestyle='-', linewidth=0.5, color='Blue', label='Compromise solution')
                ax2_et.fill_between(obs_date_senson, mean_season_ET_dict[1], mean_season_ET_dict[2], color='#BDBDBD', label='Envelope')
                ax2_et.set_xticks(obs_date_senson[::60])
                ax2_et.set_xticklabels(obs_date_senson[::60], fontsize=12, rotation=20, ha='center')
                ax2_et.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax2_et.text(0.02, 0.88, f'KGE={eva_metrics_ET_dict[0][2]:.2f}\nNSE={eva_metrics_ET_dict[1][2]:.2f}\nPBIAS={eva_metrics_ET_dict[2][2]:.2f}',
                            fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax2_et.transAxes)
                axes_et[0].set_xlim(obs_date_day_val[0], obs_date_day_val[-1])
                axes_et[1].set_xlim(obs_date_senson[0], obs_date_senson[-1])
                fig_et.text(0.02, 0.5, 'ET (mm)', va='center', rotation='vertical', fontsize=12)
                fig_et.legend(handles=[obs_et, comp_et, env_et], loc='lower center', ncol=3, columnspacing=3,
                              bbox_to_anchor=(0.5, -0.05), fontsize=12, frameon=False)
                plt.tight_layout()
                plt.savefig(f'{swat_nsga_out}\\ET_Time_Series_Stage1.jpg')
                # plt.show()

                # Flow Duration Curve (FDC)
                print('Streamflow Flow Duration Curve (FDC):')
                fig_fdc, axes_fdc = plt.subplots(3, 1, figsize=(12, 16), sharex=True, dpi=500)
                for ax in axes_fdc[:-1]:
                    ax.tick_params(labelbottom=False)
                ## 遍历所有子图并进行绘图
                plot_fdc_list = []
                for hydro_idx in zip(range(3), ['NingDu', 'ShiCheng', 'FenKeng'], sorted(hydro_stas.values())):
                    print('hydro_idx:', hydro_idx[1])
                    ## FDC
                    ## 1) Sort
                    obs_sf_data_sort = np.sort(np.log(obs_sf_data[hydro_idx[2]]))[::-1]
                    sim_sf_data_sort = np.sort(np.log(rch_sta_sim_dict[hydro_idx[2]][2]))[::-1]
                    ## 2) Assign rank
                    sf_data_sort_rank = np.arange(1, obs_sf_data_sort.shape[0] + 1)
                    ## 3) Calculate exceedence probability (P)
                    sf_data_exce_prob = 100 * (sf_data_sort_rank / (obs_sf_data_sort.shape[0] + 1))
                    #
                    ax_fdc = axes_fdc[hydro_idx[0]]
                    obs,  = ax_fdc.plot(sf_data_exce_prob, obs_sf_data_sort, linestyle='-', linewidth=1.0, color='Red', label='Observation')
                    comp, = ax_fdc.plot(sf_data_exce_prob, sim_sf_data_sort, linestyle='-', linewidth=1.0, color='Blue', label='Compromise solution')
                    ax_fdc.set_title(f'{hydro_idx[1]} Station', fontsize=16)
                    for label in (ax_fdc.get_xticklabels() + ax_fdc.get_yticklabels()):
                        label.set_fontsize(16)
                    ax_fdc.axvline(x=2, color='Grey', linestyle='--', linewidth=1)
                    ax_fdc.axvline(x=20, color='Grey', linestyle='--', linewidth=1)
                    ax_fdc.axvline(x=70, color='Grey', linestyle='--', linewidth=1)
                    if hydro_idx[1] == 'NingDu':
                        ax_fdc.text(0.015, 0.12, 'Peak flow', ha='center', va='center', rotation='vertical', fontsize=16, transform=ax_fdc.transAxes)
                        ax_fdc.text(0.12, 0.12, 'High flow', ha='center', va='center', fontsize=16, transform=ax_fdc.transAxes)
                        ax_fdc.text(0.46, 0.12, 'Mid flow', ha='center', va='center', fontsize=16, transform=ax_fdc.transAxes)
                        ax_fdc.text(0.86, 0.12, 'Low flow', ha='center', va='center', fontsize=16, transform=ax_fdc.transAxes)
                    plot_fdc_list.append([obs, comp])
                axes_fdc[2].set_xlim(-0.3, 100.3)
                axes_fdc[2].set_xlabel('Exceedance Probability (%)', fontsize=18)
                axes_fdc[1].set_ylabel('Log Streamflow (m$^3$/s)', fontsize=18)
                axes_fdc[0].legend(handles=plot_fdc_list[0], loc='upper right', fontsize=16, frameon=False)
                plt.tight_layout()
                plt.savefig(f'{swat_nsga_out}\\Streamflow_FDC_Stage1.jpg')
                # plt.show()
            else:
                rch_sta_sim_dict, ET_sim_arr = sim_dict
                mean_season_rch_dict, mean_season_ET_dict = mean_season_dict
                eva_metrics_rch_dict, eva_metrics_ET_dict = eva_metrics

                # Streamflow Time Series
                print('Streamflow Time Series:')
                obs_sf_data = SWAT_Execution_Run.obs_sf_data
                ## 创建一个3x2的子图布局，宽高比为3:1
                fig_sf, axes_sf = plt.subplots(3, 2, figsize=(12, 6), sharex='col', gridspec_kw={'width_ratios': [3, 1]}, dpi=500)
                ## 关闭除了最后一行之外的所有子图的x轴刻度标签
                for ax_sf in axes_sf[:-1, :].ravel():
                    ax_sf.tick_params(labelbottom=False)
                ## 遍历所有子图并进行绘图
                plot_list_sf = []
                for hydro_idx in zip(range(3), ['NingDu', 'ShiCheng', 'FenKeng'], sorted(hydro_stas.values())):
                    print('hydro_idx:', hydro_idx[1])
                    # ax1_sf
                    ax1_sf = axes_sf[hydro_idx[0], 0]
                    obs_sf,  = ax1_sf.plot(obs_date_day_val, obs_sf_data[hydro_idx[2]], linestyle='-', linewidth=0.5, color='Red',
                                           label='Observation')
                    comp_sf, = ax1_sf.plot(obs_date_day_val, rch_sta_sim_dict[hydro_idx[2]][2], linestyle='-', linewidth=0.5,
                                           color='Blue', label='Compromise solution')
                    env_sf = ax1_sf.fill_between(obs_date_day_val, rch_sta_sim_dict[hydro_idx[2]][0], rch_sta_sim_dict[hydro_idx[2]][1],
                                                 color='#BDBDBD', label='Envelope')
                    ax1_sf.set_title(f'{hydro_idx[1]} Station', fontsize=12)
                    if hydro_idx[1] == 'FenKeng':
                        ax1_sf.set_xticks(obs_date_day_val[::180])
                        ax1_sf.set_xticklabels(obs_date_day_val[::180], fontsize=12, rotation=20, ha='right')
                    for label in ax1_sf.get_yticklabels():
                        label.set_fontsize(12)
                    ax1_sf.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax1_sf.axvline(x=obs_date_day_val[obs_date_day_val.get_loc(f'{cal_period[1]}-12-31')], color='Grey', linestyle='--', linewidth=1)
                    ax1_sf.text(0.40, 0.85,  # 0.43, 0.77
                                f'Calibration\n',
                                # f'NSE={eva_metrics_rch_dict[hydro_idx[2]][1][0]:.2f}\n'
                                # f'PBIAS={eva_metrics_rch_dict[hydro_idx[2]][2][0]:.2f}',
                                fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax1_sf.transAxes)
                    ax1_sf.text(0.60, 0.85,  # 0.565, 0.77
                                f'Validation\n',
                                # f'NSE={eva_metrics_rch_dict[hydro_idx[2]][1][1]:.2f}\n'
                                # f'PBIAS={eva_metrics_rch_dict[hydro_idx[2]][2][1]:.2f}',
                                fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax1_sf.transAxes)
                    # ax2_sf
                    ax2_sf = axes_sf[hydro_idx[0], 1]
                    ax2_sf.plot(obs_date_senson, mean_season_rch_dict[hydro_idx[2]][0], linestyle='-', linewidth=0.5, color='Red', label='Observation')
                    ax2_sf.plot(obs_date_senson, mean_season_rch_dict[hydro_idx[2]][-1], linestyle='-', linewidth=0.5, color='Blue', label='Compromise solution')
                    ax2_sf.fill_between(obs_date_senson, mean_season_rch_dict[hydro_idx[2]][1], mean_season_rch_dict[hydro_idx[2]][2], color='#BDBDBD',
                                        label='Envelope')
                    if hydro_idx[1] == 'FenKeng':
                        ax2_sf.set_xticks(obs_date_senson[::60])
                        ax2_sf.set_xticklabels(obs_date_senson[::60], fontsize=12, rotation=20, ha='center')
                    ax2_sf.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                    # ax2_sf.text(0.02, 0.79,
                    #             f'KGE={eva_metrics_rch_dict[hydro_idx[2]][0][2]:.2f}\n'
                    #             f'NSE={eva_metrics_rch_dict[hydro_idx[2]][1][2]:.2f}\n'
                    #             f'PBIAS={eva_metrics_rch_dict[hydro_idx[2]][2][2]:.2f}',
                    #             fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax2_sf.transAxes)
                    plot_list_sf.append([obs_sf, comp_sf, env_sf])
                axes_sf[0, 0].set_xlim(obs_date_day_val[0], obs_date_day_val[-1])
                axes_sf[0, 1].set_xlim(obs_date_senson[0], obs_date_senson[-1])
                fig_sf.text(0.02, 0.5, 'Streamflow (m$^3$/s)', va='center', rotation='vertical', fontsize=12)
                fig_sf.legend(handles=plot_list_sf[0], loc='lower center', ncol=3, columnspacing=3,
                              bbox_to_anchor=(0.5, -0.025), fontsize=12, frameon=False)
                plt.tight_layout()
                plt.savefig(f'{swat_nsga_out}\\Streamflow_Time_Series_Stage1.jpg')
                # plt.show()

                # ET Time Series
                print('ET Time Series:')
                ET_obs_data_dict_area_w = SWAT_Execution_Run.ET_obs_data_dict_area_w
                ## 创建一个3x2的子图布局，宽高比为3:1
                fig_et, axes_et = plt.subplots(1, 2, figsize=(12, 3), sharex='col', gridspec_kw={'width_ratios': [3, 1]}, dpi=500)
                ## ax1_et
                ax1_et = axes_et[0]
                obs_et,  = ax1_et.plot(obs_date_day_val, ET_obs_data_dict_area_w, linestyle='-', linewidth=0.5, color='Red', label='Observation')
                comp_et, = ax1_et.plot(obs_date_day_val, ET_sim_arr[2], linestyle='-', linewidth=0.5, color='Blue', label='Compromise solution')
                env_et = ax1_et.fill_between(obs_date_day_val, ET_sim_arr[0], ET_sim_arr[1], color='#BDBDBD', label='Envelope')
                ax1_et.set_xticks(obs_date_day_val[::180])
                ax1_et.set_xticklabels(obs_date_day_val[::180], fontsize=12, rotation=20, ha='right')
                for label in ax1_et.get_yticklabels():
                    label.set_fontsize(12)
                ax1_et.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax1_et.axvline(x=obs_date_day_val[obs_date_day_val.get_loc(f'{cal_period[1]}-12-31')], color='Grey', linestyle='--', linewidth=1)
                ax1_et.text(0.40, 0.95,  # 0.44, 0.85
                            f'Calibration',
                            fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax1_et.transAxes)
                # ax1_et.text(0.44, 0.85,
                #             f'Calibration\nKGE={eva_metrics_ET_dict[0][0]:.2f}\nNSE={eva_metrics_ET_dict[1][0]:.2f}\nPBIAS={eva_metrics_ET_dict[2][0]:.2f}',
                #             fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax1_et.transAxes)
                ax1_et.text(0.60, 0.95,  # 0.565, 0.85
                            f'Validation',
                            fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax1_et.transAxes)
                ## ax2_et
                ax2_et = axes_et[1]
                ax2_et.plot(obs_date_senson, mean_season_ET_dict[0], linestyle='-', linewidth=0.5, color='Red', label='Observation')
                ax2_et.plot(obs_date_senson, mean_season_ET_dict[-1], linestyle='-', linewidth=0.5, color='Blue', label='Compromise solution')
                ax2_et.fill_between(obs_date_senson, mean_season_ET_dict[1], mean_season_ET_dict[2], color='#BDBDBD', label='Envelope')
                ax2_et.set_xticks(obs_date_senson[::60])
                ax2_et.set_xticklabels(obs_date_senson[::60], fontsize=12, rotation=20, ha='center')
                ax2_et.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                # ax2_et.text(0.02, 0.88, f'KGE={eva_metrics_ET_dict[0][2]:.2f}\nNSE={eva_metrics_ET_dict[1][2]:.2f}\nPBIAS={eva_metrics_ET_dict[2][2]:.2f}',
                #             fontstyle='italic', ha='left', va='center', fontsize=10, transform=ax2_et.transAxes)
                axes_et[0].set_xlim(obs_date_day_val[0], obs_date_day_val[-1])
                axes_et[1].set_xlim(obs_date_senson[0], obs_date_senson[-1])
                fig_et.text(0.02, 0.5, 'ET (mm)', va='center', rotation='vertical', fontsize=12)
                fig_et.legend(handles=[obs_et, comp_et, env_et], loc='lower center', ncol=3, columnspacing=3,
                              bbox_to_anchor=(0.5, -0.05), fontsize=12, frameon=False)
                plt.tight_layout()
                plt.savefig(f'{swat_nsga_out}\\ET_Time_Series_Stage1.jpg')
                # plt.show()

                # Flow Duration Curve (FDC)
                print('Streamflow Flow Duration Curve (FDC):')
                fig_fdc, axes_fdc = plt.subplots(3, 1, figsize=(12, 16), sharex=True, dpi=500)
                for ax in axes_fdc[:-1]:
                    ax.tick_params(labelbottom=False)
                ## 遍历所有子图并进行绘图
                plot_fdc_list = []
                for hydro_idx in zip(range(3), ['NingDu', 'ShiCheng', 'FenKeng'], sorted(hydro_stas.values())):
                    print('hydro_idx:', hydro_idx[1])
                    ## FDC
                    ## 1) Sort
                    obs_sf_data_sort = np.sort(np.log(obs_sf_data[hydro_idx[2]]))[::-1]
                    sim_sf_data_sort = np.sort(np.log(rch_sta_sim_dict[hydro_idx[2]][2]))[::-1]
                    ## 2) Assign rank
                    sf_data_sort_rank = np.arange(1, obs_sf_data_sort.shape[0] + 1)
                    ## 3) Calculate exceedence probability (P)
                    sf_data_exce_prob = 100 * (sf_data_sort_rank / (obs_sf_data_sort.shape[0] + 1))
                    #
                    ax_fdc = axes_fdc[hydro_idx[0]]
                    obs,  = ax_fdc.plot(sf_data_exce_prob, obs_sf_data_sort, linestyle='-', linewidth=1.0, color='Red', label='Observation')
                    comp, = ax_fdc.plot(sf_data_exce_prob, sim_sf_data_sort, linestyle='-', linewidth=1.0, color='Blue', label='Compromise solution')
                    ax_fdc.set_title(f'{hydro_idx[1]} Station', fontsize=16)
                    for label in (ax_fdc.get_xticklabels() + ax_fdc.get_yticklabels()):
                        label.set_fontsize(16)
                    ax_fdc.axvline(x=2, color='Grey', linestyle='--', linewidth=1)
                    ax_fdc.axvline(x=20, color='Grey', linestyle='--', linewidth=1)
                    ax_fdc.axvline(x=70, color='Grey', linestyle='--', linewidth=1)
                    if hydro_idx[1] == 'NingDu':
                        ax_fdc.text(0.015, 0.12, 'Peak flow', ha='center', va='center', rotation='vertical', fontsize=16, transform=ax_fdc.transAxes)
                        ax_fdc.text(0.12, 0.12, 'High flow', ha='center', va='center', fontsize=16, transform=ax_fdc.transAxes)
                        ax_fdc.text(0.46, 0.12, 'Mid flow', ha='center', va='center', fontsize=16, transform=ax_fdc.transAxes)
                        ax_fdc.text(0.86, 0.12, 'Low flow', ha='center', va='center', fontsize=16, transform=ax_fdc.transAxes)
                    plot_fdc_list.append([obs, comp])
                axes_fdc[2].set_xlim(-0.3, 100.3)
                axes_fdc[2].set_xlabel('Exceedance Probability (%)', fontsize=18)
                axes_fdc[1].set_ylabel('Log Streamflow (m$^3$/s)', fontsize=18)
                axes_fdc[0].legend(handles=plot_fdc_list[0], loc='upper right', fontsize=16, frameon=False)
                plt.tight_layout()
                plt.savefig(f'{swat_nsga_out}\\Streamflow_FDC_Stage1.jpg')
                # plt.show()

        # Exit
        sys.exit()

    #################### Multi-objective Optimization ####################
    # Step 1: Create Multi-objective Optimization Problem Object
    seed = 22864
    SWAT_Problem = SWAT_Optimization_Problems.SWATProblem(SWAT_Execution_Run)

    # Step 2: Create Genetic Algorithm Object
    # Please make sure pop_size is equal or larger than the number of reference directions
    # Dimensionality of reference points must be equal to the number of objectives
    if cal_vars_list == ['Streamflow', 'ET'] and len(hydro_stas) >= 2:
        print('Adjusted weights:')
        ref_dirs_two = get_reference_directions('energy', 2, pop_size, seed=seed)
        ref_dirs = np.zeros((ref_dirs_two.shape[0], SWAT_Problem.n_obj))  # SWAT_Problem.n_obj = 3 or 4
        if len(hydro_stas) == 2:
            ref_dirs[:, 2] = ref_dirs_two[:, 1]
            equal_w = ref_dirs_two[:, 0] / 2
            ref_dirs[:, 0] = equal_w
            ref_dirs[:, 1] = equal_w
        elif len(hydro_stas) == 3:
            ref_dirs[:, 3] = ref_dirs_two[:, 1]
            equal_w = ref_dirs_two[:, 0] / 3
            ref_dirs[:, 0] = equal_w
            ref_dirs[:, 1] = equal_w
            ref_dirs[:, 2] = equal_w
    else:
        ref_dirs = get_reference_directions('energy', SWAT_Problem.n_obj, pop_size, seed=seed)
    print('ref_dirs:', ref_dirs.shape)
    print(ref_dirs)

    # By default, the population size is set to None which means that it will be equal to the number of reference line.
    # However, if desired this can be overwritten by providing a positive number.
    mu_prob = 1 / len(swat_parameter)
    print('mu_prob:', mu_prob)
    algorithm = UNSGA3(ref_dirs,
                       pop_size=pop_size,
                       n_offsprings=pop_size,
                       sampling=LHS(),
                       crossover=SBX(prob=0.9, eta=10),
                       mutation=PM(prob=mu_prob, eta=20), # Mutation probability were defined as the reciprocal of the number of calibration parameters
                       eliminate_duplicates=True,
                       save_history=True)

    # Step 3: Create Optimization Object
    num_gen = 200
    termination = get_termination('n_gen', num_gen)
    res = minimize(SWAT_Problem,
                   algorithm,
                   termination,
                   seed=seed,
                   save_history=True,
                   verbose=True)

    # Pareto-front solutions
    X = res.X  # Design space values
    F = res.F  # Objective spaces values
    if X is None:
        print(f'The algorithm was not able to find any feasible solution in {num_gen} generations')
    else:
        print('X.shape:', X.shape, 'F.shape:', F.shape)
        print('Best solution found: \nX = %s\nF = %s' % (X, F))
        # The final population
        pop_final_X = res.pop.get('X')
        pop_final_F = res.pop.get('F')
        print('\n')

        # Step 4: Multi-Criteria Decision Making (MCDM)
        approx_ideal = F.min(axis=0)
        approx_nadir = F.max(axis=0)
        print('approx_ideal:', approx_ideal)
        print('approx_nadir:', approx_nadir)
        # Normalization
        Norm_F = (F - approx_ideal) / (approx_nadir - approx_ideal)
        print('Norm_F:', Norm_F.shape)
        # Compromise Programming/Pseudo-Weights
        MCDM_scheme = 'Compromise Programming'
        Compromise_sol = None
        weights = np.array([1 / obj_func_num] * obj_func_num)  # Weights of the objectives
        if MCDM_scheme == 'Compromise Programming':
            # Compromise Programming
            Compromise_sol_ASF = ASF().do(Norm_F, 1 / weights).argmin()  # Compromise solution
            Compromise_sol = Compromise_sol_ASF
        elif MCDM_scheme == 'Pseudo-Weights':
            # Pseudo-Weights
            Compromise_sol_Pseudo = PseudoWeights(weights).do(Norm_F)
            Compromise_sol = Compromise_sol_Pseudo
        print('Compromise_sol:', Compromise_sol)
        print(f'Best regarding {MCDM_scheme}: Point \ni = %s\nF = %s' % (Compromise_sol, F[Compromise_sol]))
        print('\n')

        # Step 5: Report Results
        XF = np.hstack((X, F))  # Design space and Objective spaces values
        print('XF:', XF.shape)
        Compromise_XF = np.hstack((X[Compromise_sol], F[Compromise_sol]))
        print('Compromise_XF:', Compromise_XF.shape, Compromise_XF)
        cal_vars = ''
        header_X_F = ''
        label_objs = []
        F_drop = F  # Drop Compromise solution
        if F.shape[0] > 1:
            F_drop = np.delete(F, Compromise_sol, axis=0)
        F_drop_trans = np.zeros_like(F_drop)
        F_comp_trans = np.zeros_like(F[Compromise_sol])
        para_obj_val_dict_merge = np.array(list(chain(*para_obj_val_dict.values())))
        if cal_vars_list == ['Streamflow']:
            if XF.ndim == 1:
                XF = XF.reshape(1, -1)
            if cal_scheme == 'Multi-site':
                cal_vars = 'Streamflow_Multi-site'
                para_obj_val_dict_merge = para_obj_val_dict_merge[:, :para_obj_val_dict_merge.shape[1] - len(hydro_stas)]
                if len(hydro_stas) == 2:
                    label_objs = [f'${objective_funs[0]}_{{NingDu}}$', f'${objective_funs[0]}_{{FenKeng}}$']
                    header_X_F = ''.join([f'{item:<18s}' for item in (para_names + [f'{hydro_idx}_{obj_idx}' for obj_idx in objective_funs
                                                                                    for hydro_idx in ['NingDu', 'FenKeng']])])
                elif len(hydro_stas) == 3:
                    label_objs = [f'${objective_funs[0]}_{{NingDu}}$', f'${objective_funs[0]}_{{ShiCheng}}$', f'${objective_funs[0]}_{{FenKeng}}$']
                    header_X_F = ''.join([f'{item:<18s}' for item in (para_names + [f'{hydro_idx}_{obj_idx}' for obj_idx in objective_funs
                                                                                    for hydro_idx in ['NingDu', 'ShiCheng', 'FenKeng']])])
                #
                if objective_funs[0] in ['NSE', 'KGE', 'R2']:
                    F_drop_trans = 1 - F_drop
                    F_comp_trans = 1 - F[Compromise_sol]
                elif objective_funs[0] in ['RMSE', 'PBIAS']:
                    F_drop_trans = F_drop
                    F_comp_trans = F[Compromise_sol]
            elif cal_scheme == 'Multi-objective':
                cal_vars = 'Streamflow_Multi-objective'
                para_obj_val_dict_merge = para_obj_val_dict_merge[:, :para_obj_val_dict_merge.shape[1] - len(hydro_stas) * len(objective_funs)]
                if len(hydro_stas) == 1:
                    header_X_F = ''.join([f'{item:<18s}' for item in (para_names + [f'{hydro_idx}_{obj_idx}' for obj_idx in objective_funs
                                                                                    for hydro_idx in ['FenKeng']])])
                    if len(objective_funs) == 2:
                        if objective_funs == ['NSE', 'KGE'] or objective_funs == ['NSE', 'R2'] or objective_funs == ['KGE', 'R2']:
                            F_drop_trans = 1 - F_drop
                            F_comp_trans = 1 - F[Compromise_sol]
                            if objective_funs[1] == 'R2':
                                label_objs = [f'${objective_funs[0]}_{{FenKeng}}$', f'$R^{2}_{{FenKeng}}$']
                            else:
                                label_objs = [f'${objective_funs[0]}_{{FenKeng}}$', f'${objective_funs[1]}_{{FenKeng}}$']
                        elif (objective_funs == ['NSE', 'PBIAS'] or objective_funs == ['NSE', 'RMSE'] or
                              objective_funs == ['KGE', 'PBIAS'] or objective_funs == ['KGE', 'RMSE'] or
                              objective_funs == ['R2', 'PBIAS'] or objective_funs == ['R2', 'RMSE']):
                            F_drop_trans[:, :1] = 1 - F_drop[:, :1]
                            F_drop_trans[:, 1]  = F_drop[:, 1]
                            F_comp_trans[:1] = 1 - F[Compromise_sol][:1]
                            F_comp_trans[1]  = F[Compromise_sol][1]
                            if objective_funs[0] == 'R2':
                                label_objs = [f'$R^{2}_{{FenKeng}}$', f'${objective_funs[1]}_{{FenKeng}}$']
                            else:
                                label_objs = [f'${objective_funs[0]}_{{FenKeng}}$', f'${objective_funs[1]}_{{FenKeng}}$']
                        elif objective_funs == ['PBIAS', 'RMSE']:
                            F_drop_trans = F_drop
                            F_comp_trans = F[Compromise_sol]
                    elif len(objective_funs) == 3:
                        if objective_funs == ['NSE', 'KGE', 'R2']:
                            F_drop_trans = 1 - F_drop
                            F_comp_trans = 1 - F[Compromise_sol]
                            label_objs = [f'${objective_funs[0]}_{{FenKeng}}$', f'${objective_funs[1]}_{{FenKeng}}$', f'$R^{2}_{{FenKeng}}$']
                        elif (objective_funs == ['NSE', 'KGE', 'PBIAS'] or objective_funs == ['NSE', 'KGE', 'RMSE'] or
                              objective_funs == ['NSE', 'R2', 'PBIAS'] or objective_funs == ['NSE', 'R2', 'RMSE'] or
                              objective_funs == ['KGE', 'R2', 'PBIAS'] or objective_funs == ['KGE', 'R2', 'RMSE']):
                            F_drop_trans[:, :2] = 1 - F_drop[:, :2]
                            F_drop_trans[:, 2]  = F_drop[:, 2]
                            F_comp_trans[:2] = 1 - F[Compromise_sol][:2]
                            F_comp_trans[2]  = F[Compromise_sol][2]
                            if objective_funs[1] == 'R2':
                                label_objs = [f'${objective_funs[0]}_{{FenKeng}}$', f'$R^{2}_{{FenKeng}}$', f'${objective_funs[2]}_{{FenKeng}}$']
                            else:
                                label_objs = [f'${objective_funs[0]}_{{FenKeng}}$', f'${objective_funs[1]}_{{FenKeng}}$', f'${objective_funs[2]}_{{FenKeng}}$']
                        elif (objective_funs == ['NSE', 'PBIAS', 'RMSE'] or objective_funs == ['KGE', 'PBIAS', 'RMSE'] or
                              objective_funs == ['R2', 'PBIAS', 'RMSE']):
                            F_drop_trans[:, :1] = 1 - F_drop[:, :1]
                            F_drop_trans[:, 1:] = F_drop[:, 1:]
                            F_comp_trans[:1] = 1 - F[Compromise_sol][:1]
                            F_comp_trans[1:] = F[Compromise_sol][1:]
                            if objective_funs[0] == 'R2':
                                label_objs = [f'$R^{2}_{{FenKeng}}$', f'${objective_funs[1]}_{{FenKeng}}$', f'${objective_funs[2]}_{{FenKeng}}$']
                            else:
                                label_objs = [f'${objective_funs[0]}_{{FenKeng}}$', f'${objective_funs[1]}_{{FenKeng}}$', f'${objective_funs[2]}_{{FenKeng}}$']
                elif len(hydro_stas) == 2:
                    header_X_F = ''.join([f'{item:<18s}' for item in (para_names + [f'{hydro_idx}_{obj_idx}' for obj_idx in objective_funs
                                                                                    for hydro_idx in ['NingDu', 'FenKeng']])])
                    if len(objective_funs) == 2:
                        pass
                    elif len(objective_funs) == 3:
                        pass
                elif len(hydro_stas) == 3:
                    header_X_F = ''.join([f'{item:<18s}' for item in (para_names + [f'{hydro_idx}_{obj_idx}' for obj_idx in objective_funs
                                                                                    for hydro_idx in ['NingDu', 'ShiCheng', 'FenKeng']])])
                    if len(objective_funs) == 2:
                        pass
                    elif len(objective_funs) == 3:
                        pass
            # obs_data = SWAT_Execution_Run.obs_sf_data
        elif cal_vars_list == ['LAI']:
            if XF.ndim == 1:
                XF = XF.reshape(1, -1)
            if cal_scheme == 'Single-objective':
                cal_vars = 'LAI_Single-objective'
                para_obj_val_dict_merge = para_obj_val_dict_merge[:, :para_obj_val_dict_merge.shape[1] - 1]
            elif cal_scheme == 'Multi-objective':
                cal_vars = 'LAI_Multi-objective'
                para_obj_val_dict_merge = para_obj_val_dict_merge[:, :para_obj_val_dict_merge.shape[1] - len(objective_funs)]
                if len(objective_funs) == 2:
                    if objective_funs == ['NSE', 'KGE'] or objective_funs == ['NSE', 'R2'] or objective_funs == ['KGE', 'R2']:
                        F_drop_trans = 1 - F_drop
                        F_comp_trans = 1 - F[Compromise_sol]
                        if objective_funs[1] == 'R2':
                            label_objs = [f'${objective_funs[0]}_{{LAI}}$', f'$R^{2}_{{LAI}}$']
                        else:
                            label_objs = [f'${objective_funs[0]}_{{LAI}}$', f'${objective_funs[1]}_{{LAI}}$']
                    elif (objective_funs == ['NSE', 'PBIAS'] or objective_funs == ['NSE', 'RMSE'] or
                          objective_funs == ['KGE', 'PBIAS'] or objective_funs == ['KGE', 'RMSE'] or
                          objective_funs == ['R2', 'PBIAS'] or objective_funs == ['R2', 'RMSE']):
                        F_drop_trans[:, :1] = 1 - F_drop[:, :1]
                        F_drop_trans[:, 1]  = F_drop[:, 1]
                        F_comp_trans[:1] = 1 - F[Compromise_sol][:1]
                        F_comp_trans[1]  = F[Compromise_sol][1]
                        if objective_funs[0] == 'R2':
                            label_objs = [f'$R^{2}_{{LAI}}$', f'${objective_funs[1]}_{{LAI}}$']
                        else:
                            label_objs = [f'${objective_funs[0]}_{{LAI}}$', f'${objective_funs[1]}_{{LAI}}$']
                    elif objective_funs == ['PBIAS', 'RMSE']:
                        F_drop_trans = F_drop
                        F_comp_trans = F[Compromise_sol]
                        label_objs = [f'${objective_funs[0]}_{{LAI}}$', f'${objective_funs[1]}_{{LAI}}$']
                elif len(objective_funs) == 3:
                    if objective_funs == ['NSE', 'KGE', 'R2']:
                        F_drop_trans = 1 - F_drop
                        F_comp_trans = 1 - F[Compromise_sol]
                        label_objs = [f'${objective_funs[0]}_{{LAI}}$', f'${objective_funs[1]}_{{LAI}}$', f'$R^{2}_{{LAI}}$']
                    elif (objective_funs == ['NSE', 'KGE', 'PBIAS'] or objective_funs == ['NSE', 'KGE', 'RMSE'] or
                          objective_funs == ['NSE', 'R2', 'PBIAS'] or objective_funs == ['NSE', 'R2', 'RMSE'] or
                          objective_funs == ['KGE', 'R2', 'PBIAS'] or objective_funs == ['KGE', 'R2', 'RMSE']):
                        F_drop_trans[:, :2] = 1 - F_drop[:, :2]
                        F_drop_trans[:, 2]  = F_drop[:, 2]
                        F_comp_trans[:2] = 1 - F[Compromise_sol][:2]
                        F_comp_trans[2]  = F[Compromise_sol][2]
                        if objective_funs[1] == 'R2':
                            label_objs = [f'${objective_funs[0]}_{{LAI}}$', f'$R^{2}_{{LAI}}$', f'${objective_funs[2]}_{{LAI}}$']
                        else:
                            label_objs = [f'${objective_funs[0]}_{{LAI}}$', f'${objective_funs[1]}_{{LAI}}$', f'${objective_funs[2]}_{{LAI}}$']
                    elif (objective_funs == ['NSE', 'PBIAS', 'RMSE'] or objective_funs == ['KGE', 'PBIAS', 'RMSE'] or
                          objective_funs == ['R2', 'PBIAS', 'RMSE']):
                        F_drop_trans[:, :1] = 1 - F_drop[:, :1]
                        F_drop_trans[:, 1:] = F_drop[:, 1:]
                        F_comp_trans[:1] = 1 - F[Compromise_sol][:1]
                        F_comp_trans[1:] = F[Compromise_sol][1:]
                        if objective_funs[0] == 'R2':
                            label_objs = [f'$R^{2}_{{LAI}}$', f'${objective_funs[1]}_{{LAI}}$', f'${objective_funs[2]}_{{LAI}}$']
                        else:
                            label_objs = [f'${objective_funs[0]}_{{LAI}}$', f'${objective_funs[1]}_{{LAI}}$', f'${objective_funs[2]}_{{LAI}}$']
            # obs_data = SWAT_Execution_Run.LAI_obs_data_dict_area_w
            header_X_F = ''.join([f'{item:<18s}' for item in (para_names + [f'LAI_{obj_idx}' for obj_idx in objective_funs])])
        elif cal_vars_list == ['BIOM']:
            if XF.ndim == 1:
                XF = XF.reshape(1, -1)
            if cal_scheme == 'Single-objective':
                cal_vars = 'BIOM_Single-objective'
                para_obj_val_dict_merge = para_obj_val_dict_merge[:, :para_obj_val_dict_merge.shape[1] - 1]
            elif cal_scheme == 'Multi-objective':
                cal_vars = 'BIOM_Multi-objective'
                para_obj_val_dict_merge = para_obj_val_dict_merge[:, :para_obj_val_dict_merge.shape[1] - len(objective_funs)]
                if len(objective_funs) == 2:
                    if objective_funs == ['NSE', 'KGE'] or objective_funs == ['NSE', 'R2'] or objective_funs == ['KGE', 'R2']:
                        F_drop_trans = 1 - F_drop
                        F_comp_trans = 1 - F[Compromise_sol]
                        if objective_funs[1] == 'R2':
                            label_objs = [f'${objective_funs[0]}_{{BIOM}}$', f'$R^{2}_{{BIOM}}$']
                        else:
                            label_objs = [f'${objective_funs[0]}_{{BIOM}}$', f'${objective_funs[1]}_{{BIOM}}$']
                    elif (objective_funs == ['NSE', 'PBIAS'] or objective_funs == ['NSE', 'RMSE'] or
                          objective_funs == ['KGE', 'PBIAS'] or objective_funs == ['KGE', 'RMSE'] or
                          objective_funs == ['R2', 'PBIAS'] or objective_funs == ['R2', 'RMSE']):
                        F_drop_trans[:, :1] = 1 - F_drop[:, :1]
                        F_drop_trans[:, 1]  = F_drop[:, 1]
                        F_comp_trans[:1] = 1 - F[Compromise_sol][:1]
                        F_comp_trans[1]  = F[Compromise_sol][1]
                        if objective_funs[0] == 'R2':
                            label_objs = [f'$R^{2}_{{BIOM}}$', f'${objective_funs[1]}_{{BIOM}}$']
                        else:
                            label_objs = [f'${objective_funs[0]}_{{BIOM}}$', f'${objective_funs[1]}_{{BIOM}}$']
                    elif objective_funs == ['PBIAS', 'RMSE']:
                        F_drop_trans = F_drop
                        F_comp_trans = F[Compromise_sol]
                        label_objs = [f'${objective_funs[0]}_{{BIOM}}$', f'${objective_funs[1]}_{{BIOM}}$']
                elif len(objective_funs) == 3:
                    if objective_funs == ['NSE', 'KGE', 'R2']:
                        F_drop_trans = 1 - F_drop
                        F_comp_trans = 1 - F[Compromise_sol]
                        label_objs = [f'${objective_funs[0]}_{{BIOM}}$', f'${objective_funs[1]}_{{BIOM}}$', f'$R^{2}_{{BIOM}}$']
                    elif (objective_funs == ['NSE', 'KGE', 'PBIAS'] or objective_funs == ['NSE', 'KGE', 'RMSE'] or
                          objective_funs == ['NSE', 'R2', 'PBIAS'] or objective_funs == ['NSE', 'R2', 'RMSE'] or
                          objective_funs == ['KGE', 'R2', 'PBIAS'] or objective_funs == ['KGE', 'R2', 'RMSE']):
                        F_drop_trans[:, :2] = 1 - F_drop[:, :2]
                        F_drop_trans[:, 2]  = F_drop[:, 2]
                        F_comp_trans[:2] = 1 - F[Compromise_sol][:2]
                        F_comp_trans[2]  = F[Compromise_sol][2]
                        if objective_funs[1] == 'R2':
                            label_objs = [f'${objective_funs[0]}_{{BIOM}}$', f'$R^{2}_{{BIOM}}$', f'${objective_funs[2]}_{{BIOM}}$']
                        else:
                            label_objs = [f'${objective_funs[0]}_{{BIOM}}$', f'${objective_funs[1]}_{{BIOM}}$', f'${objective_funs[2]}_{{BIOM}}$']
                    elif (objective_funs == ['NSE', 'PBIAS', 'RMSE'] or objective_funs == ['KGE', 'PBIAS', 'RMSE'] or
                          objective_funs == ['R2', 'PBIAS', 'RMSE']):
                        F_drop_trans[:, :1] = 1 - F_drop[:, :1]
                        F_drop_trans[:, 1:]  = F_drop[:, 1:]
                        F_comp_trans[:1] = 1 - F[Compromise_sol][:1]
                        F_comp_trans[1:]  = F[Compromise_sol][1:]
                        if objective_funs[0] == 'R2':
                            label_objs = [f'$R^{2}_{{BIOM}}$', f'${objective_funs[1]}_{{BIOM}}$', f'${objective_funs[2]}_{{BIOM}}$']
                        else:
                            label_objs = [f'${objective_funs[0]}_{{BIOM}}$', f'${objective_funs[1]}_{{BIOM}}$', f'${objective_funs[2]}_{{BIOM}}$']
            obs_data = SWAT_Execution_Run.BIOM_obs_data_dict_area_w
            header_X_F = ''.join([f'{item:<18s}' for item in (para_names + [f'BIOM_{obj_idx}' for obj_idx in objective_funs])])
        elif cal_vars_list == ['ET']:
            if XF.ndim == 1:
                XF = XF.reshape(1, -1)
            if cal_scheme == 'Single-objective':
                cal_vars = 'ET_Single-objective'
                para_obj_val_dict_merge = para_obj_val_dict_merge[:, :para_obj_val_dict_merge.shape[1] - 1]
            elif cal_scheme == 'Multi-objective':
                cal_vars = 'ET_Multi-objective'
                if len(objective_funs) == 2:
                    para_obj_val_dict_merge = para_obj_val_dict_merge[:, :para_obj_val_dict_merge.shape[1] - 2]
                    if objective_funs == ['NSE', 'KGE'] or objective_funs == ['NSE', 'R2'] or objective_funs == ['KGE', 'R2']:
                        F_drop_trans = 1 - F_drop
                        F_comp_trans = 1 - F[Compromise_sol]
                        if objective_funs[1] == 'R2':
                            label_objs = [f'${objective_funs[0]}_{{ET}}$', f'$R^{2}_{{ET}}$']
                        else:
                            label_objs = [f'${objective_funs[0]}_{{ET}}$', f'${objective_funs[1]}_{{ET}}$']
                    elif (objective_funs == ['NSE', 'PBIAS'] or objective_funs == ['NSE', 'RMSE'] or
                          objective_funs == ['KGE', 'PBIAS'] or objective_funs == ['KGE', 'RMSE'] or
                          objective_funs == ['R2', 'PBIAS'] or objective_funs == ['R2', 'RMSE']):
                        F_drop_trans[:, :1] = 1 - F_drop[:, :1]
                        F_drop_trans[:, 1]  = F_drop[:, 1]
                        F_comp_trans[:1] = 1 - F[Compromise_sol][:1]
                        F_comp_trans[1]  = F[Compromise_sol][1]
                        if objective_funs[0] == 'R2':
                            label_objs = [f'$R^{2}_{{ET}}$', f'${objective_funs[1]}_{{ET}}$']
                        else:
                            label_objs = [f'${objective_funs[0]}_{{ET}}$', f'${objective_funs[1]}_{{ET}}$']
                    elif objective_funs == ['PBIAS', 'RMSE']:
                        F_drop_trans = F_drop
                        F_comp_trans = F[Compromise_sol]
                        label_objs = [f'${objective_funs[0]}_{{ET}}$', f'${objective_funs[1]}_{{ET}}$']
                elif len(objective_funs) == 3:
                    para_obj_val_dict_merge = para_obj_val_dict_merge[:, :para_obj_val_dict_merge.shape[1] - 3]
                    if objective_funs == ['NSE', 'KGE', 'R2']:
                        F_drop_trans = 1 - F_drop
                        F_comp_trans = 1 - F[Compromise_sol]
                        label_objs = [f'${objective_funs[0]}_{{ET}}$', f'${objective_funs[1]}_{{ET}}$', f'$R^{2}_{{ET}}$']
                    elif (objective_funs == ['NSE', 'KGE', 'PBIAS'] or objective_funs == ['NSE', 'KGE', 'RMSE'] or
                          objective_funs == ['NSE', 'R2', 'PBIAS'] or objective_funs == ['NSE', 'R2', 'RMSE'] or
                          objective_funs == ['KGE', 'R2', 'PBIAS'] or objective_funs == ['KGE', 'R2', 'RMSE']):
                        F_drop_trans[:, :2] = 1 - F_drop[:, :2]
                        F_drop_trans[:, 2]  = F_drop[:, 2]
                        F_comp_trans[:2] = 1 - F[Compromise_sol][:2]
                        F_comp_trans[2]  = F[Compromise_sol][2]
                        if objective_funs[1] == 'R2':
                            label_objs = [f'${objective_funs[0]}_{{ET}}$', f'$R^{2}_{{ET}}$', f'${objective_funs[2]}_{{ET}}$']
                        else:
                            label_objs = [f'${objective_funs[0]}_{{ET}}$', f'${objective_funs[1]}_{{ET}}$', f'${objective_funs[2]}_{{ET}}$']
                    elif (objective_funs == ['NSE', 'PBIAS', 'RMSE'] or objective_funs == ['KGE', 'PBIAS', 'RMSE'] or
                          objective_funs == ['R2', 'PBIAS', 'RMSE']):
                        F_drop_trans[:, :1] = 1 - F_drop[:, :1]
                        F_drop_trans[:, 1:]  = F_drop[:, 1:]
                        F_comp_trans[:1] = 1 - F[Compromise_sol][:1]
                        F_comp_trans[1:]  = F[Compromise_sol][1:]
                        if objective_funs[0] == 'R2':
                            label_objs = [f'$R^{2}_{{ET}}$', f'${objective_funs[1]}_{{ET}}$', f'${objective_funs[2]}_{{ET}}$']
                        else:
                            label_objs = [f'${objective_funs[0]}_{{ET}}$', f'${objective_funs[1]}_{{ET}}$', f'${objective_funs[2]}_{{ET}}$']
            obs_data = SWAT_Execution_Run.ET_obs_data_dict_area_w
            header_X_F = ''.join([f'{item:<18s}' for item in (para_names + [f'ET_{obj_idx}' for obj_idx in objective_funs])])
        elif cal_vars_list == ['Streamflow', 'ET']:
            cal_vars = 'Streamflow_ET'
            para_obj_val_dict_merge = para_obj_val_dict_merge[:, :para_obj_val_dict_merge.shape[1] - (len(hydro_stas) + 1)]
            if len(hydro_stas) == 1:
                header_X_F = ''.join([f'{item:<18s}' for item in (para_names + [f'{item}_{objective_funs[0]}' for item in ['FenKeng', 'ET']])])
                if objective_funs[0] in ['NSE', 'KGE', 'R2']:
                    F_drop_trans = 1 - F_drop
                    F_comp_trans = 1 - F[Compromise_sol]
                    if 'R2' in objective_funs:
                        label_objs = [f'$R^{2}_{{FenKeng}}$', f'$R^{2}_{{ET}}$']
                    else:
                        label_objs = [f'${objective_funs[0]}_{{FenKeng}}$', f'${objective_funs[0]}_{{ET}}$']
                elif objective_funs[0] in ['RMSE', 'PBIAS']:
                    F_drop_trans = F_drop
                    F_comp_trans = F[Compromise_sol]
                    label_objs = [f'${objective_funs[0]}_{{FenKeng}}$', f'${objective_funs[0]}_{{ET}}$']
            elif len(hydro_stas) == 2:
                header_X_F = ''.join([f'{item:<18s}' for item in (para_names + [f'{item}_{objective_funs[0]}' for item in ['NingDu', 'FenKeng', 'ET']])])
                if objective_funs[0] in ['NSE', 'KGE', 'R2']:
                    F_drop_trans = 1 - F_drop
                    F_comp_trans = 1 - F[Compromise_sol]
                    if 'R2' in objective_funs:
                        label_objs = [f'$R^{2}_{{NingDu}}$', f'$R^{2}_{{FenKeng}}$', f'$R^{2}_{{ET}}$']
                    else:
                        label_objs = [f'${objective_funs[0]}_{{NingDu}}$', f'${objective_funs[0]}_{{FenKeng}}$', f'${objective_funs[0]}_{{ET}}$']
                elif objective_funs[0] in ['RMSE', 'PBIAS']:
                    F_drop_trans = F_drop
                    F_comp_trans = F[Compromise_sol]
                    label_objs = [f'${objective_funs[0]}_{{NingDu}}$', f'${objective_funs[0]}_{{FenKeng}}$', f'${objective_funs[0]}_{{ET}}$']
            elif len(hydro_stas) == 3:
                header_X_F = ''.join([f'{item:<18s}' for item in (para_names + [f'{item}_{objective_funs[0]}'
                                                                                for item in ['NingDu', 'ShiCheng', 'FenKeng', 'ET']])])
                if objective_funs[0] in ['NSE', 'KGE', 'R2']:
                    F_drop_trans = 1 - F_drop
                    F_comp_trans = 1 - F[Compromise_sol]
                    if 'R2' in objective_funs:
                        label_objs = [f'$R^{2}_{{NingDu}}$', f'$R^{2}_{{ShiCheng}}$', f'$R^{2}_{{FenKeng}}$', f'$R^{2}_{{ET}}$']
                    else:
                        label_objs = [f'${objective_funs[0]}_{{NingDu}}$', f'${objective_funs[0]}_{{ShiCheng}}$',
                                      f'${objective_funs[0]}_{{FenKeng}}$', f'${objective_funs[0]}_{{ET}}$']
                elif objective_funs[0] in ['RMSE', 'PBIAS']:
                    F_drop_trans = F_drop
                    F_comp_trans = F[Compromise_sol]
                    label_objs = [f'${objective_funs[0]}_{{NingDu}}$', f'${objective_funs[0]}_{{ShiCheng}}$',
                                  f'${objective_funs[0]}_{{FenKeng}}$', f'${objective_funs[0]}_{{ET}}$']
        print('para_obj_val_dict_merge:', para_obj_val_dict_merge.shape)
        # Parameter values and objective function values for each generation
        with open(f'{swat_nsga_out}\\Para_Obj_Values_Gen_{cal_vars}.txt', 'w') as f_gen:
            f_gen.writelines(header_X_F + '\n')
            for gen_idx in range(num_gen):
                f_gen.writelines(str(gen_idx + 1) + '\n')
                for row_idx in range(para_obj_val_dict[gen_idx + 1].shape[0]):
                    para_obj_val = ''.join([f'{item:<18.8f}' for item in para_obj_val_dict[gen_idx + 1][row_idx]])
                    f_gen.writelines(para_obj_val + '\n')
                f_gen.writelines('\n')
        # Pareto-front solutions
        with open(f'{swat_nsga_out}\\Pareto-front_solutions_{cal_vars}.txt', 'w') as f_p:
            f_p.write(f'{header_X_F}\n')  # 手动写入header，不加#
            np.savetxt(fname=f_p, X=XF, fmt='%-14.8f', delimiter='    ')
        # Compromise solution
        with open(f'{swat_nsga_out}\\Compromise_solution_{cal_vars}.txt', 'w') as f_bp:
            f_bp.write(f'{header_X_F}\n')  # 手动写入header，不加#
            np.savetxt(fname=f_bp, X=Compromise_XF.reshape(1, -1), fmt='%-14.8f', delimiter='    ')
        print('Results Export Finished!\n')

        # Step 6: Plotting
        # 1) Analysis of Convergence
        hist = res.history  # history
        print('hist:', len(hist))
        # Running Metric
        running = RunningMetricAnimation(delta_gen=50,
                                         n_plots=5,
                                         key_press=False,
                                         do_show=True)
        for algorithm in hist:
            running.update(algorithm)

        # Hypvervolume (HV)
        n_gens  = []  # generation number
        n_evals = []  # corresponding number of function evaluations
        n_nds   = []  # the number of non-dominated solutions
        hist_F  = []  # the objective space values in each generation
        for algo in hist:
            n_gens.append(algo.output.n_gen.value)
            # store the number of function evaluations
            n_evals.append(algo.evaluator.n_eval)
            n_nds.append(algo.output.n_nds.value)
            # retrieve the optimum from the algorithm
            opt = algo.opt
            # filter out only the feasible and append and objective space values
            feas = np.where(opt.get('feasible'))[0]
            hist_F.append(opt.get('F')[feas])
        print('n_nds:', len(n_nds), n_nds)
        # reference point
        ref_point = np.array([1.1] * obj_func_num)
        metric = Hypervolume(ref_point=ref_point, norm_ref_point=False, zero_to_one=True, ideal=approx_ideal, nadir=approx_nadir)
        hv = [metric.do(_F) for _F in hist_F]
        print('Hypvervolume:', len(hv), hv)
        # Output
        hypvervolume_arr = np.array(list(zip(n_gens, hv)))
        # print('hypvervolume_arr:', hypvervolume_arr.shape, '\n', hypvervolume_arr)
        n_nds_arr = np.array(list(zip(n_gens, n_nds)))
        # print('n_nds_arr:', n_nds_arr.shape, '\n', n_nds_arr)
        # Hypvervolume
        with open(f'{swat_nsga_out}\\Hypvervolume_{cal_vars}.txt', 'w') as f_h:
            f_h.write('num_gen'.ljust(14) + 'hypvervolume'.ljust(14) + '\n')  # 手动写入header，不加#
            # 遍历数组的每一行
            for row in hypvervolume_arr:
                line = f'{int(row[0]):<13d} {row[1]:<13.8f}\n'
                f_h.write(line)
        # Non-dominated solutions
        with open(f'{swat_nsga_out}\\Non-dominated_solutions_{cal_vars}.txt', 'w') as f_h:
            f_h.write('num_gen'.ljust(14) + 'n_nds'.ljust(14) + '\n')  # 手动写入header，不加#
            # 遍历数组的每一行
            for row in n_nds_arr:
                line = f'{int(row[0]):<13d} {row[1]:<13d}\n'
                f_h.write(line)
        # Hypvervolume/Non-dominated solutions Plot
        fig_hv_nds, ax_hv = plt.subplots(figsize=(12, 12), sharex=True, dpi=400)
        # 绘制柱状图
        hv_color = '#00AFEF'
        ax_hv.set_xlabel('Generation number', fontsize=18)
        ax_hv.set_ylabel('Normalized Hypervolume Indicator', color=hv_color, fontsize=18)
        hv_line, = ax_hv.plot(n_gens, hv, color=hv_color, linestyle='-')
        ax_hv.tick_params(axis='x', labelsize=18)
        ax_hv.tick_params(axis='y', colors=hv_color, labelsize=18)
        # 创建第二个y轴
        ax_nds = ax_hv.twinx()  # 共享x轴
        nds_color = '#008B00'
        ax_nds.set_ylabel('Number of near-optimal Pareto solutions', color=nds_color, fontsize=18)
        nds_line, = ax_nds.plot(n_gens, n_nds, color=nds_color, linestyle='-')
        ax_nds.spines['left'].set_color(hv_color)
        ax_nds.spines['right'].set_color(nds_color)
        ax_nds.tick_params(axis='y', colors=nds_color, labelsize=18)
        # 设置图例的handle和label
        ax_hv.legend(handles=[hv_line, nds_line], labels=['Normalized Hypervolume Indicator', 'Number of near-optimal Pareto solutions'],
                     loc='best', fontsize=18)
        # 显示图形
        plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
        plt.savefig(f'{swat_nsga_out}\\HV_NDS_{cal_vars}.jpg')
        # plt.show()
        print('\n')

        # 2) Parallel Coordinate Plots (Para Num <= 26)
        pareto_front_solutions_df  = pd.DataFrame(X, columns=para_names)
        para_obj_val_dict_merge_df = pd.DataFrame(para_obj_val_dict_merge, columns=para_names)
        compromise_solution_df     = pd.DataFrame([X[Compromise_sol]], columns=para_names)
        ## Only keep FRST and FRSE
        col_to_drop = [
            col for col in pareto_front_solutions_df.columns
            if '{' in col and '}' in col and re.search(r'\{(\d+)\}', col) and int(re.search(r'\{(\d+)\}', col).group(1)) not in [6, 8]
        ]
        pareto_front_solutions_df.drop(columns=col_to_drop, inplace=True)
        para_obj_val_dict_merge_df.drop(columns=col_to_drop, inplace=True)
        compromise_solution_df.drop(columns=col_to_drop, inplace=True)

        print('pareto_front_solutions_df:', pareto_front_solutions_df.shape, '\n', pareto_front_solutions_df)
        print('para_obj_val_dict_merge_df:', para_obj_val_dict_merge_df.shape, '\n', para_obj_val_dict_merge_df)
        print('compromise_solution_df:', compromise_solution_df.shape, '\n', compromise_solution_df)

        para_pf            = pareto_front_solutions_df.to_numpy()   # Convert input data to numpy
        para_non_pf        = para_obj_val_dict_merge_df.to_numpy()  # Convert input data to numpy
        compromise_para_pf = compromise_solution_df.to_numpy()   # Convert input data to numpy
        print('para_pf:', para_pf.shape)
        print('para_non_pf:', para_non_pf.shape)
        print('compromise_para_pf:', compromise_para_pf.shape)
        para_bounds_pf = [para_bounds[para_names.index(para_idx)] for para_idx in pareto_front_solutions_df.columns]
        print('para_bounds_pf:', len(para_bounds_pf))
        para_bounds_pf_min = np.array(para_bounds_pf)[:, 0]
        para_bounds_pf_max = np.array(para_bounds_pf)[:, 1]
        para_dys = para_bounds_pf_max - para_bounds_pf_min
        # transform all data to be compatible with the main axis
        para_pf_trans = np.zeros_like(para_pf)
        para_non_pf_trans = np.zeros_like(para_non_pf)
        compromise_para_pf_trans = np.zeros_like(compromise_para_pf)
        para_pf_trans[:, 0] = para_pf[:, 0]
        para_non_pf_trans[:, 0] = para_non_pf[:, 0]
        compromise_para_pf_trans[:, 0] = compromise_para_pf[:, 0]
        para_pf_trans[:, 1:] = (para_pf[:, 1:] - para_bounds_pf_min[1:]) / para_dys[1:] * para_dys[0] + para_bounds_pf_min[0]
        para_non_pf_trans[:, 1:] = (para_non_pf[:, 1:] - para_bounds_pf_min[1:]) / para_dys[1:] * para_dys[0] + para_bounds_pf_min[0]
        compromise_para_pf_trans[:, 1:] = (compromise_para_pf[:, 1:] - para_bounds_pf_min[1:]) / para_dys[1:] * para_dys[0] + para_bounds_pf_min[0]
        fig, axs = plt.subplots(figsize=(16, 8), sharey=False, dpi=300)
        axes = [axs] + [axs.twinx() for _ in range(para_non_pf.shape[1] - 1)]
        ticks_axs = []
        for i, ax in enumerate(axes):
            ax.set_ylim(para_bounds_pf_min[i], para_bounds_pf_max[i])
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_position(('axes', i / (para_pf.shape[1] - 1)))
            ax.yaxis.set_ticks_position('left')
            ax.tick_params(axis='y', labelsize=12)  # 设置y轴刻度标签的字体大小
            ax.set_yticks(np.linspace(para_bounds_pf_min[i], para_bounds_pf_max[i], 5))  # 刻度标签5等分
        axs.set_xlim(0, para_pf.shape[1] - 1)
        axs.set_xticks(range(para_pf.shape[1]))
        if len(para_names) > 14:
            axs.set_xticklabels(pareto_front_solutions_df.columns, rotation=30, fontsize=12)
        else:
            axs.set_xticklabels(pareto_front_solutions_df.columns, fontsize=10)
        axs.tick_params(axis='x', which='major', pad=6)
        axs.xaxis.tick_top()
        # Plot Non-pareto solutions
        pf_pcp_plots, non_pf_pcp_plots, best_pf_pcp_plots = [], [], []
        for j in range(para_non_pf.shape[0]):
            non_pf_pcp_plot, = axs.plot(range(para_non_pf.shape[1]), para_non_pf_trans[j, :], c='#C7E9B4', label='Non-pareto solutions', zorder=1)
            non_pf_pcp_plots.append(non_pf_pcp_plot)
        for j in range(para_pf.shape[0]):
            pf_pcp_plot, = axs.plot(range(para_pf.shape[1]), para_pf_trans[j, :], c='#40B6C4', label='Pareto solutions', zorder=2)
            pf_pcp_plots.append(pf_pcp_plot)
        for j in range(compromise_para_pf.shape[0]):
            best_pf_pcp_plot, = axs.plot(range(compromise_para_pf.shape[1]), compromise_para_pf_trans[j, :], c='#FFB228', label='Compromise solution', zorder=3)
            best_pf_pcp_plots.append(best_pf_pcp_plot)
        axs.legend(handles=[best_pf_pcp_plots[0], pf_pcp_plots[0], non_pf_pcp_plots[0]], loc='lower center', ncol=3, columnspacing=8,
                   bbox_to_anchor=(0.5, -0.10), fontsize=14, frameon=False)
        plt.tight_layout()
        plt.savefig(f'{swat_nsga_out}\\Parallel_Coordinate_Plots_{cal_vars}.jpg')
        # plt.show()
        print('\n')

        # 3) Objective Space
        if obj_func_num == 2:
            if ((cal_vars_list == ['LAI'] and cal_scheme == 'Multi-objective') or
                (cal_vars_list == ['BIOM'] and cal_scheme == 'Multi-objective') or
                (cal_vars_list == ['ET'] and cal_scheme == 'Multi-objective') or
                (cal_vars_list == ['Streamflow', 'ET'] and len(hydro_stas) == 1) or
                (cal_vars_list == ['Streamflow'] and cal_scheme == 'Multi-site' and len(hydro_stas) == 2) or
                (cal_vars_list == ['Streamflow'] and cal_scheme == 'Multi-objective' and len(hydro_stas) == 1 and len(objective_funs) == 2)):
                ### 2D Scatter
                fig2d, ax2d = plt.subplots(figsize=(12, 12), dpi=300)
                # 绘制散点图
                # Compromise solution
                scatter_2d_cp = ax2d.scatter(F_comp_trans[0], F_comp_trans[1], c='#FFB228', s=240, edgecolors='white',
                                             label='Compromise solution', alpha=1.0, marker='*', zorder=2)
                # Pareto front solutions
                scatter_2d_pf = ax2d.scatter(F_drop_trans[:, 0], F_drop_trans[:, 1], c='#40B6C4', s=80, edgecolors='white',
                                             label='Pareto solutions', alpha=1.0, zorder=1)
                # 设置子图的标题和坐标轴标签
                ax2d.set_xlabel(f'{label_objs[0]}', fontsize=14)
                ax2d.set_ylabel(f'{label_objs[1]}', fontsize=14)
                ax2d.tick_params(axis='x', pad=2, labelsize=14)
                ax2d.tick_params(axis='y', pad=2, labelsize=14)
                # 反转x轴坐标
                plt.gca().invert_xaxis()
                # 添加图例，横跨三个散点图
                plt.legend(loc='upper right', ncol=1, columnspacing=2, fontsize=14, frameon=False)
                plt.tight_layout()
                plt.savefig(f'{swat_nsga_out}\\2D_Scatter_Pareto_Front_{cal_vars}.jpg')
                # plt.show()
        elif obj_func_num == 3:
            ### 3D Scatter
            fig2d3d = plt.figure(figsize=(12, 12), dpi=300)
            # 创建GridSpec，这里不设置width_ratios，因为宽度很难精确匹配
            gs2d3d = gridspec.GridSpec(2, 3, height_ratios=[3, 1], wspace=0.35, hspace=0.2, figure=fig2d3d)
            ax3d = fig2d3d.add_subplot(gs2d3d[0, 0:3], projection='3d')
            # Compromise solution
            sc_cp = ax3d.scatter(F_comp_trans[0], F_comp_trans[1], F_comp_trans[2], c='#FFB228', s=180,
                                 edgecolors='white', label='Compromise solution', marker='*', alpha=1.0, zorder=2)
            # Pareto front solutions
            sc_pf = ax3d.scatter(F_drop_trans[:, 0], F_drop_trans[:, 1], F_drop_trans[:, 2], c='#40B6C4', s=60,
                                 edgecolors='white', label='Pareto solutions', alpha=1.0, zorder=1)
            ax3d.set_title('(a)', y=1.03, fontsize=14)
            ### 设置坐标轴标签
            ax3d.set_xlabel(label_objs[0], labelpad=12, fontsize=14)
            ax3d.set_ylabel(label_objs[1], labelpad=12, fontsize=14)
            ax3d.set_zlabel(label_objs[2], labelpad=12, fontsize=14)
            # 设置每个轴的刻度标签字体大小
            ax3d.tick_params(axis='x', pad=4, labelsize=14)
            ax3d.tick_params(axis='y', pad=4, labelsize=14)
            ax3d.tick_params(axis='z', pad=7, labelsize=14)
            ax3d.xaxis.set_major_locator(MaxNLocator(4))
            ax3d.yaxis.set_major_locator(MaxNLocator(4))
            ax3d.zaxis.set_major_locator(MaxNLocator(4))
            ax3d.view_init(elev=30, azim=30)  # Set elevation and azimuth angles
            # 设置坐标轴比例相等，以更好地显示3D效果
            ax3d.set_box_aspect((1, 1, 1))

            ### 2D Scatter
            ax2ds = [fig2d3d.add_subplot(gs2d3d[1, i]) for i in range(3)]
            # 绘制散点图
            scatter_2d_cps, scatter_2d_pfs = [], []
            for i, (x_dim, y_dim, title_2d) in enumerate([(0, 1, '(b)'), (0, 2, '(c)'), (1, 2, '(d)')]):
                # Compromise solution
                scatter_2d_cp = ax2ds[i].scatter(F_comp_trans[x_dim], F_comp_trans[y_dim], c='#FFB228', s=180, edgecolors='white',
                                                 label='Compromise solution', alpha=1.0, marker='*', zorder=2)
                # Pareto front solutions
                scatter_2d_pf = ax2ds[i].scatter(F_drop_trans[:, x_dim], F_drop_trans[:, y_dim], c='#40B6C4', s=60, edgecolors='white',
                                                 label='Pareto solutions', alpha=1.0, zorder=1)
                # 设置子图的标题和坐标轴标签
                ax2ds[i].set_xlabel(f'{label_objs[x_dim]}', fontsize=14)
                ax2ds[i].set_ylabel(f'{label_objs[y_dim]}', fontsize=14)
                ax2ds[i].tick_params(axis='x', pad=2, labelsize=14)
                ax2ds[i].tick_params(axis='y', pad=2, labelsize=14)
                ax2ds[i].xaxis.set_major_locator(MaxNLocator(5))
                ax2ds[i].yaxis.set_major_locator(MaxNLocator(5))
                ax2ds[i].xaxis.tick_top()
                ax2ds[i].yaxis.tick_right()
                ax2ds[i].xaxis.set_label_position('top')  # 将x轴标签放置在上方
                ax2ds[i].yaxis.set_label_position('right')  # 将y轴标签放置在右边
                ax2ds[i].spines['bottom'].set_visible(False)
                ax2ds[i].spines['left'].set_visible(False)
                ax2ds[i].set_title(title_2d, fontsize=14)
                scatter_2d_cps.append(scatter_2d_cp)
                scatter_2d_pfs.append(scatter_2d_pf)
            # 添加图例，横跨三个散点图
            fig2d3d.legend(handles=[scatter_2d_cps[1], scatter_2d_pfs[1]], loc='lower center', ncol=2, columnspacing=2,
                           bbox_to_anchor=(0.5, 0), fontsize=14, frameon=False)
            fig2d3d.subplots_adjust(bottom=0.05)
            plt.savefig(f'{swat_nsga_out}\\2D_3D_Scatter_Pareto_Front_{cal_vars}.jpg', bbox_inches='tight')
            # plt.show()
        elif obj_func_num == 4:
            ### 3D Scatter
            fig2d3d = plt.figure(figsize=(12, 16), dpi=300)
            # 创建GridSpec，这里不设置width_ratios，因为宽度很难精确匹配
            gs2d3d = gridspec.GridSpec(3, 3, height_ratios=[3, 1, 1], wspace=0.35, hspace=0.2, figure=fig2d3d)
            ax3d = fig2d3d.add_subplot(gs2d3d[0, 0:4], projection='3d')
            # Compromise solution
            sc_cp = ax3d.scatter(F_comp_trans[0], F_comp_trans[1], F_comp_trans[2], c='#FFB228', s=180,
                                 edgecolors='white', label='Compromise solution', marker='*', alpha=1.0, zorder=2)
            # Pareto front solutions
            sc_pf = ax3d.scatter(F_drop_trans[:, 0], F_drop_trans[:, 1], F_drop_trans[:, 2], c=F_drop_trans[:, 3], cmap='rainbow', s=60,
                                 edgecolors='white', label='Pareto solutions', alpha=1.0, zorder=1)
            # 添加颜色条
            cbar = fig2d3d.colorbar(sc_pf, shrink=0.65, pad=0)
            cbar.set_label(label_objs[3], fontsize=14)  # 设置颜色条的标签
            # 设置颜色条刻度标签大小
            cbar.ax.tick_params(labelsize=14)  # 调整刻度标签大小
            ax3d.set_title('(a)', y=1.03, fontsize=14)
            ### 设置坐标轴标签
            ax3d.set_xlabel(label_objs[0], labelpad=12, fontsize=14)
            ax3d.set_ylabel(label_objs[1], labelpad=12, fontsize=14)
            ax3d.set_zlabel(label_objs[2], labelpad=12, fontsize=14)
            # 设置每个轴的刻度标签字体大小
            ax3d.tick_params(axis='x', pad=4, labelsize=14)
            ax3d.tick_params(axis='y', pad=4, labelsize=14)
            ax3d.tick_params(axis='z', pad=7, labelsize=14)
            ax3d.xaxis.set_major_locator(MaxNLocator(4))
            ax3d.yaxis.set_major_locator(MaxNLocator(4))
            ax3d.zaxis.set_major_locator(MaxNLocator(4))
            ax3d.view_init(elev=30, azim=30)  # Set elevation and azimuth angles
            # 设置坐标轴比例相等，以更好地显示3D效果
            ax3d.set_box_aspect((1, 1, 1))

            ### 2D Scatter
            ax2ds = [fig2d3d.add_subplot(gs2d3d[row, col]) for row in [1, 2] for col in range(3)]
            # 绘制散点图
            scatter_2d_cps, scatter_2d_pfs = [], []
            for i, (x_dim, y_dim, title_2d) in enumerate([(0, 1, '(b)'), (0, 2, '(c)'), (0, 3, '(d)'), (1, 2, '(e)'), (1, 3, '(f)'), (2, 3, '(g)')]):
                # Compromise solution
                scatter_2d_cp = ax2ds[i].scatter(F_comp_trans[x_dim], F_comp_trans[y_dim], c='#FFB228', s=180, edgecolors='white',
                                                 label='Compromise solution', alpha=1.0, marker='*', zorder=2)
                # Pareto front solutions
                scatter_2d_pf = ax2ds[i].scatter(F_drop_trans[:, x_dim], F_drop_trans[:, y_dim], c='#40B6C4', s=60, edgecolors='white',
                                                 label='Pareto solutions', alpha=1.0, zorder=1)
                # 设置子图的标题和坐标轴标签
                ax2ds[i].set_xlabel(f'{label_objs[x_dim]}', fontsize=14)
                ax2ds[i].set_ylabel(f'{label_objs[y_dim]}', fontsize=14)
                ax2ds[i].tick_params(axis='x', pad=2, labelsize=14)
                ax2ds[i].tick_params(axis='y', pad=2, labelsize=14)
                ax2ds[i].xaxis.set_major_locator(MaxNLocator(5))
                ax2ds[i].yaxis.set_major_locator(MaxNLocator(5))
                ax2ds[i].xaxis.tick_top()
                ax2ds[i].yaxis.tick_right()
                ax2ds[i].xaxis.set_label_position('top')    # 将x轴标签放置在上方
                ax2ds[i].yaxis.set_label_position('right')  # 将y轴标签放置在右边
                ax2ds[i].spines['bottom'].set_visible(False)
                ax2ds[i].spines['left'].set_visible(False)
                ax2ds[i].set_title(title_2d, fontsize=14)
                scatter_2d_cps.append(scatter_2d_cp)
                scatter_2d_pfs.append(scatter_2d_pf)
            # 添加图例，横跨三个散点图
            fig2d3d.legend(handles=[scatter_2d_cps[1], scatter_2d_pfs[1]], loc='lower center', ncol=2, columnspacing=2,
                           bbox_to_anchor=(0.5, -0.01), fontsize=14, frameon=False)
            fig2d3d.subplots_adjust(bottom=0.03)
            plt.savefig(f'{swat_nsga_out}\\2D_3D_Scatter_Pareto_Front_{cal_vars}.jpg', bbox_inches='tight')
            # plt.show()
    etime = time.time()
    print('Processing Time: {ptime} Hr!'.format(ptime=round((etime - stime) / 3600.0, 2)))
