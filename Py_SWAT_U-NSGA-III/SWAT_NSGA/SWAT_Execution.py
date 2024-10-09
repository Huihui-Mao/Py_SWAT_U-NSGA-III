# -*- coding: utf-8 -*-
"""
Created on 2023.10.29
@author: Mao Huihui
"""
import os
import sys
import time
import glob
import shutil
import rasterio
import subprocess
import numpy as np
import pandas as pd
from scipy import stats
import geopandas as gpd
from datetime import datetime
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from rasterstats import zonal_stats
from matplotlib.pyplot import MultipleLocator
from concurrent.futures import ThreadPoolExecutor


class SWAT_Run():
    def __init__(self, SWATtxtinoutFolder, pop_size_project):
        self.pop_size      = pop_size_project
        self.swat_exe_name = 'SWAT_Rev_692_64rel.exe'
        # self.swat_exe_name = 'SWAT_Rev_692_64rel_Mod.exe'
        self.swat_exe      = f'{SWATtxtinoutFolder}\\SWAT_Executable'
        self.swat_model    = f'{SWATtxtinoutFolder}\\SWAT_model'
        self.swat_TxtInOut = f'{SWATtxtinoutFolder}\\SWAT_Model\\TxtInOut'
        self.swat_nsga_in  = f'{SWATtxtinoutFolder}\\SWAT_Model\\NSGA.IN'
        self.swat_nsga_out = f'{SWATtxtinoutFolder}\\SWAT_Model\\NSGA.OUT'
        self.swat_parallel = f'{SWATtxtinoutFolder}\\SWAT_Model\\ParallelProcessing'

        # SWAT Parameters
        self.swat_parameter = []
        self.swat_para_catg = {}
        self.swat_para_prec = {}
        self.swat_para_sol  = {}
        self.swat_plant_pos = {}

        # Calibration/Validation
        self.cal_period = (2003, 2007)
        self.val_period = (2003, 2011)

        # Processing Output Files
        self.reach_num = 17
        self.hru_num   = 1589

        # Sensitivity Analysis
        self.SA_flag    = False
        self.TVSA       = False  # Time-varying sensitivity analysis
        self.half_win   = 7      # Semi-length of the moving window
        self.SA_method  = 'PAWN' # Sobol/PAWN
        self.hydro_stas = {'FenKeng':16}  # station name/reach number
        # self.hydro_stas = {'NingDu':3, 'ShiCheng':6, 'FenKeng':16}  # station name/reach number
        self.sf_obs_path   = r'*************************************'
        self.bf_obs_path   = r'*************************************'
        self.et_obs_path   = r'*************************************'
        self.lai_obs_path  = r'*************************************'
        self.rzsw_obs_path = r'*************************************'

        # Objective Functions Mode (Average_Obj_Funcs/Area_Weighted_Avg)
        self.objec_mode = 'Area_Weighted_Avg'

        # CPU Worker Number
        self.cpu_worker_num = 5
        # self.cpu_worker_num = cpu_count() if cpu_count() <= 61 else 61  # Windows
        print('cpu_worker_num:', self.cpu_worker_num)

        # Calibration Mode (Lumped/Distributed)
        self.cal_mode = 'Lumped'

        # Subbasin/HRU
        HRU_df            = pd.read_excel(f'{self.swat_nsga_in}\\FullHRU_Landuse_Area.xlsx', dtype={'HRU_GIS': str})  # 将HRU_GIS指定为字符串类型
        self.spatial_unit = 'Subbasin'
        self.HRU_shp      = r'*************************************\FullHRU\FullHRU_Zonal.shp'
        self.HRU_ID       = list(range(1, self.hru_num + 1))
        self.Subbasin_shp = r'*************************************\Watershed\Watershed_Zonal.shp'
        self.Subbasin_ID  = list(range(1, self.reach_num + 1))
        self.Plant_ID     = pd.read_excel(f'{self.swat_nsga_in}\\crop.xlsx').set_index('ICNUM')['CPNM'].to_dict()
        self.Plant_ID_re  = {v: k for k, v in self.Plant_ID.items()}
        self.HRU_ID_Veg   = HRU_df[(HRU_df['LANDUSE'] == 'FRST') | (HRU_df['LANDUSE'] == 'FRSE')]['HRU_ID'].tolist()
        print('self.HRU_ID_Veg:', len(self.HRU_ID_Veg))
        self.HRU_GIS_ID   = (HRU_df.groupby('LANDUSE')['HRU_GIS'].apply(list).to_dict())
        self.HRU_GIS_Forest = {6:HRU_df[HRU_df['LANDUSE'] == 'FRST']['HRU_GIS'].tolist(), 8:HRU_df[HRU_df['LANDUSE'] == 'FRSE']['HRU_GIS'].tolist()}
        print('self.HRU_GIS_Forest:', len(self.HRU_GIS_Forest))
        self.HRU_ID_BIOM  = pd.read_excel(f'{self.swat_nsga_in}\\FullHRU_Forest_Biomass.xlsx')['HRU_ID'].tolist()
        print('self.HRU_ID_BIOM:', len(self.HRU_ID_BIOM))
        self.HRU_ID_BIOM_Forest = {k: v['HRU_ID'].tolist() for k, v in pd.read_excel(f'{self.swat_nsga_in}\\FullHRU_Forest_Biomass.xlsx').groupby('LANDUSE')}
        print('self.HRU_ID_BIOM_Forest:', len(self.HRU_ID_BIOM_Forest))
        # Read Subbasin/HRU Area
        self.Sub_area_dict = pd.read_excel(io=f'{self.swat_nsga_in}\\Watershed_Zonal_Area.xlsx').set_index(['Subbasin'])['Area'].to_dict()
        print('self.Sub_area_dict:', len(self.Sub_area_dict))
        self.HRU_area_dict = pd.read_excel(io=f'{self.swat_nsga_in}\\FullHRU_Landuse_Area.xlsx').set_index(['HRU_ID'])['Area'].to_dict()
        print('self.HRU_area_dict:', len(self.HRU_area_dict))
        self.Area_dict = self.HRU_area_dict if self.spatial_unit == 'HRU' else self.Sub_area_dict
        self.forest_cal = False  # For Forest Only Calibration
        print('\n')

        # Format
        self.comments_len = 90

        # Calibration/Validation
        print('Calibration/Validation Setting'.center(self.comments_len, '='))
        self.water_budget = False
        print('self.water_budget:', self.water_budget) if self.water_budget == True else None
        # Compromise solution (Fixed model parameters, generally used for stepwise calibration,
        # the fixed parameter file can replace the original parameter file in the TxtInOut folder)
        self.para_fix_mode = False
        # Print Mode
        self.print_key = 'day'
        print(f'Time step: {self.print_key}')
        self.print_mode = {'month': 0, 'day': 1, 'year': 2}  # 0-month, 1-day, 2-year
        # ['Streamflow']//['LAI']//['BIOM']//['ET']//['Streamflow', 'ET']//['Streamflow', 'LAI', 'ET']//['Streamflow', 'RZSW']//['Streamflow', 'ET', 'RZSW']
        self.cal_vars_list = ['Streamflow']
        self.cal_val_state = 'Calibration'  # Calibration/Validation
        print(f'Cal period: {self.cal_period}') if self.cal_val_state == 'Calibration' else print(f'Val period: {self.val_period}')
        print(f'Runing state: {self.cal_val_state}')
        print(f'Calibration Variables: {self.cal_vars_list}')

        # Calibration Scheme (Multi-site/Multi-objective/Multi-variable/Single-objective)
        self.cal_scheme = 'Multi-objective'
        print(f'Calibration Scheme: {self.cal_scheme}')
        # Objective Functions (NSE, KGE, R2, PBIAS, RMSE)
        self.objective_funs = ['R2', 'PBIAS']
        print('self.objective_funs:', self.objective_funs)
        self.obj_func_num, self.constraint_num = self.objective_contra()
        print('self.obj_func_num:', self.obj_func_num)
        print('self.constraint_num:', self.constraint_num)

        # Spatial Unit
        print('self.spatial_unit:', self.spatial_unit)

        # Day
        self.obs_date_day_cal = pd.date_range(start=f'{self.cal_period[0]}-01-01', end=f'{self.cal_period[1]}-12-31', freq='D').strftime('%Y-%m-%d')
        self.obs_date_day_val = pd.date_range(start=f'{self.val_period[0]}-01-01', end=f'{self.val_period[1]}-12-31', freq='D').strftime('%Y-%m-%d')
        if self.print_key == 'day' and self.cal_val_state == 'Calibration':
            print('self.obs_date_day_cal:', len(self.obs_date_day_cal))
        elif self.print_key == 'day' and self.cal_val_state == 'Validation':
            print('self.obs_date_day_val:', len(self.obs_date_day_val))
        # Month
        self.obs_date_mon_cal = pd.date_range(start=f'{self.cal_period[0]}-01-01', end=f'{self.cal_period[1]}-12-31', freq='ME').strftime('%Y-%m')
        self.obs_date_mon_val = pd.date_range(start=f'{self.val_period[0]}-01-01', end=f'{self.val_period[1]}-12-31', freq='ME').strftime('%Y-%m')
        if self.print_key == 'month' and self.cal_val_state == 'Calibration':
            print('self.obs_date_mon_cal:', len(self.obs_date_mon_cal))
        elif self.print_key == 'month' and self.cal_val_state == 'Validation':
            print('self.obs_date_mon_val:', len(self.obs_date_mon_val))
        print('\n')

        # Output
        self.n_gen = 1
        self.para_obj_val_dict = {}

        # Step 1: Read SWAT Parameters
        print('Step 1: Read SWAT Parameters'.center(self.comments_len, '='))
        self.read_swat_para()
        print('Read SWAT parameters finished!')
        print('\n')


        # Step 2: Modify file.cio/Copy TxtInOut to Working Dir
        modify_cio_start_time = time.time()
        print('Step 2: Modify file.cio/Copy TxtInOut to Working Dir'.center(self.comments_len, '='))
        self.NYSKIP = 2  # Warm‐up years
        mod_val_dict = {}
        ICALEN = 1
        if self.print_key == 'day':
            ICALEN = 1
        elif self.print_key == 'month' or self.print_key == 'year':
            ICALEN = 0
        if self.cal_val_state == 'Calibration':
            mod_val_dict = {'NBYR': self.cal_period[1] - self.cal_period[0] + self.NYSKIP + 1,
                            'IYR': self.cal_period[0] - self.NYSKIP,
                            'IPRINT': self.print_mode[self.print_key],
                            'NYSKIP': self.NYSKIP,
                            'ICALEN': ICALEN}
        elif self.cal_val_state == 'Validation':
            mod_val_dict = {'NBYR': self.val_period[1] - self.val_period[0] + self.NYSKIP + 1,
                            'IYR': self.val_period[0] - self.NYSKIP,
                            'IPRINT': self.print_mode[self.print_key],
                            'NYSKIP': self.NYSKIP,
                            'ICALEN': ICALEN}
        # Output Variables Print Dict
        self.output_var_print = {0: 'Print Only Calibration Variables',
                                 1: 'Print Main Output Variables',
                                 2: 'Print All Output Variables'}
        # Output Variables Print Mode
        out_var_flag = self.output_var_print[0]
        print('Output Variables Print Mode:', out_var_flag)
        if self.para_fix_mode:
            print('para_fix_mode:')
            if self.water_budget:
                out_var_flag = self.output_var_print[1]
            print('out_var_flag:', out_var_flag)
            args = self.swat_TxtInOut, mod_val_dict, out_var_flag
            self.modify_cio_file(args)
            print('Modify file.cio file finished!')

            # Delete SWAT Executable File if already exist
            for exe_file in glob.glob(f'{self.swat_TxtInOut}\\*.exe'):
                os.remove(f'{exe_file}')

            # Copy SWAT Executable File to Backup Folder
            shutil.copy(f'{self.swat_exe}\\{self.swat_exe_name}', f'{self.swat_TxtInOut}\\{self.swat_exe_name}')
        elif self.cal_val_state == 'Validation':
            print('Validation:')
            if self.water_budget:
                out_var_flag = self.output_var_print[1]
            print('out_var_flag:', out_var_flag)
            args = self.swat_TxtInOut, mod_val_dict, out_var_flag
            self.modify_cio_file(args)
            print('Modify file.cio file finished!')

            # Delete SWAT Executable File if already exist
            for exe_file in glob.glob(f'{self.swat_TxtInOut}\\*.exe'):
                os.remove(f'{exe_file}')

            # Copy SWAT Executable File to Backup Folder
            shutil.copy(f'{self.swat_exe}\\{self.swat_exe_name}', f'{self.swat_TxtInOut}\\{self.swat_exe_name}')
        elif sum(1 for file in os.scandir(self.swat_parallel) if file.is_dir()) == 0:
            print('Folder is empty!')
            args = self.swat_TxtInOut, mod_val_dict, out_var_flag
            self.modify_cio_file(args)
            print('Modify file.cio file finished!')

            # Delete SWAT Executable File if already exist
            for exe_file in glob.glob(f'{self.swat_TxtInOut}\\*.exe'):
                os.remove(f'{exe_file}')

            # Copy SWAT Executable File to Backup Folder
            shutil.copy(f'{self.swat_exe}\\{self.swat_exe_name}', f'{self.swat_TxtInOut}\\{self.swat_exe_name}')
            # Multiprocessing
            print('Copy SWAT project with population size using multiprocessing:')
            process_args = [(self.swat_TxtInOut, f'{self.swat_parallel}\\Population_{pop_idx}') for pop_idx in range(1, self.pop_size + 1)]
            with Pool(processes=61) as p:
                p.starmap(shutil.copytree, process_args)
            print('Copy file finished!')
        elif ((0 < sum(1 for file in os.scandir(self.swat_parallel) if file.is_dir()) < self.pop_size) or
              (sum(1 for file in os.scandir(self.swat_parallel) if file.is_dir()) > self.pop_size)):
            print('Folder is not empty, but the number and the population size do not match!')
            args = self.swat_TxtInOut, mod_val_dict, out_var_flag
            self.modify_cio_file(args)
            print('Modify file.cio file finished!')

            # Remove folders
            shutil.rmtree(self.swat_parallel)
            # Delete SWAT Executable File if already exist
            for exe_file in glob.glob(f'{self.swat_TxtInOut}\\*.exe'):
                os.remove(f'{exe_file}')

            # Copy SWAT Executable File to Backup Folder
            shutil.copy(f'{self.swat_exe}\\{self.swat_exe_name}', f'{self.swat_TxtInOut}\\{self.swat_exe_name}')
            # Multiprocessing
            print('Copy SWAT project with population size using multiprocessing:')
            process_args = [(self.swat_TxtInOut, f'{self.swat_parallel}\\Population_{pop_idx}') for pop_idx in range(1, self.pop_size + 1)]
            with Pool(self.cpu_worker_num) as p:
                p.starmap(shutil.copytree, process_args)
            print('Copy file finished!')
        else:
            print('Population folder already exist!')
            # This part was added in case modification of the SWAT executable file
            for pop_path_idx in [f'{self.swat_parallel}\\Population_{pop_idx}' for pop_idx in range(1, self.pop_size + 1)]:
                # print(os.path.basename(pop_path_idx))
                # Delete SWAT Executable File if already exist
                for exe_file in glob.glob(f'{pop_path_idx}\\*.exe'):
                    os.remove(f'{exe_file}')
                # Copy SWAT Executable File to Backup Folder
                shutil.copy(f'{self.swat_exe}\\{self.swat_exe_name}', f'{pop_path_idx}\\{self.swat_exe_name}')

            ## Multithreading
            thread_args = [(f'{self.swat_parallel}\\{pop_idx}', mod_val_dict, out_var_flag) for pop_idx in os.listdir(f'{self.swat_parallel}')]
            with ThreadPoolExecutor(max_workers=len(thread_args)) as pool:
                pool.map(self.modify_cio_file, thread_args)
            print('Modify file.cio file finished!')
        modify_cio_end_time = time.time()
        print(f'Modify/Copy Time: {round((modify_cio_end_time - modify_cio_start_time) / 60.0, 2)} min!')
        print('\n')


        # Step 3: Read Observation Data
        print('Step 3: Read Observation Data'.center(self.comments_len, '='))
        read_obs_flag = True
        if read_obs_flag:
            read_obs_start_time = time.time()
            # self.HRU_veg_area_dict is the area of all forest type HRUs containing only FRST and FRSE
            self.HRU_veg_area_dict  = {HRU_veg_idx:self.HRU_area_dict[HRU_veg_idx] for HRU_veg_idx in self.HRU_ID_Veg}
            self.HRU_BIOM_area_dict = {HRU_BIOM_idx:self.HRU_area_dict[HRU_BIOM_idx] for HRU_BIOM_idx in self.HRU_ID_BIOM}
            self.HRU_BIOM_area_dict_FRST = {HRU_BIOM_idx: self.HRU_area_dict[HRU_BIOM_idx] for HRU_BIOM_idx in self.HRU_ID_BIOM_Forest['FRST']}
            self.HRU_BIOM_area_dict_FRSE = {HRU_BIOM_idx: self.HRU_area_dict[HRU_BIOM_idx] for HRU_BIOM_idx in self.HRU_ID_BIOM_Forest['FRSE']}
            # One Variables
            if self.cal_vars_list == ['Streamflow']:
                self.obs_sf_data = self.read_obs_streamflow_data()
                print('self.obs_sf_data:', len(self.obs_sf_data), self.obs_sf_data)
            elif self.cal_vars_list == ['LAI']:
                self.obs_lai_data = self.read_obs_lai_data()
                print('self.obs_lai_data:', len(self.obs_lai_data))
                if self.objec_mode == 'Area_Weighted_Avg':
                    if self.forest_cal:
                        # LAI observation
                        LAI_obs_data_dict = {k: v for k, v in filter(lambda x: x[0] in self.HRU_ID_Veg, self.obs_lai_data.items())}
                        # print('LAI_obs_data_dict:', len(LAI_obs_data_dict))
                        self.LAI_obs_data_dict_area_w = np.zeros(shape=len(list(LAI_obs_data_dict.values())[0]))
                        for LAI_HRU, LAI_val in LAI_obs_data_dict.items():
                            self.LAI_obs_data_dict_area_w += LAI_val * (self.HRU_veg_area_dict[LAI_HRU] / sum(list(self.HRU_veg_area_dict.values())))
                        print('self.LAI_obs_data_dict_area_w:', self.LAI_obs_data_dict_area_w.shape)
                    else:
                        # LAI observation
                        LAI_obs_data_dict = self.obs_lai_data
                        # print('LAI_obs_data_dict:', len(LAI_obs_data_dict))
                        self.LAI_obs_data_dict_area_w = np.zeros(shape=len(list(LAI_obs_data_dict.values())[0]))
                        for LAI_HRU, LAI_val in LAI_obs_data_dict.items():
                            self.LAI_obs_data_dict_area_w += LAI_val * (self.Area_dict[LAI_HRU] / sum(list(self.Area_dict.values())))
                        print('self.LAI_obs_data_dict_area_w:', self.LAI_obs_data_dict_area_w.shape)
            elif self.cal_vars_list == ['BIOM']:
                self.obs_biom_data = self.read_obs_biom_data()
                print('self.obs_biom_data:', len(self.obs_biom_data))
                if self.objec_mode == 'Area_Weighted_Avg':
                    if self.cal_val_state == 'Calibration':
                        # BIOM observation
                        BIOM_obs_data_dict = {k: v for k, v in filter(lambda x: x[0] in self.HRU_ID_BIOM, self.obs_biom_data.items())}
                        # print('BIOM_obs_data_dict:', len(BIOM_obs_data_dict))
                        self.BIOM_obs_data_dict_area_w = np.zeros(shape=len(list(BIOM_obs_data_dict.values())[0]))
                        for BIOM_HRU, BIOM_val in BIOM_obs_data_dict.items():
                            self.BIOM_obs_data_dict_area_w += BIOM_val * (self.HRU_BIOM_area_dict[BIOM_HRU] / sum(list(self.HRU_BIOM_area_dict.values())))
                        print('self.BIOM_obs_data_dict_area_w:', self.BIOM_obs_data_dict_area_w.shape)
                    elif self.cal_val_state == 'Validation':
                        # BIOM observation FRST/FRSE
                        BIOM_obs_data_dict_FRST = {k: v for k, v in filter(lambda x: x[0] in self.HRU_ID_BIOM_Forest['FRST'], self.obs_biom_data.items())}
                        BIOM_obs_data_dict_FRSE = {k: v for k, v in filter(lambda x: x[0] in self.HRU_ID_BIOM_Forest['FRSE'], self.obs_biom_data.items())}
                        print('BIOM_obs_data_dict_FRST:', len(BIOM_obs_data_dict_FRST))
                        print('BIOM_obs_data_dict_FRSE:', len(BIOM_obs_data_dict_FRSE))
                        self.BIOM_obs_data_dict_area_w_FRST = np.zeros(shape=len(list(BIOM_obs_data_dict_FRST.values())[0]))
                        self.BIOM_obs_data_dict_area_w_FRSE = np.zeros(shape=len(list(BIOM_obs_data_dict_FRSE.values())[0]))
                        # FRST
                        for BIOM_HRU_FRST, BIOM_val_FRST in BIOM_obs_data_dict_FRST.items():
                            self.BIOM_obs_data_dict_area_w_FRST += (
                                    BIOM_val_FRST * (self.HRU_BIOM_area_dict_FRST[BIOM_HRU_FRST] / sum(list(self.HRU_BIOM_area_dict_FRST.values()))))
                        print('self.BIOM_obs_data_dict_area_w_FRST:', self.BIOM_obs_data_dict_area_w_FRST.shape, self.BIOM_obs_data_dict_area_w_FRST)
                        # FRSE
                        for BIOM_HRU_FRSE, BIOM_val_FRSE in BIOM_obs_data_dict_FRSE.items():
                            self.BIOM_obs_data_dict_area_w_FRSE += (
                                    BIOM_val_FRSE * (self.HRU_BIOM_area_dict_FRSE[BIOM_HRU_FRSE] / sum(list(self.HRU_BIOM_area_dict_FRSE.values()))))
                        print('self.BIOM_obs_data_dict_area_w_FRSE:', self.BIOM_obs_data_dict_area_w_FRSE.shape, self.BIOM_obs_data_dict_area_w_FRSE)
            elif self.cal_vars_list == ['ET']:
                self.obs_et_data = self.read_obs_et_data()
                print('self.obs_et_data:', len(self.obs_et_data))
                if self.objec_mode == 'Area_Weighted_Avg':
                    if self.forest_cal:
                        # ET observation
                        ET_obs_data_dict = {k: v for k, v in filter(lambda x: x[0] in self.HRU_ID_Veg, self.obs_et_data.items())}
                        # print('ET_obs_data_dict:', len(ET_obs_data_dict))
                        self.ET_obs_data_dict_area_w = np.zeros(shape=len(list(ET_obs_data_dict.values())[0]))
                        for ET_HRU, ET_val in ET_obs_data_dict.items():
                            self.ET_obs_data_dict_area_w += ET_val * (self.HRU_veg_area_dict[ET_HRU] / sum(list(self.HRU_veg_area_dict.values())))
                        print('self.ET_obs_data_dict_area_weight:', self.ET_obs_data_dict_area_w.shape)
                    else:
                        # ET observation
                        ET_obs_data_dict = self.obs_et_data
                        # print('ET_obs_data_dict:', len(ET_obs_data_dict))
                        self.ET_obs_data_dict_area_w = np.zeros(shape=len(list(ET_obs_data_dict.values())[0]))
                        for ET_Sp_ID, ET_val in ET_obs_data_dict.items():
                            self.ET_obs_data_dict_area_w += ET_val * (self.Area_dict[ET_Sp_ID] / sum(list(self.Area_dict.values())))
                        print('self.ET_obs_data_dict_area_w:', self.ET_obs_data_dict_area_w.shape)

            # Two Variables
            elif self.cal_vars_list == ['Streamflow', 'ET']:
                self.obs_sf_data = self.read_obs_streamflow_data()
                self.obs_et_data = self.read_obs_et_data()
                print('self.obs_sf_data:', len(self.obs_sf_data))
                print('self.obs_et_data:', len(self.obs_et_data))
                if self.objec_mode == 'Area_Weighted_Avg':
                    if self.forest_cal:
                        # ET observation
                        ET_obs_data_dict = {k: v for k, v in filter(lambda x: x[0] in self.HRU_ID_Veg, self.obs_et_data.items())}
                        # print('ET_obs_data_dict:', len(ET_obs_data_dict))
                        self.ET_obs_data_dict_area_w = np.zeros(shape=len(list(ET_obs_data_dict.values())[0]))
                        for ET_HRU, ET_val in ET_obs_data_dict.items():
                            self.ET_obs_data_dict_area_w += ET_val * (self.HRU_veg_area_dict[ET_HRU] / sum(list(self.HRU_veg_area_dict.values())))
                        print('self.ET_obs_data_dict_area_weight:', self.ET_obs_data_dict_area_w.shape)
                    else:
                        # ET observation
                        ET_obs_data_dict = self.obs_et_data
                        # print('ET_obs_data_dict:', len(ET_obs_data_dict))
                        self.ET_obs_data_dict_area_w = np.zeros(shape=len(list(ET_obs_data_dict.values())[0]))
                        for ET_Sp_ID, ET_val in ET_obs_data_dict.items():
                            self.ET_obs_data_dict_area_w += ET_val * (self.Area_dict[ET_Sp_ID] / sum(list(self.Area_dict.values())))
                        print('self.ET_obs_data_dict_area_w:', self.ET_obs_data_dict_area_w.shape)
            elif self.cal_vars_list == ['Streamflow', 'RZSW']:
                self.obs_sf_data   = self.read_obs_streamflow_data()
                self.obs_rzsw_data = self.read_obs_rzsw_data()
                print('self.obs_sf_data:', len(self.obs_sf_data))
                print('self.obs_rzsw_data:', len(self.obs_rzsw_data))

            # Three Variables
            elif self.cal_vars_list == ['Streamflow', 'LAI', 'ET']:
                self.obs_sf_data  = self.read_obs_streamflow_data()
                self.obs_lai_data = self.read_obs_lai_data()
                self.obs_et_data  = self.read_obs_et_data()
                print('self.obs_sf_data:', len(self.obs_sf_data))
                print('self.obs_lai_data:', len(self.obs_lai_data))
                print('self.obs_et_data:', len(self.obs_et_data))
                if self.objec_mode == 'Area_Weighted_Avg':
                    if self.forest_cal:
                        # LAI observation
                        LAI_obs_data_dict = {k: v for k, v in filter(lambda x: x[0] in self.HRU_ID_Veg, self.obs_lai_data.items())}
                        # print('LAI_obs_data_dict:', len(LAI_obs_data_dict))
                        self.LAI_obs_data_dict_area_w = np.zeros(shape=len(list(LAI_obs_data_dict.values())[0]))
                        for LAI_HRU, LAI_val in LAI_obs_data_dict.items():
                            self.LAI_obs_data_dict_area_w += LAI_val * (self.HRU_veg_area_dict[LAI_HRU] / sum(list(self.HRU_veg_area_dict.values())))
                        print('self.LAI_obs_data_dict_area_w:', self.LAI_obs_data_dict_area_w.shape)

                        # ET observation
                        ET_obs_data_dict = {k: v for k, v in filter(lambda x: x[0] in self.HRU_ID_Veg, self.obs_et_data.items())}
                        # print('ET_obs_data_dict:', len(ET_obs_data_dict))
                        self.ET_obs_data_dict_area_w = np.zeros(shape=len(list(ET_obs_data_dict.values())[0]))
                        for ET_HRU, ET_val in ET_obs_data_dict.items():
                            self.ET_obs_data_dict_area_w += ET_val * (self.HRU_veg_area_dict[ET_HRU] / sum(list(self.HRU_veg_area_dict.values())))
                        print('self.ET_obs_data_dict_area_weight:', self.ET_obs_data_dict_area_w.shape)
                    else:
                        # LAI observation
                        LAI_obs_data_dict = self.obs_lai_data
                        # print('LAI_obs_data_dict:', len(LAI_obs_data_dict))
                        self.LAI_obs_data_dict_area_w = np.zeros(shape=len(list(LAI_obs_data_dict.values())[0]))
                        for LAI_HRU, LAI_val in LAI_obs_data_dict.items():
                            self.LAI_obs_data_dict_area_w += LAI_val * (self.Area_dict[LAI_HRU] / sum(list(self.Area_dict.values())))
                        print('self.LAI_obs_data_dict_area_w:', self.LAI_obs_data_dict_area_w.shape)

                        # ET observation
                        ET_obs_data_dict = self.obs_et_data
                        # print('ET_obs_data_dict:', len(ET_obs_data_dict))
                        self.ET_obs_data_dict_area_w = np.zeros(shape=len(list(ET_obs_data_dict.values())[0]))
                        for ET_HRU, ET_val in ET_obs_data_dict.items():
                            self.ET_obs_data_dict_area_w += ET_val * (self.Area_dict[ET_HRU] / sum(list(self.Area_dict.values())))
                        print('self.ET_obs_data_dict_area_w:', self.ET_obs_data_dict_area_w.shape)

            elif self.cal_vars_list == ['Streamflow', 'ET', 'RZSW']:
                self.obs_sf_data   = self.read_obs_streamflow_data()
                self.obs_et_data   = self.read_obs_et_data()
                self.obs_rzsw_data = self.read_obs_rzsw_data()
                print('self.obs_sf_data:', len(self.obs_sf_data))
                print('self.obs_et_data:', len(self.obs_et_data))
                print('self.obs_rzsw_data:', len(self.obs_rzsw_data))
            read_obs_end_time = time.time()
            print(f'Read Observation Time: {round((read_obs_end_time - read_obs_start_time) / 60.0, 2)} min!')
            print('\n')


    def SWAT_model_execution(self, SWATParas_Sampling):
        # Suspend for a period of time to prevent the hard disk from continuously reading or writing data
        time.sleep(5)
        print('\n\n' + '*' * 90)
        print('Generation:', self.n_gen)
        print('\n')

        # Step 4: Restore Files from TxtInOut
        print('Step 4: Restore Files from TxtInOut'.center(self.comments_len, '='))
        restore_flag = False if self.cal_val_state == 'Validation' else True
        if restore_flag:
            ## Filter the Parameter Files Corresponding to All the Parameters that Vary in 'r' and 'a'
            restore_start_time = time.time()
            ## 1) Filter Out All Suffixes that Change Parameters in 'r' and 'a'
            rpara_suff_list = []
            for filter_idx in list(filter(lambda x: (x[2] == 'r') or (x[2] == 'a'), self.swat_parameter)):
                rpara_suff_list.append(filter_idx[3])
            rpara_suff_list = list(set(rpara_suff_list))
            print('rpara_suff_list:', len(rpara_suff_list), rpara_suff_list)

            ## 2) Search for All Parameter Files that Need to be Updated by Their Suffixes
            rpara_file_list = []
            for rpara_suff_idx in rpara_suff_list:
                rpara_file_list.extend(self.search_para_files(self.swat_TxtInOut, '?????????', rpara_suff_idx))
            print('rpara_file_list:', len(rpara_file_list))

            ## 3) Updating the ParallelProcessing Catalog
            # Multiprocessing
            process_args = [(rpara_file_list, f'{self.swat_parallel}\\{pop_idx}') for pop_idx in os.listdir(f'{self.swat_parallel}')]
            print('process_args:', len(process_args))
            with Pool(processes=20) as p:
                p.starmap(self.copy_to_folder, process_args)
            print('Parameter Files Updated!')
            restore_end_time = time.time()
            print(f'Restore Time: {round((restore_end_time - restore_start_time) / 60.0, 2)} min!')
        print('\n')


        # Step 5: Modify SWAT Project Parameter Files
        print('Step 5: Modify SWAT Project Parameter Files'.center(self.comments_len, '='))
        print('SWATParas_Sampling:', SWATParas_Sampling.shape)
        ## For Sensitivity Analysis/Compromise solution
        mod_flag = True
        if mod_flag:
            modify_start_time = time.time()
            if SWATParas_Sampling.shape[0] != self.pop_size:
                if self.SA_flag:
                    self.pop_size = SWATParas_Sampling.shape[0]
                    print('self.pop_size-SA:', self.pop_size)
                    print('\n')
                else:
                    if self.cal_val_state == 'Validation' and SWATParas_Sampling.ndim == 1:
                        self.pop_size = 1
                        print('self.pop_size-Val:', self.pop_size)
                        print('\n')
                    elif self.cal_val_state == 'Validation' and SWATParas_Sampling.ndim == 2:
                        self.pop_size = SWATParas_Sampling.shape[0]
                        print('self.pop_size-Val:', self.pop_size)
                        print('\n')
            ## Read SWAT Parameters' Latin Hypercube Sampling
            para_sampling_dict = {}
            if self.para_fix_mode or (self.cal_val_state == 'Validation' and SWATParas_Sampling.ndim == 1):
                for num_idx, para_idx in enumerate(self.swat_parameter):
                    para_sampling_dict[para_idx[0]] = [para_idx[2], SWATParas_Sampling[num_idx]]
            else:
                for num_idx, para_idx in enumerate(self.swat_parameter):
                    para_sampling_dict[para_idx[0]] = [para_idx[2], SWATParas_Sampling[:, num_idx].tolist()]
            print('para_sampling_dict:', len(para_sampling_dict))

            ## Multiprocessing
            print('Modify SWAT parameters using multiprocessing:')
            if self.para_fix_mode:
                self.modify_SWAT_para_multiprocessing_v2(1, f'{self.swat_model}\\Compro_sol', para_sampling_dict)
            elif self.cal_val_state == 'Validation':
                process_args = [(pop_idx, f'{self.swat_model}\\Validation_Pareto_Sol\\Population_{pop_idx}', para_sampling_dict)
                                for pop_idx in range(1, self.pop_size + 1)]
                print('process_args:', len(process_args))
                with Pool(processes=self.cpu_worker_num) as p:
                    p.starmap(self.modify_SWAT_para_multiprocessing_v2, process_args)
            else:
                process_args = [(pop_idx, f'{self.swat_parallel}\\Population_{pop_idx}', para_sampling_dict) for pop_idx in range(1, self.pop_size + 1)]
                print('process_args:', len(process_args))
                with Pool(processes=self.cpu_worker_num) as p:
                    p.starmap(self.modify_SWAT_para_multiprocessing_v2, process_args)
            print('Modify SWAT parameters finished!')
            modify_end_time = time.time()
            print(f'Modify Time: {round((modify_end_time - modify_start_time) / 60.0, 2)} min!')
        # Suspend for a period of time to prevent the hard disk from continuously reading or writing data
        time.sleep(5)
        print('\n')


        # Step 6: SWAT Model Execution
        print('Step 6: SWAT Model Execution'.center(self.comments_len, '='))
        run_flag = True
        if run_flag:
            run_start_time = time.time()
            ## Delete Population_x.txt files
            if self.cal_val_state == 'Validation':
                for pop_idx in range(1, self.pop_size + 1):
                    if os.path.exists(f'{self.swat_model}\\Validation_Pareto_Sol\\Population_{pop_idx}\\Population_{pop_idx}.txt'):
                        os.remove(f'{self.swat_model}\\Validation_Pareto_Sol\\Population_{pop_idx}\\Population_{pop_idx}.txt')
            else:
                for pop_idx in range(1, self.pop_size + 1):
                    if os.path.exists(f'{self.swat_parallel}\\Population_{pop_idx}\\Population_{pop_idx}.txt'):
                        os.remove(f'{self.swat_parallel}\\Population_{pop_idx}\\Population_{pop_idx}.txt')
            ## Multiprocessing
            print('Run SWAT model using multiprocessing:')
            print_infor  = 'Pop_folder'  # Curr_screen/Pop_folder
            if self.para_fix_mode:
                self.run_SWAT_model([1, f'{self.swat_model}\\Compro_sol', self.swat_exe_name, print_infor])
            elif self.cal_val_state == 'Validation':
                process_args = [[pop_idx, f'{self.swat_model}\\Validation_Pareto_Sol\\Population_{pop_idx}', self.swat_exe_name, print_infor]
                                for pop_idx in range(1, self.pop_size + 1)]
                with Pool(processes=self.cpu_worker_num) as p:
                    p.map(self.run_SWAT_model, process_args)
            else:
                process_args = [[pop_idx, f'{self.swat_parallel}\\Population_{pop_idx}', self.swat_exe_name, print_infor]
                                for pop_idx in range(1, self.pop_size + 1)]
                with Pool(processes=self.cpu_worker_num) as p:
                    p.map(self.run_SWAT_model, process_args)
            print('Run SWAT model finished!')
            run_end_time = time.time()
            print(f'Run Time: {round((run_end_time - run_start_time) / 60.0, 2)} min!')
        sys.exit() if self.para_fix_mode == True else None
        # Suspend for a period of time to prevent the hard disk from continuously reading or writing data
        time.sleep(5)
        print('\n')


        # Step 7: Read SWAT Model Simulation Data
        read_sim_start_time = time.time()
        print('Step 7: Read SWAT Model Simulation Data'.center(self.comments_len, '='))
        HRU_data_dict_PopList_sort = None
        rch_data_dict_PopList_sort, LAI_data_dict_PopList_sort, BIOM_data_dict_PopList_sort = None, None, None
        ET_HRU_data_dict_PopList_sort, ET_Sub_data_dict_PopList_sort = None, None
        SW_HRU_data_dict_PopList_sort, SW_Sub_data_dict_PopList_sort = None, None
        ET_SW_HRU_data_dict_PopList_sort, ET_SW_Sub_data_dict_PopList_sort, LAI_ET_HRU_data_dict_PopList_sort = None, None, None
        read_sim_flag = True
        if read_sim_flag:
            ## Multiprocessing
            print('Read SWAT simulation data with multiprocessing:')
            # 创建一个包含pop_size条线程的进程池
            rch_thread_args = [(pop_idx, f'{self.swat_parallel}\\Population_{pop_idx}\\output.rch') for pop_idx in range(1, self.pop_size + 1)]
            hru_thread_args = [(pop_idx, f'{self.swat_parallel}\\Population_{pop_idx}\\output.hru') for pop_idx in range(1, self.pop_size + 1)]
            sub_thread_args = [(pop_idx, f'{self.swat_parallel}\\Population_{pop_idx}\\output.sub') for pop_idx in range(1, self.pop_size + 1)]
            if self.cal_val_state == 'Validation':
                rch_thread_args = [(pop_idx, f'{self.swat_model}\\Validation_Pareto_Sol\\Population_{pop_idx}\\output.rch')
                                   for pop_idx in range(1, self.pop_size + 1)]
                hru_thread_args = [(pop_idx, f'{self.swat_model}\\Validation_Pareto_Sol\\Population_{pop_idx}\\output.hru')
                                   for pop_idx in range(1, self.pop_size + 1)]
                sub_thread_args = [(pop_idx, f'{self.swat_model}\\Validation_Pareto_Sol\\Population_{pop_idx}\\output.sub')
                                   for pop_idx in range(1, self.pop_size + 1)]
            # One Variables
            if self.cal_vars_list == ['Streamflow']:
                # output.rch
                with Pool(processes=self.cpu_worker_num) as p:
                    rch_data_dict_PopList = p.map(self.read_rch_sim_data, rch_thread_args)
                rch_data_dict_PopList_sort = sorted(rch_data_dict_PopList, key=lambda x: x[0])
                print('rch_data_dict_PopList_sort:', len(rch_data_dict_PopList_sort))
            elif self.cal_vars_list == ['LAI']:
                # output.hru
                with Pool(processes=self.cpu_worker_num) as p:
                    LAI_data_dict_PopList = p.map(self.read_hru_sim_data, hru_thread_args)
                LAI_data_dict_PopList_sort = [(lai_dict_idx[0], lai_dict_idx[1][3]) for lai_dict_idx in sorted(LAI_data_dict_PopList, key=lambda x:x[0])]
                print('LAI_data_dict_PopList_sort:', len(LAI_data_dict_PopList_sort))
            elif self.cal_vars_list == ['BIOM']:
                # output.hru
                with Pool(processes=self.cpu_worker_num) as p:
                    BIOM_data_dict_PopList = p.map(self.read_hru_sim_data, hru_thread_args)
                BIOM_data_dict_PopList_sort = [(biom_dict_idx[0], biom_dict_idx[1][2]) for biom_dict_idx in sorted(BIOM_data_dict_PopList, key=lambda x:x[0])]
                print('BIOM_data_dict_PopList_sort:', len(BIOM_data_dict_PopList_sort))
            elif self.cal_vars_list == ['ET']:
                if self.spatial_unit == 'HRU':
                    # output.hru
                    with Pool(processes=self.cpu_worker_num) as p:
                        ET_HRU_data_dict_PopList = p.map(self.read_hru_sim_data, hru_thread_args)
                    ET_HRU_data_dict_PopList_sort = [(et_hru_dict_idx[0], et_hru_dict_idx[1][0])
                                                     for et_hru_dict_idx in sorted(ET_HRU_data_dict_PopList, key=lambda x:x[0])]
                    print('ET_HRU_data_dict_PopList_sort:', len(ET_HRU_data_dict_PopList_sort))
                elif self.spatial_unit == 'Subbasin':
                    # output.sub
                    with Pool(processes=self.cpu_worker_num) as p:
                        ET_Sub_data_dict_PopList = p.map(self.read_sub_sim_data, sub_thread_args)
                    ET_Sub_data_dict_PopList_sort = [(et_sub_dict_idx[0], et_sub_dict_idx[1][0])
                                                     for et_sub_dict_idx in sorted(ET_Sub_data_dict_PopList, key=lambda x: x[0])]
                    print('ET_Sub_data_dict_PopList_sort:', len(ET_Sub_data_dict_PopList_sort))

            # Two Variables
            elif self.cal_vars_list == ['Streamflow', 'ET']:
                # output.rch
                with Pool(processes=self.cpu_worker_num) as p:
                    rch_data_dict_PopList = p.map(self.read_rch_sim_data, rch_thread_args)
                rch_data_dict_PopList_sort = sorted(rch_data_dict_PopList, key=lambda x: x[0])
                print('rch_data_dict_PopList_sort:', len(rch_data_dict_PopList_sort))
                if self.spatial_unit == 'HRU':
                    # output.hru
                    with Pool(processes=50) as p:
                        ET_HRU_data_dict_PopList = p.map(self.read_hru_sim_data, hru_thread_args)
                    if self.water_budget:
                        # pop_idx, [ET (0), SW (1), PRECIP (2), SURQ (3), LATQ (4), GWQ (5), BIOM (6), LAI (7)]
                        HRU_data_dict_PopList_sort = [(et_hru_dict_idx[0], et_hru_dict_idx[1][0], et_hru_dict_idx[1][2], et_hru_dict_idx[1][3],
                                                       et_hru_dict_idx[1][4], et_hru_dict_idx[1][5])
                                                      for et_hru_dict_idx in sorted(ET_HRU_data_dict_PopList, key=lambda x: x[0])]
                        print('HRU_data_dict_PopList_sort:', len(HRU_data_dict_PopList_sort))
                    else:
                        ET_HRU_data_dict_PopList_sort = [(et_hru_dict_idx[0], et_hru_dict_idx[1][0])
                                                         for et_hru_dict_idx in sorted(ET_HRU_data_dict_PopList, key=lambda x: x[0])]
                        print('ET_HRU_data_dict_PopList_sort:', len(ET_HRU_data_dict_PopList_sort))
                elif self.spatial_unit == 'Subbasin':
                    # output.sub
                    with Pool(processes=self.cpu_worker_num) as p:
                        ET_Sub_data_dict_PopList = p.map(self.read_sub_sim_data, sub_thread_args)
                    ET_Sub_data_dict_PopList_sort = [(et_sub_dict_idx[0], et_sub_dict_idx[1][0])
                                                     for et_sub_dict_idx in sorted(ET_Sub_data_dict_PopList, key=lambda x: x[0])]
                    print('ET_Sub_data_dict_PopList_sort:', len(ET_Sub_data_dict_PopList_sort))
            elif self.cal_vars_list == ['Streamflow', 'RZSW']:
                # output.rch
                with Pool(processes=self.cpu_worker_num) as p:
                    rch_data_dict_PopList = p.map(self.read_rch_sim_data, rch_thread_args)
                rch_data_dict_PopList_sort = sorted(rch_data_dict_PopList, key=lambda x: x[0])
                print('rch_data_dict_PopList_sort:', len(rch_data_dict_PopList_sort))
                if self.spatial_unit == 'HRU':
                    # output.hru
                    with Pool(processes=self.cpu_worker_num) as p:
                        SW_HRU_data_dict_PopList = p.map(self.read_hru_sim_data, hru_thread_args)
                    SW_HRU_data_dict_PopList_sort = [(sw_hru_dict_idx[0], sw_hru_dict_idx[1][1])
                                                     for sw_hru_dict_idx in sorted(SW_HRU_data_dict_PopList, key=lambda x: x[0])]
                    print('SW_HRU_data_dict_PopList_sort:', len(SW_HRU_data_dict_PopList_sort))
                elif self.spatial_unit == 'Subbasin':
                    # output.sub
                    with Pool(processes=self.cpu_worker_num) as p:
                        SW_Sub_data_dict_PopList = p.map(self.read_sub_sim_data, sub_thread_args)
                    SW_Sub_data_dict_PopList_sort = [(sw_sub_dict_idx[0], sw_sub_dict_idx[1][1])
                                                     for sw_sub_dict_idx in sorted(SW_Sub_data_dict_PopList, key=lambda x: x[0])]
                    print('SW_Sub_data_dict_PopList_sort:', len(SW_Sub_data_dict_PopList_sort))

            # Three Variables
            elif self.cal_vars_list == ['Streamflow', 'LAI', 'ET']:
                # output.rch
                with Pool(processes=self.cpu_worker_num) as p:
                    rch_data_dict_PopList = p.map(self.read_rch_sim_data, rch_thread_args)
                rch_data_dict_PopList_sort = sorted(rch_data_dict_PopList, key=lambda x: x[0])
                print('rch_data_dict_PopList_sort:', len(rch_data_dict_PopList_sort))
                if self.spatial_unit == 'HRU':
                    # output.hru
                    with Pool(processes=10) as p:
                        LAI_ET_HRU_data_dict_PopList = p.map(self.read_hru_sim_data, hru_thread_args)
                    # pop_idx, [ET_data_dict, SW_data_dict, BIOM_data_dict, LAI_data_dict]
                    LAI_ET_HRU_data_dict_PopList_sort = [(lai_et_hru_dict_idx[0], lai_et_hru_dict_idx[1][3], lai_et_hru_dict_idx[1][0])
                                                         for lai_et_hru_dict_idx in sorted(LAI_ET_HRU_data_dict_PopList, key=lambda x: x[0])]
                    print('LAI_ET_HRU_data_dict_PopList_sort:', len(LAI_ET_HRU_data_dict_PopList_sort))
            elif self.cal_vars_list == ['Streamflow', 'ET', 'RZSW']:
                # output.rch
                with Pool(processes=self.cpu_worker_num) as p:
                    rch_data_dict_PopList = p.map(self.read_rch_sim_data, rch_thread_args)
                rch_data_dict_PopList_sort = sorted(rch_data_dict_PopList, key=lambda x: x[0])
                print('rch_data_dict_PopList_sort:', len(rch_data_dict_PopList_sort))
                if self.spatial_unit == 'HRU':
                    # output.hru
                    with Pool(processes=self.cpu_worker_num) as p:
                        ET_SW_HRU_data_dict_PopList = p.map(self.read_hru_sim_data, hru_thread_args)
                    ET_SW_HRU_data_dict_PopList_sort = [(et_sw_hru_dict_idx[0], et_sw_hru_dict_idx[1][0], et_sw_hru_dict_idx[1][1])
                                                        for et_sw_hru_dict_idx in sorted(ET_SW_HRU_data_dict_PopList, key=lambda x: x[0])]
                    print('ET_SW_HRU_data_dict_PopList_sort:', len(ET_SW_HRU_data_dict_PopList_sort))
                elif self.spatial_unit == 'Subbasin':
                    # output.sub
                    with Pool(processes=self.cpu_worker_num) as p:
                        ET_SW_Sub_data_dict_PopList = p.map(self.read_sub_sim_data, sub_thread_args)
                    ET_SW_Sub_data_dict_PopList_sort = [(et_sw_sub_dict_idx[0], et_sw_sub_dict_idx[1][0], et_sw_sub_dict_idx[1][1])
                                                        for et_sw_sub_dict_idx in sorted(ET_SW_Sub_data_dict_PopList, key=lambda x: x[0])]
                    print('ET_SW_Sub_data_dict_PopList_sort:', len(ET_SW_Sub_data_dict_PopList_sort))
            read_sim_end_time = time.time()
            print(f'Read Simulation Time: {round((read_sim_end_time - read_sim_start_time) / 60.0, 2)} min!')
        # Suspend for a period of time to prevent the hard disk from continuously reading or writing data
        time.sleep(5)
        print('\n')


        # Step 8: Calculate Objective Functions
        cal_obj_start_time = time.time()
        print('Step 8: Calculate Objective Functions'.center(self.comments_len, '='))
        # One Variables
        if self.cal_vars_list == ['Streamflow']:
            rch_pop_obj_func_list = []
            rch_sta_sim_dict = defaultdict(list)
            mean_season_rch_dict = defaultdict(list)
            eva_metrics_rch_dict = defaultdict(list)
            rch_pop_obj_func1_list, rch_pop_obj_func2_list, rch_pop_obj_func3_list = [], [], []
            for rch_sta_idx in sorted(self.hydro_stas.values()):
                print('rch_sta_idx:', rch_sta_idx)
                rch_obj_list = []
                rch_obj1_list, rch_obj2_list, rch_obj3_list = [], [], []
                for rch_pop_idx in rch_data_dict_PopList_sort:
                    pop_idx = rch_pop_idx[0]
                    # print('pop_idx:', pop_idx)
                    rch_sta_sim_data = rch_pop_idx[1][rch_sta_idx]
                    # print('rch_sta_sim_data:', rch_sta_sim_data.shape, rch_sta_sim_data)
                    rch_sta_obs_data = self.obs_sf_data[rch_sta_idx]
                    # print('rch_sta_obs_data:', rch_sta_obs_data.shape, rch_sta_obs_data)
                    if self.cal_val_state == 'Calibration':
                        if self.cal_scheme == 'Multi-site':
                            if self.TVSA:
                                rch_obj_list_TVSA = []
                                winsize = 2 * self.half_win + 1
                                for win_idx in range(0, rch_sta_obs_data.shape[0], winsize):
                                    rch_sta_obs_data_win = rch_sta_obs_data[win_idx:win_idx+winsize]
                                    rch_sta_sim_data_win = rch_sta_sim_data[win_idx:win_idx+winsize]
                                    rch_obj_list_TVSA.append(self.RMSE_Moving_Window(rch_sta_obs_data_win, rch_sta_sim_data_win, self.half_win))
                                # print('rch_obj_list_TVSA:', len(rch_obj_list_TVSA))
                                rch_obj_list.append(rch_obj_list_TVSA)
                            else:
                                # Objective Functions (NSE, KGE, R2, PBIAS, RMSE)
                                if self.objective_funs[0] == 'NSE':
                                    rch_obj_list.append(self.NSE(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs[0] == 'KGE':
                                    rch_obj_list.append(self.KGE(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs[0] == 'R2':
                                    rch_obj_list.append(self.R2(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs[0] == 'PBIAS':
                                    rch_obj_list.append(self.PBIAS(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs[0] == 'RMSE':
                                    rch_obj_list.append(self.RMSE(rch_sta_obs_data, rch_sta_sim_data))
                        elif self.cal_scheme == 'Multi-objective':
                            if len(self.objective_funs) == 2:
                                if self.objective_funs == ['NSE', 'KGE']:
                                    rch_obj1_list.append(self.NSE(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.KGE(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs == ['NSE', 'R2']:
                                    rch_obj1_list.append(self.NSE(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.R2(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs == ['NSE', 'PBIAS']:
                                    rch_obj1_list.append(self.NSE(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.PBIAS(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs == ['NSE', 'RMSE']:
                                    rch_obj1_list.append(self.NSE(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.RMSE(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs == ['KGE', 'R2']:
                                    rch_obj1_list.append(self.KGE(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.R2(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs == ['KGE', 'PBIAS']:
                                    rch_obj1_list.append(self.KGE(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.PBIAS(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs == ['KGE', 'RMSE']:
                                    rch_obj1_list.append(self.KGE(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.RMSE(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs == ['R2', 'PBIAS']:
                                    rch_obj1_list.append(self.R2(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.PBIAS(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs == ['R2', 'RMSE']:
                                    rch_obj1_list.append(self.R2(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.RMSE(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs == ['PBIAS', 'RMSE']:
                                    rch_obj1_list.append(self.PBIAS(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.RMSE(rch_sta_obs_data, rch_sta_sim_data))
                            elif len(self.objective_funs) == 3:
                                if self.objective_funs == ['NSE', 'KGE', 'R2']:
                                    rch_obj1_list.append(self.NSE(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.KGE(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj3_list.append(self.R2(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs == ['NSE', 'KGE', 'PBIAS']:
                                    rch_obj1_list.append(self.NSE(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.KGE(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj3_list.append(self.PBIAS(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs == ['NSE', 'KGE', 'RMSE']:
                                    rch_obj1_list.append(self.NSE(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.KGE(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj3_list.append(self.RMSE(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs == ['NSE', 'R2', 'PBIAS']:
                                    rch_obj1_list.append(self.NSE(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.R2(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj3_list.append(self.PBIAS(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs == ['NSE', 'R2', 'RMSE']:
                                    rch_obj1_list.append(self.NSE(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.R2(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj3_list.append(self.RMSE(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs == ['NSE', 'PBIAS', 'RMSE']:
                                    rch_obj1_list.append(self.NSE(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.PBIAS(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj3_list.append(self.RMSE(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs == ['KGE', 'R2', 'PBIAS']:
                                    rch_obj1_list.append(self.KGE(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.R2(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj3_list.append(self.PBIAS(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs == ['KGE', 'R2', 'RMSE']:
                                    rch_obj1_list.append(self.KGE(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.R2(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj3_list.append(self.RMSE(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs == ['KGE', 'PBIAS', 'RMSE']:
                                    rch_obj1_list.append(self.KGE(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.PBIAS(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj3_list.append(self.RMSE(rch_sta_obs_data, rch_sta_sim_data))
                                elif self.objective_funs == ['R2', 'PBIAS', 'RMSE']:
                                    rch_obj1_list.append(self.R2(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj2_list.append(self.PBIAS(rch_sta_obs_data, rch_sta_sim_data))
                                    rch_obj3_list.append(self.RMSE(rch_sta_obs_data, rch_sta_sim_data))
                    elif self.cal_val_state == 'Validation':
                        rch_sta_sim_dict[rch_sta_idx].append(rch_sta_sim_data)
                rch_pop_obj_func_list.append(rch_obj_list)
                rch_pop_obj_func1_list.append(rch_obj1_list)
                rch_pop_obj_func2_list.append(rch_obj2_list)
                rch_pop_obj_func3_list.append(rch_obj3_list)
            #
            if self.cal_val_state == 'Calibration':
                if self.cal_scheme == 'Multi-site':
                    if self.TVSA:
                        print('rch_pop_obj_func_list:', len(rch_pop_obj_func_list))
                        cal_obj_end_time = time.time()
                        print(f'Calculate Objective Functions Time: {round((cal_obj_end_time - cal_obj_start_time) / 60.0, 2)} min!')
                        print('\n')
                        return rch_pop_obj_func_list
                    else:
                        print('rch_pop_obj_func_list:', len(rch_pop_obj_func_list))
                        self.para_obj_val_dict[self.n_gen] = np.hstack((SWATParas_Sampling, np.array(rch_pop_obj_func_list).T))
                        self.n_gen += 1
                        cal_obj_end_time = time.time()
                        print(f'Calculate Objective Functions Time: {round((cal_obj_end_time - cal_obj_start_time) / 60.0, 2)} min!')
                        print('\n')
                        return rch_pop_obj_func_list
                elif self.cal_scheme == 'Multi-objective':
                    if len(self.objective_funs) == 2:
                        print('rch_pop_obj_func1_list:', len(rch_pop_obj_func1_list))
                        print('rch_pop_obj_func2_list:', len(rch_pop_obj_func2_list))
                        self.para_obj_val_dict[self.n_gen] = np.hstack((SWATParas_Sampling,
                                                                        np.array(rch_pop_obj_func1_list).T,
                                                                        np.array(rch_pop_obj_func2_list).T))
                        self.n_gen += 1
                        cal_obj_end_time = time.time()
                        print(f'Calculate Objective Functions Time: {round((cal_obj_end_time - cal_obj_start_time) / 60.0, 2)} min!')
                        print('\n')
                        return rch_pop_obj_func1_list, rch_pop_obj_func2_list
                    elif len(self.objective_funs) == 3:
                        print('rch_pop_obj_func1_list:', len(rch_pop_obj_func1_list))
                        print('rch_pop_obj_func2_list:', len(rch_pop_obj_func2_list))
                        print('rch_pop_obj_func3_list:', len(rch_pop_obj_func3_list))
                        self.para_obj_val_dict[self.n_gen] = np.hstack((SWATParas_Sampling,
                                                                        np.array(rch_pop_obj_func1_list).T,
                                                                        np.array(rch_pop_obj_func2_list).T,
                                                                        np.array(rch_pop_obj_func3_list).T))
                        self.n_gen += 1
                        cal_obj_end_time = time.time()
                        print(f'Calculate Objective Functions Time: {round((cal_obj_end_time - cal_obj_start_time) / 60.0, 2)} min!')
                        print('\n')
                        return rch_pop_obj_func1_list, rch_pop_obj_func2_list, rch_pop_obj_func3_list
            elif self.cal_val_state == 'Validation':
                comp_txt = r'Compromise_solution_Streamflow_Multi-site.txt'
                if self.cal_scheme == 'Multi-site':
                    comp_txt = r'Compromise_solution_Streamflow_Multi-site.txt'
                elif self.cal_scheme == 'Multi-objective':
                    comp_txt = r'Compromise_solution_Streamflow_Multi-objective.txt'
                param_val_arr_comp = np.loadtxt(f'{self.swat_nsga_out}\\{comp_txt}',
                                                skiprows=1, usecols=range(len(self.swat_parameter)), dtype=float)
                print('param_val_arr_comp:', param_val_arr_comp.shape, '\n', param_val_arr_comp)
                comp_row_idx = np.where(np.all(SWATParas_Sampling == param_val_arr_comp, axis=1))[0][0]
                date_val = pd.date_range(start=f'{self.val_period[0]}-01-01', end=f'{self.val_period[1]}-12-31', freq='D')
                for rch_sta_idx in sorted(self.hydro_stas.values()):
                    # Sim_min, Sim_max, Sim_compro_sol
                    rch_sta_sim_dict[rch_sta_idx] = [np.min(np.array(rch_sta_sim_dict[rch_sta_idx]), axis=0),
                                                     np.max(np.array(rch_sta_sim_dict[rch_sta_idx]), axis=0),
                                                     np.array(rch_sta_sim_dict[rch_sta_idx])[comp_row_idx]]
                    # mean_season_obs, mean_season_min, mean_season_max, mean_season_compro_sol
                    mean_season_rch_obs = self.mean_seasonal_cycle(date_val, self.obs_sf_data[rch_sta_idx])
                    mean_season_rch_min = self.mean_seasonal_cycle(date_val, rch_sta_sim_dict[rch_sta_idx][0])
                    mean_season_rch_max = self.mean_seasonal_cycle(date_val, rch_sta_sim_dict[rch_sta_idx][1])
                    mean_season_rch_comp = self.mean_seasonal_cycle(date_val, rch_sta_sim_dict[rch_sta_idx][2])
                    mean_season_rch_dict[rch_sta_idx] = [mean_season_rch_obs, mean_season_rch_min, mean_season_rch_max, mean_season_rch_comp]
                    # Evaluation metric
                    rch_sta_obs_cal = self.obs_sf_data[rch_sta_idx][:len(self.obs_date_day_cal)]
                    rch_sta_obs_val = self.obs_sf_data[rch_sta_idx][len(self.obs_date_day_cal):]
                    #
                    rch_sta_sim_cal = rch_sta_sim_dict[rch_sta_idx][2][:len(self.obs_date_day_cal)]
                    rch_sta_sim_val = rch_sta_sim_dict[rch_sta_idx][2][len(self.obs_date_day_cal):]
                    #
                    rch_sta_cal_KGE = self.KGE(rch_sta_obs_cal, rch_sta_sim_cal)
                    rch_sta_val_KGE = self.KGE(rch_sta_obs_val, rch_sta_sim_val)
                    rch_sta_mean_season_KGE = self.KGE(mean_season_rch_obs, mean_season_rch_comp)
                    #
                    rch_sta_cal_NSE = self.NSE(rch_sta_obs_cal, rch_sta_sim_cal)
                    rch_sta_val_NSE = self.NSE(rch_sta_obs_val, rch_sta_sim_val)
                    rch_sta_mean_season_NSE = self.NSE(mean_season_rch_obs, mean_season_rch_comp)
                    #
                    rch_sta_cal_PBIAS = self.PBIAS(rch_sta_obs_cal, rch_sta_sim_cal)
                    rch_sta_val_PBIAS = self.PBIAS(rch_sta_obs_val, rch_sta_sim_val)
                    rch_sta_mean_season_PBIAS = self.PBIAS(mean_season_rch_obs, mean_season_rch_comp)
                    #
                    eva_metrics_rch_dict[rch_sta_idx] = [(rch_sta_cal_KGE, rch_sta_val_KGE, rch_sta_mean_season_KGE),
                                                         (rch_sta_cal_NSE, rch_sta_val_NSE, rch_sta_mean_season_NSE),
                                                         (rch_sta_cal_PBIAS, rch_sta_val_PBIAS, rch_sta_mean_season_PBIAS)]
                return rch_sta_sim_dict, mean_season_rch_dict, eva_metrics_rch_dict
        elif self.cal_vars_list == ['LAI']:
            LAI_sim_arr = []
            LAI_pop_obj_func_list = []
            LAI_data_dict_PopList_sort_w = []
            LAI_pop_obj_func1_list, LAI_pop_obj_func2_list, LAI_pop_obj_func3_list = [], [], []
            # Objective Functions Mode (Average_Obj_Funcs/Area_Weighted_Avg)
            if self.objec_mode == 'Area_Weighted_Avg':
                ## Area weighted average LAI simulation with Multiprocessing
                print('Area weighted average LAI simulation using multiprocessing:')
                if self.forest_cal:
                    process_args = [(LAI_pop_idx, self.HRU_ID_Veg, self.HRU_veg_area_dict) for LAI_pop_idx in LAI_data_dict_PopList_sort]
                    print('process_args:', len(process_args))
                    with Pool(processes=5) as p:
                        sim_LAI_avg = p.starmap(self.Veg_sim_area_weighted_avg, process_args)
                    LAI_data_dict_PopList_sort_w = [sim_lai_idx for sim_lai_idx in sorted(sim_LAI_avg, key=lambda x: x[0])]
                    print('LAI_data_dict_PopList_sort_w:', len(LAI_data_dict_PopList_sort_w))
                else:
                    process_args = [LAI_pop_idx for LAI_pop_idx in LAI_data_dict_PopList_sort]
                    print('process_args:', len(process_args))
                    with Pool(processes=5) as p:
                        sim_LAI_avg = p.map(self.RS_sim_area_weighted_avg, process_args)
                    LAI_data_dict_PopList_sort_w = [sim_lai_idx for sim_lai_idx in sorted(sim_LAI_avg, key=lambda x: x[0])]
                    print('LAI_data_dict_PopList_sort_w:', len(LAI_data_dict_PopList_sort_w))
                #
                for LAI_sim_pop_idx in LAI_data_dict_PopList_sort_w:
                    pop_idx = LAI_sim_pop_idx[0]
                    # print('pop_idx:', pop_idx)
                    LAI_sim_data_dict_area_w_pop = LAI_sim_pop_idx[1]
                    if self.cal_val_state == 'Calibration':
                        if self.cal_scheme == 'Single-objective':
                            if self.TVSA:
                                LAI_obj_list_TVSA = []
                                winsize = 2 * self.half_win + 1
                                for win_idx in range(0, self.LAI_obs_data_dict_area_w.shape[0], winsize):
                                    LAI_obs_data_win = self.LAI_obs_data_dict_area_w[win_idx:win_idx + winsize]
                                    LAI_sim_data_win = LAI_sim_data_dict_area_w_pop[win_idx:win_idx + winsize]
                                    LAI_obj_list_TVSA.append(self.RMSE_Moving_Window(LAI_obs_data_win, LAI_sim_data_win, self.half_win))
                                # print('LAI_obj_list_TVSA:', len(LAI_obj_list_TVSA))
                                LAI_pop_obj_func_list.append(LAI_obj_list_TVSA)
                            else:
                                # Objective Functions (NSE, KGE, R2, PBIAS, RMSE)
                                if self.objective_funs[0] == 'NSE':
                                    LAI_pop_obj_func_list.append(self.NSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs[0] == 'KGE':
                                    LAI_pop_obj_func_list.append(self.KGE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs[0] == 'R2':
                                    LAI_pop_obj_func_list.append(self.R2(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs[0] == 'PBIAS':
                                    LAI_pop_obj_func_list.append(self.PBIAS(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs[0] == 'RMSE':
                                    LAI_pop_obj_func_list.append(self.RMSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                        elif self.cal_scheme == 'Multi-objective':
                            if len(self.objective_funs) == 2:
                                if self.objective_funs == ['NSE', 'KGE']:
                                    LAI_pop_obj_func1_list.append(self.NSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.KGE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['NSE', 'R2']:
                                    LAI_pop_obj_func1_list.append(self.NSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.R2(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['NSE', 'PBIAS']:
                                    LAI_pop_obj_func1_list.append(self.NSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.PBIAS(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['NSE', 'RMSE']:
                                    LAI_pop_obj_func1_list.append(self.NSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.RMSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['KGE', 'R2']:
                                    LAI_pop_obj_func1_list.append(self.KGE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.R2(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['KGE', 'PBIAS']:
                                    LAI_pop_obj_func1_list.append(self.KGE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.PBIAS(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['KGE', 'RMSE']:
                                    LAI_pop_obj_func1_list.append(self.KGE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.RMSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['R2', 'PBIAS']:
                                    LAI_pop_obj_func1_list.append(self.R2(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.PBIAS(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['R2', 'RMSE']:
                                    LAI_pop_obj_func1_list.append(self.R2(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.RMSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['PBIAS', 'RMSE']:
                                    LAI_pop_obj_func1_list.append(self.PBIAS(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.RMSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                            elif len(self.objective_funs) == 3:
                                if self.objective_funs == ['NSE', 'KGE', 'R2']:
                                    LAI_pop_obj_func1_list.append(self.NSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.KGE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func3_list.append(self.R2(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['NSE', 'KGE', 'PBIAS']:
                                    LAI_pop_obj_func1_list.append(self.NSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.KGE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func3_list.append(self.PBIAS(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['NSE', 'KGE', 'RMSE']:
                                    LAI_pop_obj_func1_list.append(self.NSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.KGE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func3_list.append(self.RMSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['NSE', 'R2', 'PBIAS']:
                                    LAI_pop_obj_func1_list.append(self.NSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.R2(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func3_list.append(self.PBIAS(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['NSE', 'R2', 'RMSE']:
                                    LAI_pop_obj_func1_list.append(self.NSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.R2(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func3_list.append(self.RMSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['NSE', 'PBIAS', 'RMSE']:
                                    LAI_pop_obj_func1_list.append(self.NSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.PBIAS(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func3_list.append(self.RMSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['KGE', 'R2', 'PBIAS']:
                                    LAI_pop_obj_func1_list.append(self.KGE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.R2(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func3_list.append(self.PBIAS(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['KGE', 'R2', 'RMSE']:
                                    LAI_pop_obj_func1_list.append(self.KGE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.R2(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func3_list.append(self.RMSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['KGE', 'PBIAS', 'RMSE']:
                                    LAI_pop_obj_func1_list.append(self.KGE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.PBIAS(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func3_list.append(self.RMSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['R2', 'PBIAS', 'RMSE']:
                                    LAI_pop_obj_func1_list.append(self.R2(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func2_list.append(self.PBIAS(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                                    LAI_pop_obj_func3_list.append(self.RMSE(self.LAI_obs_data_dict_area_w, LAI_sim_data_dict_area_w_pop))
                    elif self.cal_val_state == 'Validation':
                        LAI_sim_arr.append(LAI_sim_data_dict_area_w_pop)
            #
            if self.cal_val_state == 'Calibration':
                if self.cal_scheme == 'Single-objective':
                    print('LAI_pop_obj_func_list:', len(LAI_pop_obj_func_list))
                    self.para_obj_val_dict[self.n_gen] = np.hstack((SWATParas_Sampling, np.array(LAI_pop_obj_func_list).reshape(-1, 1)))
                    self.n_gen += 1
                    cal_obj_end_time = time.time()
                    print(f'Calculate Objective Functions Time: {round((cal_obj_end_time - cal_obj_start_time) / 60.0, 2)} min!')
                    print('\n')
                    return LAI_pop_obj_func_list
                elif self.cal_scheme == 'Multi-objective':
                    if len(self.objective_funs) == 2:
                        print('LAI_pop_obj_func1_list:', len(LAI_pop_obj_func1_list))
                        print('LAI_pop_obj_func2_list:', len(LAI_pop_obj_func2_list))
                        self.para_obj_val_dict[self.n_gen] = np.hstack((SWATParas_Sampling,
                                                                        np.array(LAI_pop_obj_func1_list).reshape(-1, 1),
                                                                        np.array(LAI_pop_obj_func2_list).reshape(-1, 1)))
                        self.n_gen += 1
                        cal_obj_end_time = time.time()
                        print(f'Calculate Objective Functions Time: {round((cal_obj_end_time - cal_obj_start_time) / 60.0, 2)} min!')
                        print('\n')
                        return LAI_pop_obj_func1_list, LAI_pop_obj_func2_list
                    elif len(self.objective_funs) == 3:
                        print('LAI_pop_obj_func1_list:', len(LAI_pop_obj_func1_list))
                        print('LAI_pop_obj_func2_list:', len(LAI_pop_obj_func2_list))
                        print('LAI_pop_obj_func3_list:', len(LAI_pop_obj_func3_list))
                        self.para_obj_val_dict[self.n_gen] = np.hstack((SWATParas_Sampling,
                                                                        np.array(LAI_pop_obj_func1_list).reshape(-1, 1),
                                                                        np.array(LAI_pop_obj_func2_list).reshape(-1, 1),
                                                                        np.array(LAI_pop_obj_func3_list).reshape(-1, 1)))
                        self.n_gen += 1
                        cal_obj_end_time = time.time()
                        print(f'Calculate Objective Functions Time: {round((cal_obj_end_time - cal_obj_start_time) / 60.0, 2)} min!')
                        print('\n')
                        return LAI_pop_obj_func1_list, LAI_pop_obj_func2_list, LAI_pop_obj_func3_list
            elif self.cal_val_state == 'Validation':
                param_val_arr_comp = np.loadtxt(f'{self.swat_nsga_out}\\Compromise_solution_LAI_Multi-objective.txt',
                                                skiprows=1, usecols=range(len(self.swat_parameter)), dtype=float)
                print('param_val_arr_comp:', param_val_arr_comp.shape, '\n', param_val_arr_comp)
                comp_row_idx = 0 if SWATParas_Sampling.ndim == 1 else np.where(np.all(SWATParas_Sampling == param_val_arr_comp, axis=1))[0][0]
                print('comp_row_idx:', comp_row_idx)
                date_val = pd.date_range(start=f'{self.val_period[0]}-01-01', end=f'{self.val_period[1]}-12-31', freq='D')
                # Sim_min, Sim_max, Sim_compro_sol
                LAI_sim_arr = [np.min(np.array(LAI_sim_arr), axis=0), np.max(np.array(LAI_sim_arr), axis=0), np.array(LAI_sim_arr)[comp_row_idx]]
                # mean_season_obs, mean_season_min, mean_season_max, mean_season_compro_sol
                mean_season_LAI_obs = self.mean_seasonal_cycle(date_val, self.LAI_obs_data_dict_area_w)
                mean_season_LAI_min = self.mean_seasonal_cycle(date_val, LAI_sim_arr[0])
                mean_season_LAI_max = self.mean_seasonal_cycle(date_val, LAI_sim_arr[1])
                mean_season_LAI_comp = self.mean_seasonal_cycle(date_val, LAI_sim_arr[2])
                mean_season_LAI_dict = [mean_season_LAI_obs, mean_season_LAI_min, mean_season_LAI_max, mean_season_LAI_comp]
                # Evaluation metric
                LAI_obs_cal = self.LAI_obs_data_dict_area_w[:len(self.obs_date_day_cal)]
                LAI_obs_val = self.LAI_obs_data_dict_area_w[len(self.obs_date_day_cal):]
                #
                LAI_sim_cal = LAI_sim_arr[2][:len(self.obs_date_day_cal)]
                LAI_sim_val = LAI_sim_arr[2][len(self.obs_date_day_cal):]
                #
                LAI_cal_KGE = self.KGE(LAI_obs_cal, LAI_sim_cal)
                LAI_val_KGE = self.KGE(LAI_obs_val, LAI_sim_val)
                LAI_mean_season_KGE = self.KGE(mean_season_LAI_obs, mean_season_LAI_comp)
                #
                LAI_cal_NSE = self.NSE(LAI_obs_cal, LAI_sim_cal)
                LAI_val_NSE = self.NSE(LAI_obs_val, LAI_sim_val)
                LAI_mean_season_NSE = self.NSE(mean_season_LAI_obs, mean_season_LAI_comp)
                #
                LAI_cal_PBIAS = self.PBIAS(LAI_obs_cal, LAI_sim_cal)
                LAI_val_PBIAS = self.PBIAS(LAI_obs_val, LAI_sim_val)
                LAI_mean_season_PBIAS = self.PBIAS(mean_season_LAI_obs, mean_season_LAI_comp)
                #
                eva_metrics_LAI_dict = [(LAI_cal_KGE, LAI_val_KGE, LAI_mean_season_KGE),
                                        (LAI_cal_NSE, LAI_val_NSE, LAI_mean_season_NSE),
                                        (LAI_cal_PBIAS, LAI_val_PBIAS, LAI_mean_season_PBIAS)]
                return LAI_sim_arr, mean_season_LAI_dict, eva_metrics_LAI_dict
        elif self.cal_vars_list == ['BIOM']:
            BIOM_pop_obj_func_list = []
            BIOM_pop_obj_func1_list, BIOM_pop_obj_func2_list, BIOM_pop_obj_func3_list = [], [], []
            # Objective Functions Mode (Average_Obj_Funcs/Area_Weighted_Avg)
            if self.objec_mode == 'Area_Weighted_Avg':
                ## Area weighted average LAI simulation with Multiprocessing
                print('Area weighted average BIOM simulation using multiprocessing:')
                if self.cal_val_state == 'Calibration':
                    process_args = [(BIOM_pop_idx, self.HRU_ID_BIOM, self.HRU_BIOM_area_dict) for BIOM_pop_idx in BIOM_data_dict_PopList_sort]
                    print('process_args:', len(process_args))
                    with Pool(processes=self.cpu_worker_num) as p:
                        sim_BIOM_avg = p.starmap(self.Veg_sim_area_weighted_avg, process_args)
                    BIOM_data_dict_PopList_sort_w = [sim_biom_idx for sim_biom_idx in sorted(sim_BIOM_avg, key=lambda x: x[0])]
                    print('BIOM_data_dict_PopList_sort_w:', len(BIOM_data_dict_PopList_sort_w))
                    #
                    for BIOM_sim_pop_idx in BIOM_data_dict_PopList_sort_w:
                        pop_idx = BIOM_sim_pop_idx[0]
                        # print('pop_idx:', pop_idx)
                        BIOM_sim_data_dict_area_w_pop = BIOM_sim_pop_idx[1]
                        if self.cal_scheme == 'Single-objective':
                            # Objective Functions (NSE, KGE, R2, PBIAS, RMSE)
                            if self.objective_funs[0] == 'NSE':
                                BIOM_pop_obj_func_list.append(self.NSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                            elif self.objective_funs[0] == 'KGE':
                                BIOM_pop_obj_func_list.append(self.KGE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                            elif self.objective_funs[0] == 'R2':
                                BIOM_pop_obj_func_list.append(self.R2(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                            elif self.objective_funs[0] == 'PBIAS':
                                BIOM_pop_obj_func_list.append(self.PBIAS(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                            elif self.objective_funs[0] == 'RMSE':
                                BIOM_pop_obj_func_list.append(self.RMSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                        elif self.cal_scheme == 'Multi-objective':
                            if len(self.objective_funs) == 2:
                                if self.objective_funs == ['NSE', 'KGE']:
                                    BIOM_pop_obj_func1_list.append(self.NSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.KGE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['NSE', 'R2']:
                                    BIOM_pop_obj_func1_list.append(self.NSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.R2(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['NSE', 'PBIAS']:
                                    BIOM_pop_obj_func1_list.append(self.NSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.PBIAS(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['NSE', 'RMSE']:
                                    BIOM_pop_obj_func1_list.append(self.NSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.RMSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['KGE', 'R2']:
                                    BIOM_pop_obj_func1_list.append(self.KGE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.R2(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['KGE', 'PBIAS']:
                                    BIOM_pop_obj_func1_list.append(self.KGE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.PBIAS(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['KGE', 'RMSE']:
                                    BIOM_pop_obj_func1_list.append(self.KGE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.RMSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['R2', 'PBIAS']:
                                    BIOM_pop_obj_func1_list.append(self.R2(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.PBIAS(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['R2', 'RMSE']:
                                    BIOM_pop_obj_func1_list.append(self.R2(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.RMSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['PBIAS', 'RMSE']:
                                    BIOM_pop_obj_func1_list.append(self.PBIAS(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.RMSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                            elif len(self.objective_funs) == 3:
                                if self.objective_funs == ['NSE', 'KGE', 'R2']:
                                    BIOM_pop_obj_func1_list.append(self.NSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.KGE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func3_list.append(self.R2(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['NSE', 'KGE', 'PBIAS']:
                                    BIOM_pop_obj_func1_list.append(self.NSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.KGE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func3_list.append(self.PBIAS(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['NSE', 'KGE', 'RMSE']:
                                    BIOM_pop_obj_func1_list.append(self.NSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.KGE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func3_list.append(self.RMSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['NSE', 'R2', 'PBIAS']:
                                    BIOM_pop_obj_func1_list.append(self.NSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.R2(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func3_list.append(self.PBIAS(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['NSE', 'R2', 'RMSE']:
                                    BIOM_pop_obj_func1_list.append(self.NSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.R2(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func3_list.append(self.RMSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['NSE', 'PBIAS', 'RMSE']:
                                    BIOM_pop_obj_func1_list.append(self.NSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.PBIAS(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func3_list.append(self.RMSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['KGE', 'R2', 'PBIAS']:
                                    BIOM_pop_obj_func1_list.append(self.KGE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.R2(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func3_list.append(self.PBIAS(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['KGE', 'R2', 'RMSE']:
                                    BIOM_pop_obj_func1_list.append(self.KGE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.R2(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func3_list.append(self.RMSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['KGE', 'PBIAS', 'RMSE']:
                                    BIOM_pop_obj_func1_list.append(self.KGE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.PBIAS(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func3_list.append(self.RMSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                elif self.objective_funs == ['R2', 'PBIAS', 'RMSE']:
                                    BIOM_pop_obj_func1_list.append(self.R2(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func2_list.append(self.PBIAS(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                                    BIOM_pop_obj_func3_list.append(self.RMSE(self.BIOM_obs_data_dict_area_w, BIOM_sim_data_dict_area_w_pop))
                    if self.cal_scheme == 'Single-objective':
                        print('BIOM_pop_obj_func_list:', len(BIOM_pop_obj_func_list))
                        self.para_obj_val_dict[self.n_gen] = np.hstack((SWATParas_Sampling, np.array(BIOM_pop_obj_func_list).reshape(-1, 1)))
                        self.n_gen += 1
                        cal_obj_end_time = time.time()
                        print(f'Calculate Objective Functions Time: {round((cal_obj_end_time - cal_obj_start_time) / 60.0, 2)} min!')
                        print('\n')
                        return BIOM_pop_obj_func_list
                    elif self.cal_scheme == 'Multi-objective':
                        if len(self.objective_funs) == 2:
                            print('BIOM_pop_obj_func1_list:', len(BIOM_pop_obj_func1_list))
                            print('BIOM_pop_obj_func2_list:', len(BIOM_pop_obj_func2_list))
                            self.para_obj_val_dict[self.n_gen] = np.hstack((SWATParas_Sampling,
                                                                            np.array(BIOM_pop_obj_func1_list).reshape(-1, 1),
                                                                            np.array(BIOM_pop_obj_func2_list).reshape(-1, 1)))
                            self.n_gen += 1
                            cal_obj_end_time = time.time()
                            print(f'Calculate Objective Functions Time: {round((cal_obj_end_time - cal_obj_start_time) / 60.0, 2)} min!')
                            print('\n')
                            return BIOM_pop_obj_func1_list, BIOM_pop_obj_func2_list
                        elif len(self.objective_funs) == 3:
                            print('BIOM_pop_obj_func1_list:', len(BIOM_pop_obj_func1_list))
                            print('BIOM_pop_obj_func2_list:', len(BIOM_pop_obj_func2_list))
                            print('BIOM_pop_obj_func3_list:', len(BIOM_pop_obj_func3_list))
                            self.para_obj_val_dict[self.n_gen] = np.hstack((SWATParas_Sampling,
                                                                            np.array(BIOM_pop_obj_func1_list).reshape(-1, 1),
                                                                            np.array(BIOM_pop_obj_func2_list).reshape(-1, 1),
                                                                            np.array(BIOM_pop_obj_func3_list).reshape(-1, 1)))
                            self.n_gen += 1
                            cal_obj_end_time = time.time()
                            print(f'Calculate Objective Functions Time: {round((cal_obj_end_time - cal_obj_start_time) / 60.0, 2)} min!')
                            print('\n')
                            return BIOM_pop_obj_func1_list, BIOM_pop_obj_func2_list, BIOM_pop_obj_func3_list
                elif self.cal_val_state == 'Validation':
                    process_args_FRST = [(BIOM_pop_idx_FRST, self.HRU_ID_BIOM_Forest['FRST'], self.HRU_BIOM_area_dict_FRST) for BIOM_pop_idx_FRST in BIOM_data_dict_PopList_sort]
                    process_args_FRSE = [(BIOM_pop_idx_FRSE, self.HRU_ID_BIOM_Forest['FRSE'], self.HRU_BIOM_area_dict_FRSE) for BIOM_pop_idx_FRSE in BIOM_data_dict_PopList_sort]
                    print('process_args_FRST:', len(process_args_FRST))
                    print('process_args_FRSE:', len(process_args_FRSE))
                    with Pool(processes=self.cpu_worker_num) as p:
                        sim_BIOM_avg_FRST = p.starmap(self.Veg_sim_area_weighted_avg, process_args_FRST)
                    with Pool(processes=self.cpu_worker_num) as p:
                        sim_BIOM_avg_FRSE = p.starmap(self.Veg_sim_area_weighted_avg, process_args_FRSE)
                    #
                    param_val_arr_comp = np.loadtxt(f'{self.swat_nsga_out}\\Compromise_solution_BIOM_Multi-objective.txt',
                                                    skiprows=1, usecols=range(len(self.swat_parameter)), dtype=float)
                    print('param_val_arr_comp:', param_val_arr_comp.shape, '\n', param_val_arr_comp)
                    comp_row_idx = 0 if SWATParas_Sampling.ndim == 1 else np.where(np.all(SWATParas_Sampling == param_val_arr_comp, axis=1))[0][0]
                    print('comp_row_idx:', comp_row_idx)
                    # Sim_compro_sol
                    BIOM_sim_arr_FRST_comp = list(filter(lambda x: x[0] == (comp_row_idx + 1), sim_BIOM_avg_FRST))[0][1]
                    BIOM_sim_arr_FRSE_comp = list(filter(lambda x: x[0] == (comp_row_idx + 1), sim_BIOM_avg_FRSE))[0][1]
                    print('BIOM_sim_arr_FRST_comp:', BIOM_sim_arr_FRST_comp)
                    print('BIOM_sim_arr_FRSE_comp:', BIOM_sim_arr_FRSE_comp)
                    # Std
                    obs_std_FRST = np.std(self.BIOM_obs_data_dict_area_w_FRST)
                    obs_std_FRSE = np.std(self.BIOM_obs_data_dict_area_w_FRSE)
                    sim_std_FRST = np.std(BIOM_sim_arr_FRST_comp)
                    sim_std_FRSE = np.std(BIOM_sim_arr_FRSE_comp)
                    return (BIOM_sim_arr_FRST_comp, BIOM_sim_arr_FRSE_comp), ((obs_std_FRST, obs_std_FRSE), (sim_std_FRST, sim_std_FRSE)), None
        elif self.cal_vars_list == ['ET']:
            # ET
            ET_pop_obj_func_list = []
            ET_pop_obj_func1_list, ET_pop_obj_func2_list, ET_pop_obj_func3_list = [], [], []
            # Objective Functions Mode (Average_Obj_Funcs/Area_Weighted_Avg)
            if self.objec_mode == 'Area_Weighted_Avg':
                # ET simulation
                ET_data_dict_PopList_sort = ET_HRU_data_dict_PopList_sort if self.spatial_unit == 'HRU' else ET_Sub_data_dict_PopList_sort
                ## Area weighted average LAI simulation with Multiprocessing
                print('Area weighted average ET simulation using multiprocessing:')
                process_args = [(ET_pop_idx, self.HRU_ID_Veg, self.HRU_veg_area_dict) for ET_pop_idx in ET_data_dict_PopList_sort]
                print('process_args:', len(process_args))
                with Pool(processes=5) as p:
                    sim_ET_avg = p.starmap(self.Veg_sim_area_weighted_avg, process_args)
                ET_data_dict_PopList_sort_w = [sim_et_idx for sim_et_idx in sorted(sim_ET_avg, key=lambda x: x[0])]
                print('ET_data_dict_PopList_sort_w:', len(ET_data_dict_PopList_sort_w))
                #
                for ET_sim_pop_idx in ET_data_dict_PopList_sort_w:
                    pop_idx = ET_sim_pop_idx[0]
                    # print('pop_idx:', pop_idx)
                    ET_sim_data_dict_area_w_pop = ET_sim_pop_idx[1]
                    if self.cal_scheme == 'Single-objective':
                        # Objective Functions (NSE, KGE, R2, PBIAS, RMSE)
                        if self.objective_funs[0] == 'NSE':
                            ET_pop_obj_func_list.append(self.NSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                        elif self.objective_funs[0] == 'KGE':
                            ET_pop_obj_func_list.append(self.KGE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                        elif self.objective_funs[0] == 'R2':
                            ET_pop_obj_func_list.append(self.R2(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                        elif self.objective_funs[0] == 'PBIAS':
                            ET_pop_obj_func_list.append(self.PBIAS(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                        elif self.objective_funs[0] == 'RMSE':
                            ET_pop_obj_func_list.append(self.RMSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                    elif self.cal_scheme == 'Multi-objective':
                        if len(self.objective_funs) == 2:
                            if self.objective_funs == ['NSE', 'KGE']:
                                ET_pop_obj_func1_list.append(self.NSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.KGE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs == ['NSE', 'R2']:
                                ET_pop_obj_func1_list.append(self.NSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.R2(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs == ['NSE', 'PBIAS']:
                                ET_pop_obj_func1_list.append(self.NSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.PBIAS(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs == ['NSE', 'RMSE']:
                                ET_pop_obj_func1_list.append(self.NSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.RMSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs == ['KGE', 'R2']:
                                ET_pop_obj_func1_list.append(self.KGE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.R2(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs == ['KGE', 'PBIAS']:
                                ET_pop_obj_func1_list.append(self.KGE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.PBIAS(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs == ['KGE', 'RMSE']:
                                ET_pop_obj_func1_list.append(self.KGE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.RMSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs == ['R2', 'PBIAS']:
                                ET_pop_obj_func1_list.append(self.R2(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.PBIAS(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs == ['R2', 'RMSE']:
                                ET_pop_obj_func1_list.append(self.R2(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.RMSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs == ['PBIAS', 'RMSE']:
                                ET_pop_obj_func1_list.append(self.PBIAS(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.RMSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                        elif len(self.objective_funs) == 3:
                            if self.objective_funs == ['NSE', 'KGE', 'R2']:
                                ET_pop_obj_func1_list.append(self.NSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.KGE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func3_list.append(self.R2(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs == ['NSE', 'KGE', 'PBIAS']:
                                ET_pop_obj_func1_list.append(self.NSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.KGE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func3_list.append(self.PBIAS(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs == ['NSE', 'KGE', 'RMSE']:
                                ET_pop_obj_func1_list.append(self.NSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.KGE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func3_list.append(self.RMSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs == ['NSE', 'R2', 'PBIAS']:
                                ET_pop_obj_func1_list.append(self.NSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.R2(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func3_list.append(self.PBIAS(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs == ['NSE', 'R2', 'RMSE']:
                                ET_pop_obj_func1_list.append(self.NSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.R2(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func3_list.append(self.RMSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs == ['NSE', 'PBIAS', 'RMSE']:
                                ET_pop_obj_func1_list.append(self.NSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.PBIAS(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func3_list.append(self.RMSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs == ['KGE', 'R2', 'PBIAS']:
                                ET_pop_obj_func1_list.append(self.KGE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.R2(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func3_list.append(self.PBIAS(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs == ['KGE', 'R2', 'RMSE']:
                                ET_pop_obj_func1_list.append(self.KGE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.R2(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func3_list.append(self.RMSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs == ['KGE', 'PBIAS', 'RMSE']:
                                ET_pop_obj_func1_list.append(self.KGE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.PBIAS(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func3_list.append(self.RMSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs == ['R2', 'PBIAS', 'RMSE']:
                                ET_pop_obj_func1_list.append(self.R2(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func2_list.append(self.PBIAS(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                ET_pop_obj_func3_list.append(self.RMSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                if self.cal_scheme == 'Single-objective':
                    print('ET_pop_obj_func_list:', len(ET_pop_obj_func_list))
                    self.para_obj_val_dict[self.n_gen] = np.hstack((SWATParas_Sampling, np.array(ET_pop_obj_func_list).reshape(-1, 1)))
                    self.n_gen += 1
                    cal_obj_end_time = time.time()
                    print(f'Calculate Objective Functions Time: {round((cal_obj_end_time - cal_obj_start_time) / 60.0, 2)} min!')
                    print('\n')
                    return ET_pop_obj_func_list
                elif self.cal_scheme == 'Multi-objective':
                    if len(self.objective_funs) == 2:
                        print('ET_pop_obj_func1_list:', len(ET_pop_obj_func1_list))
                        print('ET_pop_obj_func2_list:', len(ET_pop_obj_func2_list))
                        self.para_obj_val_dict[self.n_gen] = np.hstack((SWATParas_Sampling,
                                                                        np.array(ET_pop_obj_func1_list).reshape(-1, 1),
                                                                        np.array(ET_pop_obj_func2_list).reshape(-1, 1)))
                        self.n_gen += 1
                        cal_obj_end_time = time.time()
                        print(f'Calculate Objective Functions Time: {round((cal_obj_end_time - cal_obj_start_time) / 60.0, 2)} min!')
                        print('\n')
                        return ET_pop_obj_func1_list, ET_pop_obj_func2_list
                    elif len(self.objective_funs) == 3:
                        print('ET_pop_obj_func1_list:', len(ET_pop_obj_func1_list))
                        print('ET_pop_obj_func2_list:', len(ET_pop_obj_func2_list))
                        print('ET_pop_obj_func3_list:', len(ET_pop_obj_func3_list))
                        self.para_obj_val_dict[self.n_gen] = np.hstack((SWATParas_Sampling,
                                                                        np.array(ET_pop_obj_func1_list).reshape(-1, 1),
                                                                        np.array(ET_pop_obj_func2_list).reshape(-1, 1),
                                                                        np.array(ET_pop_obj_func3_list).reshape(-1, 1)))
                        self.n_gen += 1
                        cal_obj_end_time = time.time()
                        print(f'Calculate Objective Functions Time: {round((cal_obj_end_time - cal_obj_start_time) / 60.0, 2)} min!')
                        print('\n')
                        return ET_pop_obj_func1_list, ET_pop_obj_func2_list, ET_pop_obj_func3_list

        # Two Variables
        elif self.cal_vars_list == ['Streamflow', 'ET']:
            # Streamflow
            rch_pop_obj_func_list = []
            rch_sta_sim_dict = defaultdict(list)
            mean_season_rch_dict = defaultdict(list)
            eva_metrics_rch_dict = defaultdict(list)
            for rch_sta_idx in sorted(self.hydro_stas.values()):
                rch_obj_list = []
                print('rch_sta_idx:', rch_sta_idx)
                for rch_pop_idx in rch_data_dict_PopList_sort:
                    pop_idx = rch_pop_idx[0]
                    # print('pop_idx:', pop_idx)
                    rch_sta_sim_data = rch_pop_idx[1][rch_sta_idx]
                    # print('rch_sta_sim_data:', rch_sta_sim_data.shape, rch_sta_sim_data)
                    rch_sta_obs_data = self.obs_sf_data[rch_sta_idx]
                    # print('rch_sta_obs_data:', rch_sta_obs_data.shape, rch_sta_obs_data)
                    if self.cal_val_state == 'Calibration':
                        if self.TVSA:
                            rch_obj_list_TVSA = []
                            winsize = 2 * self.half_win + 1
                            for win_idx in range(0, rch_sta_obs_data.shape[0], winsize):
                                rch_sta_obs_data_win = rch_sta_obs_data[win_idx:win_idx + winsize]
                                rch_sta_sim_data_win = rch_sta_sim_data[win_idx:win_idx + winsize]
                                rch_obj_list_TVSA.append(self.RMSE_Moving_Window(rch_sta_obs_data_win, rch_sta_sim_data_win, self.half_win))
                            # print('rch_obj_list_TVSA:', len(rch_obj_list_TVSA))
                            rch_obj_list.append(rch_obj_list_TVSA)
                        else:
                            # Objective Functions (NSE, KGE, R2, PBIAS, RMSE)
                            if self.objective_funs[0] == 'NSE':
                                rch_obj_list.append(self.NSE(rch_sta_obs_data, rch_sta_sim_data))
                            elif self.objective_funs[0] == 'KGE':
                                rch_obj_list.append(self.KGE(rch_sta_obs_data, rch_sta_sim_data))
                            elif self.objective_funs[0] == 'R2':
                                rch_obj_list.append(self.R2(rch_sta_obs_data, rch_sta_sim_data))
                            elif self.objective_funs[0] == 'PBIAS':
                                rch_obj_list.append(self.PBIAS(rch_sta_obs_data, rch_sta_sim_data))
                            elif self.objective_funs[0] == 'RMSE':
                                rch_obj_list.append(self.RMSE(rch_sta_obs_data, rch_sta_sim_data))
                    elif self.cal_val_state == 'Validation':
                        rch_sta_sim_dict[rch_sta_idx].append(rch_sta_sim_data)
                rch_pop_obj_func_list.append(rch_obj_list)
            print('rch_pop_obj_func_list:', len(rch_pop_obj_func_list))
            print('\n')

            # ET
            ET_sim_arr = []
            ET_pop_obj_func_list = []
            FRST_ET_sim_arr, FRST_PRECIP_sim_arr, FRST_SURQ_sim_arr, FRST_LATQ_sim_arr, FRST_GWQ_sim_arr = [], [], [], [], []
            FRSE_ET_sim_arr, FRSE_PRECIP_sim_arr, FRSE_SURQ_sim_arr, FRSE_LATQ_sim_arr, FRSE_GWQ_sim_arr = [], [], [], [], []
            # Objective Functions Mode (Average_Obj_Funcs/Area_Weighted_Avg)
            if self.objec_mode == 'Area_Weighted_Avg':
                # ET simulation
                ET_data_dict_PopList_sort = None
                if self.spatial_unit == 'HRU':
                    if self.water_budget:
                        ET_data_dict_PopList_sort = HRU_data_dict_PopList_sort
                    else:
                        ET_data_dict_PopList_sort = ET_HRU_data_dict_PopList_sort
                elif self.spatial_unit == 'Subbasin':
                    ET_data_dict_PopList_sort = ET_Sub_data_dict_PopList_sort
                ## Area weighted average ET simulation with Multiprocessing
                print('Area weighted average ET simulation using multiprocessing:')
                if self.forest_cal:
                    process_args = [(ET_pop_idx, self.HRU_ID_Veg, self.HRU_veg_area_dict) for ET_pop_idx in ET_data_dict_PopList_sort]
                    print('process_args:', len(process_args))
                    with Pool(processes=5) as p:
                        sim_ET_avg = p.starmap(self.Veg_sim_area_weighted_avg, process_args)
                    ET_data_dict_PopList_sort_w = [sim_et_idx for sim_et_idx in sorted(sim_ET_avg, key=lambda x: x[0])]
                    print('ET_data_dict_PopList_sort_w:', len(ET_data_dict_PopList_sort_w))
                    # Annual Water Budget
                    if self.water_budget:
                        for ET_sim_pop_idx in ET_data_dict_PopList_sort_w:
                            pop_idx = ET_sim_pop_idx[0]
                            # print('pop_idx:', pop_idx)
                            ET_sim_data_dict_area_w_pop = ET_sim_pop_idx[1]
                            if self.cal_val_state == 'Validation':
                                ET_sim_arr.append(ET_sim_data_dict_area_w_pop)
                        print('\n')
                        ## ET, PRECIP, SURQ, LATQ, GWQ
                        print('Area weighted average Annual Water Budget simulation using multiprocessing:')
                        ## FRST
                        process_args_FRST_ET = [(ET_pop_idx, 1, self.HRU_ID_BIOM_Forest['FRST'], self.HRU_BIOM_area_dict_FRST)
                                                for ET_pop_idx in ET_data_dict_PopList_sort]
                        process_args_FRST_PRECIP = [(ET_pop_idx, 2, self.HRU_ID_BIOM_Forest['FRST'], self.HRU_BIOM_area_dict_FRST)
                                                    for ET_pop_idx in ET_data_dict_PopList_sort]
                        process_args_FRST_SURQ = [(ET_pop_idx, 3, self.HRU_ID_BIOM_Forest['FRST'], self.HRU_BIOM_area_dict_FRST)
                                                  for ET_pop_idx in ET_data_dict_PopList_sort]
                        process_args_FRST_LATQ = [(ET_pop_idx, 4, self.HRU_ID_BIOM_Forest['FRST'], self.HRU_BIOM_area_dict_FRST)
                                                  for ET_pop_idx in ET_data_dict_PopList_sort]
                        process_args_FRST_GWQ = [(ET_pop_idx, 5, self.HRU_ID_BIOM_Forest['FRST'], self.HRU_BIOM_area_dict_FRST)
                                                 for ET_pop_idx in ET_data_dict_PopList_sort]
                        print('process_args_FRST_ET:', len(process_args_FRST_ET))
                        print('process_args_FRST_PRECIP:', len(process_args_FRST_PRECIP))
                        print('process_args_FRST_SURQ:', len(process_args_FRST_SURQ))
                        print('process_args_FRST_LATQ:', len(process_args_FRST_LATQ))
                        print('process_args_FRST_GWQ:', len(process_args_FRST_GWQ))
                        with Pool(processes=5) as p:
                            sim_FRST_ET_avg = p.starmap(self.Water_budget_sim_area_weighted_avg, process_args_FRST_ET)
                            sim_FRST_PRECIP_avg = p.starmap(self.Water_budget_sim_area_weighted_avg, process_args_FRST_PRECIP)
                            sim_FRST_SURQ_avg = p.starmap(self.Water_budget_sim_area_weighted_avg, process_args_FRST_SURQ)
                            sim_FRST_LATQ_avg = p.starmap(self.Water_budget_sim_area_weighted_avg, process_args_FRST_LATQ)
                            sim_FRST_GWQ_avg = p.starmap(self.Water_budget_sim_area_weighted_avg, process_args_FRST_GWQ)
                        FRST_ET_data_dict_PopList_sort_w = [sim_FRST_ET_idx for sim_FRST_ET_idx in sorted(sim_FRST_ET_avg, key=lambda x: x[0])]
                        FRST_PRECIP_data_dict_PopList_sort_w = [sim_FRST_PRECIP_idx for sim_FRST_PRECIP_idx in sorted(sim_FRST_PRECIP_avg, key=lambda x: x[0])]
                        FRST_SURQ_data_dict_PopList_sort_w = [sim_FRST_SURQ_idx for sim_FRST_SURQ_idx in sorted(sim_FRST_SURQ_avg, key=lambda x: x[0])]
                        FRST_LATQ_data_dict_PopList_sort_w = [sim_FRST_LATQ_idx for sim_FRST_LATQ_idx in sorted(sim_FRST_LATQ_avg, key=lambda x: x[0])]
                        FRST_GWQ_data_dict_PopList_sort_w = [sim_FRST_GWQ_idx for sim_FRST_GWQ_idx in sorted(sim_FRST_GWQ_avg, key=lambda x: x[0])]
                        print('FRST_ET_data_dict_PopList_sort_w:', len(FRST_ET_data_dict_PopList_sort_w))
                        print('FRST_PRECIP_data_dict_PopList_sort_w:', len(FRST_PRECIP_data_dict_PopList_sort_w))
                        print('FRST_SURQ_data_dict_PopList_sort_w:', len(FRST_SURQ_data_dict_PopList_sort_w))
                        print('FRST_LATQ_data_dict_PopList_sort_w:', len(FRST_LATQ_data_dict_PopList_sort_w))
                        print('FRST_GWQ_data_dict_PopList_sort_w:', len(FRST_GWQ_data_dict_PopList_sort_w))
                        for ET_sim_pop_idx, PRECIP_sim_pop_idx, SURQ_sim_pop_idx, LATQ_sim_pop_idx, GWQ_sim_pop_idx in (
                                zip(FRST_ET_data_dict_PopList_sort_w, FRST_PRECIP_data_dict_PopList_sort_w, FRST_SURQ_data_dict_PopList_sort_w,
                                    FRST_LATQ_data_dict_PopList_sort_w, FRST_GWQ_data_dict_PopList_sort_w)):
                            pop_idx = ET_sim_pop_idx[0]
                            # print('pop_idx:', pop_idx)
                            FRST_ET_sim_data_dict_area_w_pop = ET_sim_pop_idx[1]
                            FRST_PRECIP_sim_data_dict_area_w_pop = PRECIP_sim_pop_idx[1]
                            FRST_SURQ_sim_data_dict_area_w_pop = SURQ_sim_pop_idx[1]
                            FRST_LATQ_sim_data_dict_area_w_pop = LATQ_sim_pop_idx[1]
                            FRST_GWQ_sim_data_dict_area_w_pop = GWQ_sim_pop_idx[1]
                            if self.cal_val_state == 'Validation':
                                FRST_ET_sim_arr.append(FRST_ET_sim_data_dict_area_w_pop)
                                FRST_PRECIP_sim_arr.append(FRST_PRECIP_sim_data_dict_area_w_pop)
                                FRST_SURQ_sim_arr.append(FRST_SURQ_sim_data_dict_area_w_pop)
                                FRST_LATQ_sim_arr.append(FRST_LATQ_sim_data_dict_area_w_pop)
                                FRST_GWQ_sim_arr.append(FRST_GWQ_sim_data_dict_area_w_pop)
                        print('\n')
                        ## FRSE
                        process_args_FRSE_ET = [(ET_pop_idx, 1, self.HRU_ID_BIOM_Forest['FRSE'], self.HRU_BIOM_area_dict_FRSE)
                                                for ET_pop_idx in ET_data_dict_PopList_sort]
                        process_args_FRSE_PRECIP = [(ET_pop_idx, 2, self.HRU_ID_BIOM_Forest['FRSE'], self.HRU_BIOM_area_dict_FRSE)
                                                    for ET_pop_idx in ET_data_dict_PopList_sort]
                        process_args_FRSE_SURQ = [(ET_pop_idx, 3, self.HRU_ID_BIOM_Forest['FRSE'], self.HRU_BIOM_area_dict_FRSE)
                                                  for ET_pop_idx in ET_data_dict_PopList_sort]
                        process_args_FRSE_LATQ = [(ET_pop_idx, 4, self.HRU_ID_BIOM_Forest['FRSE'], self.HRU_BIOM_area_dict_FRSE)
                                                  for ET_pop_idx in ET_data_dict_PopList_sort]
                        process_args_FRSE_GWQ = [(ET_pop_idx, 5, self.HRU_ID_BIOM_Forest['FRSE'], self.HRU_BIOM_area_dict_FRSE)
                                                 for ET_pop_idx in ET_data_dict_PopList_sort]
                        print('process_args_FRSE_ET:', len(process_args_FRSE_ET))
                        print('process_args_FRSE_PRECIP:', len(process_args_FRSE_PRECIP))
                        print('process_args_FRSE_SURQ:', len(process_args_FRSE_SURQ))
                        print('process_args_FRSE_LATQ:', len(process_args_FRSE_LATQ))
                        print('process_args_FRSE_GWQ:', len(process_args_FRSE_GWQ))
                        with Pool(processes=5) as p:
                            sim_FRSE_ET_avg = p.starmap(self.Water_budget_sim_area_weighted_avg, process_args_FRSE_ET)
                            sim_FRSE_PRECIP_avg = p.starmap(self.Water_budget_sim_area_weighted_avg, process_args_FRSE_PRECIP)
                            sim_FRSE_SURQ_avg = p.starmap(self.Water_budget_sim_area_weighted_avg, process_args_FRSE_SURQ)
                            sim_FRSE_LATQ_avg = p.starmap(self.Water_budget_sim_area_weighted_avg, process_args_FRSE_LATQ)
                            sim_FRSE_GWQ_avg = p.starmap(self.Water_budget_sim_area_weighted_avg, process_args_FRSE_GWQ)
                        FRSE_ET_data_dict_PopList_sort_w = [sim_FRSE_ET_idx for sim_FRSE_ET_idx in sorted(sim_FRSE_ET_avg, key=lambda x: x[0])]
                        FRSE_PRECIP_data_dict_PopList_sort_w = [sim_FRSE_PRECIP_idx for sim_FRSE_PRECIP_idx in sorted(sim_FRSE_PRECIP_avg, key=lambda x: x[0])]
                        FRSE_SURQ_data_dict_PopList_sort_w = [sim_FRSE_SURQ_idx for sim_FRSE_SURQ_idx in sorted(sim_FRSE_SURQ_avg, key=lambda x: x[0])]
                        FRSE_LATQ_data_dict_PopList_sort_w = [sim_FRSE_LATQ_idx for sim_FRSE_LATQ_idx in sorted(sim_FRSE_LATQ_avg, key=lambda x: x[0])]
                        FRSE_GWQ_data_dict_PopList_sort_w = [sim_FRSE_GWQ_idx for sim_FRSE_GWQ_idx in sorted(sim_FRSE_GWQ_avg, key=lambda x: x[0])]
                        print('FRSE_ET_data_dict_PopList_sort_w:', len(FRSE_ET_data_dict_PopList_sort_w))
                        print('FRSE_PRECIP_data_dict_PopList_sort_w:', len(FRSE_PRECIP_data_dict_PopList_sort_w))
                        print('FRSE_SURQ_data_dict_PopList_sort_w:', len(FRSE_SURQ_data_dict_PopList_sort_w))
                        print('FRSE_LATQ_data_dict_PopList_sort_w:', len(FRSE_LATQ_data_dict_PopList_sort_w))
                        print('FRSE_GWQ_data_dict_PopList_sort_w:', len(FRSE_GWQ_data_dict_PopList_sort_w))
                        for ET_sim_pop_idx, PRECIP_sim_pop_idx, SURQ_sim_pop_idx, LATQ_sim_pop_idx, GWQ_sim_pop_idx in (
                                zip(FRSE_ET_data_dict_PopList_sort_w, FRSE_PRECIP_data_dict_PopList_sort_w, FRST_SURQ_data_dict_PopList_sort_w,
                                    FRSE_LATQ_data_dict_PopList_sort_w, FRSE_GWQ_data_dict_PopList_sort_w)):
                            pop_idx = ET_sim_pop_idx[0]
                            # print('pop_idx:', pop_idx)
                            FRSE_ET_sim_data_dict_area_w_pop = ET_sim_pop_idx[1]
                            FRSE_PRECIP_sim_data_dict_area_w_pop = PRECIP_sim_pop_idx[1]
                            FRSE_SURQ_sim_data_dict_area_w_pop = SURQ_sim_pop_idx[1]
                            FRSE_LATQ_sim_data_dict_area_w_pop = LATQ_sim_pop_idx[1]
                            FRSE_GWQ_sim_data_dict_area_w_pop = GWQ_sim_pop_idx[1]
                            if self.cal_val_state == 'Validation':
                                FRSE_ET_sim_arr.append(FRSE_ET_sim_data_dict_area_w_pop)
                                FRSE_PRECIP_sim_arr.append(FRSE_PRECIP_sim_data_dict_area_w_pop)
                                FRSE_SURQ_sim_arr.append(FRSE_SURQ_sim_data_dict_area_w_pop)
                                FRSE_LATQ_sim_arr.append(FRSE_LATQ_sim_data_dict_area_w_pop)
                                FRSE_GWQ_sim_arr.append(FRSE_GWQ_sim_data_dict_area_w_pop)
                        print('\n')
                    else:
                        for ET_sim_pop_idx in ET_data_dict_PopList_sort_w:
                            pop_idx = ET_sim_pop_idx[0]
                            # print('pop_idx:', pop_idx)
                            ET_sim_data_dict_area_w_pop = ET_sim_pop_idx[1]
                            if self.TVSA:
                                ET_obj_list_TVSA = []
                                winsize = 2 * self.half_win + 1
                                for win_idx in range(0, self.ET_obs_data_dict_area_w.shape[0], winsize):
                                    ET_obs_data_win = self.ET_obs_data_dict_area_w[win_idx:win_idx + winsize]
                                    ET_sim_data_win = ET_sim_data_dict_area_w_pop[win_idx:win_idx + winsize]
                                    ET_obj_list_TVSA.append(self.RMSE_Moving_Window(ET_obs_data_win, ET_sim_data_win, self.half_win))
                                # print('ET_obj_list_TVSA:', len(ET_obj_list_TVSA))
                                ET_pop_obj_func_list.append(ET_obj_list_TVSA)
                            elif self.cal_val_state == 'Calibration':
                                # Objective Functions (NSE, KGE, R2, PBIAS, RMSE)
                                if self.objective_funs[0] == 'NSE':
                                    ET_pop_obj_func_list.append(self.NSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                elif self.objective_funs[0] == 'KGE':
                                    ET_pop_obj_func_list.append(self.KGE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                elif self.objective_funs[0] == 'R2':
                                    ET_pop_obj_func_list.append(self.R2(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                elif self.objective_funs[0] == 'PBIAS':
                                    ET_pop_obj_func_list.append(self.PBIAS(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                                elif self.objective_funs[0] == 'RMSE':
                                    ET_pop_obj_func_list.append(self.RMSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.cal_val_state == 'Validation':
                                ET_sim_arr.append(ET_sim_data_dict_area_w_pop)
                        print('ET_pop_obj_func_list:', len(ET_pop_obj_func_list))
                else:
                    process_args = [ET_pop_idx for ET_pop_idx in ET_data_dict_PopList_sort]
                    print('process_args:', len(process_args))
                    with Pool(processes=5) as p:
                        sim_ET_avg = p.map(self.RS_sim_area_weighted_avg, process_args)
                    ET_data_dict_PopList_sort_w = [sim_et_idx for sim_et_idx in sorted(sim_ET_avg, key=lambda x: x[0])]
                    print('ET_data_dict_PopList_sort_w:', len(ET_data_dict_PopList_sort_w))
                    #
                    for ET_sim_pop_idx in ET_data_dict_PopList_sort_w:
                        pop_idx = ET_sim_pop_idx[0]
                        # print('pop_idx:', pop_idx)
                        ET_sim_data_dict_area_w_pop = ET_sim_pop_idx[1]
                        if self.TVSA:
                            ET_obj_list_TVSA = []
                            winsize = 2 * self.half_win + 1
                            for win_idx in range(0, self.ET_obs_data_dict_area_w.shape[0], winsize):
                                ET_obs_data_win = self.ET_obs_data_dict_area_w[win_idx:win_idx + winsize]
                                ET_sim_data_win = ET_sim_data_dict_area_w_pop[win_idx:win_idx + winsize]
                                ET_obj_list_TVSA.append(self.RMSE_Moving_Window(ET_obs_data_win, ET_sim_data_win, self.half_win))
                            # print('ET_obj_list_TVSA:', len(ET_obj_list_TVSA))
                            ET_pop_obj_func_list.append(ET_obj_list_TVSA)
                        elif self.cal_val_state == 'Calibration':
                            # Objective Functions (NSE, KGE, R2, PBIAS, RMSE)
                            if self.objective_funs[0] == 'NSE':
                                ET_pop_obj_func_list.append(self.NSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs[0] == 'KGE':
                                ET_pop_obj_func_list.append(self.KGE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs[0] == 'R2':
                                ET_pop_obj_func_list.append(self.R2(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs[0] == 'PBIAS':
                                ET_pop_obj_func_list.append(self.PBIAS(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                            elif self.objective_funs[0] == 'RMSE':
                                ET_pop_obj_func_list.append(self.RMSE(self.ET_obs_data_dict_area_w, ET_sim_data_dict_area_w_pop))
                        elif self.cal_val_state == 'Validation':
                            ET_sim_arr.append(ET_sim_data_dict_area_w_pop)
                    print('ET_pop_obj_func_list:', len(ET_pop_obj_func_list))
            #
            if self.cal_val_state == 'Calibration':
                if not self.SA_flag:
                    self.para_obj_val_dict[self.n_gen] = np.hstack((SWATParas_Sampling,
                                                                    np.array(rch_pop_obj_func_list).T,
                                                                    np.array(ET_pop_obj_func_list).reshape(-1, 1)))
                    self.n_gen += 1
                cal_obj_end_time = time.time()
                print(f'Calculate Objective Functions Time: {round((cal_obj_end_time - cal_obj_start_time) / 60.0, 2)} min!')
                print('\n')
                return rch_pop_obj_func_list, ET_pop_obj_func_list
            elif self.cal_val_state == 'Validation':
                param_val_arr_comp = np.loadtxt(f'{self.swat_nsga_out}\\Compromise_solution_Streamflow_ET.txt',
                                                skiprows=1, usecols=range(len(self.swat_parameter)), dtype=float)
                print('param_val_arr_comp:', param_val_arr_comp.shape, '\n', param_val_arr_comp)
                comp_row_idx = np.where(np.all(SWATParas_Sampling == param_val_arr_comp, axis=1))[0][0]
                date_val = pd.date_range(start=f'{self.val_period[0]}-01-01', end=f'{self.val_period[1]}-12-31', freq='D')
                # Streamflow
                for rch_sta_idx in sorted(self.hydro_stas.values()):
                    print('rch_sta_idx:', np.array(rch_sta_sim_dict[rch_sta_idx]))

                    # Sim_min, Sim_max, Sim_compro_sol
                    rch_sta_sim_dict[rch_sta_idx] = [np.min(np.array(rch_sta_sim_dict[rch_sta_idx]), axis=0),
                                                     np.max(np.array(rch_sta_sim_dict[rch_sta_idx]), axis=0),
                                                     np.array(rch_sta_sim_dict[rch_sta_idx])[comp_row_idx]]
                    # mean_season_obs, mean_season_min, mean_season_max, mean_season_compro_sol
                    mean_season_rch_obs = self.mean_seasonal_cycle(date_val, self.obs_sf_data[rch_sta_idx])
                    mean_season_rch_min = self.mean_seasonal_cycle(date_val, rch_sta_sim_dict[rch_sta_idx][0])
                    mean_season_rch_max = self.mean_seasonal_cycle(date_val, rch_sta_sim_dict[rch_sta_idx][1])
                    mean_season_rch_comp = self.mean_seasonal_cycle(date_val, rch_sta_sim_dict[rch_sta_idx][2])
                    mean_season_rch_dict[rch_sta_idx] = [mean_season_rch_obs, mean_season_rch_min, mean_season_rch_max, mean_season_rch_comp]
                    # Evaluation metric
                    rch_sta_obs_cal = self.obs_sf_data[rch_sta_idx][:len(self.obs_date_day_cal)]
                    rch_sta_obs_val = self.obs_sf_data[rch_sta_idx][len(self.obs_date_day_cal):]
                    #
                    rch_sta_sim_cal = rch_sta_sim_dict[rch_sta_idx][2][:len(self.obs_date_day_cal)]
                    rch_sta_sim_val = rch_sta_sim_dict[rch_sta_idx][2][len(self.obs_date_day_cal):]
                    #
                    rch_sta_cal_KGE = self.KGE(rch_sta_obs_cal, rch_sta_sim_cal)
                    rch_sta_val_KGE = self.KGE(rch_sta_obs_val, rch_sta_sim_val)
                    rch_sta_mean_season_KGE = self.KGE(mean_season_rch_obs, mean_season_rch_comp)
                    #
                    rch_sta_cal_NSE = self.NSE(rch_sta_obs_cal, rch_sta_sim_cal)
                    rch_sta_val_NSE = self.NSE(rch_sta_obs_val, rch_sta_sim_val)
                    rch_sta_mean_season_NSE = self.NSE(mean_season_rch_obs, mean_season_rch_comp)
                    #
                    rch_sta_cal_PBIAS = self.PBIAS(rch_sta_obs_cal, rch_sta_sim_cal)
                    rch_sta_val_PBIAS = self.PBIAS(rch_sta_obs_val, rch_sta_sim_val)
                    rch_sta_mean_season_PBIAS = self.PBIAS(mean_season_rch_obs, mean_season_rch_comp)
                    #
                    eva_metrics_rch_dict[rch_sta_idx] = [(rch_sta_cal_KGE, rch_sta_val_KGE, rch_sta_mean_season_KGE),
                                                         (rch_sta_cal_NSE, rch_sta_val_NSE, rch_sta_mean_season_NSE),
                                                         (rch_sta_cal_PBIAS, rch_sta_val_PBIAS, rch_sta_mean_season_PBIAS)]

                # ET
                print('ET_sim_arr:', np.array(ET_sim_arr))

                # Sim_min, Sim_max, Sim_compro_sol
                ET_sim_arr = [np.min(np.array(ET_sim_arr), axis=0), np.max(np.array(ET_sim_arr), axis=0), np.array(ET_sim_arr)[comp_row_idx]]
                # mean_season_obs, mean_season_min, mean_season_max, mean_season_compro_sol
                mean_season_ET_obs = self.mean_seasonal_cycle(date_val, self.ET_obs_data_dict_area_w)
                mean_season_ET_min = self.mean_seasonal_cycle(date_val, ET_sim_arr[0])
                mean_season_ET_max = self.mean_seasonal_cycle(date_val, ET_sim_arr[1])
                mean_season_ET_comp = self.mean_seasonal_cycle(date_val, ET_sim_arr[2])
                mean_season_ET_dict = [mean_season_ET_obs, mean_season_ET_min, mean_season_ET_max, mean_season_ET_comp]
                # Evaluation metric
                ET_obs_cal = self.ET_obs_data_dict_area_w[:len(self.obs_date_day_cal)]
                ET_obs_val = self.ET_obs_data_dict_area_w[len(self.obs_date_day_cal):]
                #
                ET_sim_cal = ET_sim_arr[2][:len(self.obs_date_day_cal)]
                ET_sim_val = ET_sim_arr[2][len(self.obs_date_day_cal):]
                #
                ET_cal_KGE = self.KGE(ET_obs_cal, ET_sim_cal)
                ET_val_KGE = self.KGE(ET_obs_val, ET_sim_val)
                ET_mean_season_KGE = self.KGE(mean_season_ET_obs, mean_season_ET_comp)
                #
                ET_cal_NSE = self.NSE(ET_obs_cal, ET_sim_cal)
                ET_val_NSE = self.NSE(ET_obs_val, ET_sim_val)
                ET_mean_season_NSE = self.NSE(mean_season_ET_obs, mean_season_ET_comp)
                #
                ET_cal_PBIAS = self.PBIAS(ET_obs_cal, ET_sim_cal)
                ET_val_PBIAS = self.PBIAS(ET_obs_val, ET_sim_val)
                ET_mean_season_PBIAS = self.PBIAS(mean_season_ET_obs, mean_season_ET_comp)
                #
                eva_metrics_ET_dict = [(ET_cal_KGE, ET_val_KGE, ET_mean_season_KGE),
                                       (ET_cal_NSE, ET_val_NSE, ET_mean_season_NSE),
                                       (ET_cal_PBIAS, ET_val_PBIAS, ET_mean_season_PBIAS)]
                if self.water_budget:
                    ## FRST
                    FRST_ET_sim_arr_comp     = np.array(FRST_ET_sim_arr)[comp_row_idx]
                    FRST_PRECIP_sim_arr_comp = np.array(FRST_PRECIP_sim_arr)[comp_row_idx]
                    FRST_SURQ_sim_arr_comp   = np.array(FRST_SURQ_sim_arr)[comp_row_idx]
                    FRST_LATQ_sim_arr_comp   = np.array(FRST_LATQ_sim_arr)[comp_row_idx]
                    FRST_GWQ_sim_arr_comp    = np.array(FRST_GWQ_sim_arr)[comp_row_idx]
                    ##
                    mean_season_FRST_ET_comp     = self.mean_seasonal_cycle(date_val, FRST_ET_sim_arr_comp)
                    mean_season_FRST_PRECIP_comp = self.mean_seasonal_cycle(date_val, FRST_PRECIP_sim_arr_comp)
                    mean_season_FRST_SURQ_comp   = self.mean_seasonal_cycle(date_val, FRST_SURQ_sim_arr_comp)
                    mean_season_FRST_LATQ_comp   = self.mean_seasonal_cycle(date_val, FRST_LATQ_sim_arr_comp)
                    mean_season_FRST_GWQ_comp    = self.mean_seasonal_cycle(date_val, FRST_GWQ_sim_arr_comp)

                    ## FRSE
                    FRSE_ET_sim_arr_comp     = np.array(FRSE_ET_sim_arr)[comp_row_idx]
                    FRSE_PRECIP_sim_arr_comp = np.array(FRSE_PRECIP_sim_arr)[comp_row_idx]
                    FRSE_SURQ_sim_arr_comp   = np.array(FRSE_SURQ_sim_arr)[comp_row_idx]
                    FRSE_LATQ_sim_arr_comp   = np.array(FRSE_LATQ_sim_arr)[comp_row_idx]
                    FRSE_GWQ_sim_arr_comp    = np.array(FRSE_GWQ_sim_arr)[comp_row_idx]
                    ##
                    mean_season_FRSE_ET_comp     = self.mean_seasonal_cycle(date_val, FRSE_ET_sim_arr_comp)
                    mean_season_FRSE_PRECIP_comp = self.mean_seasonal_cycle(date_val, FRSE_PRECIP_sim_arr_comp)
                    mean_season_FRSE_SURQ_comp   = self.mean_seasonal_cycle(date_val, FRSE_SURQ_sim_arr_comp)
                    mean_season_FRSE_LATQ_comp   = self.mean_seasonal_cycle(date_val, FRSE_LATQ_sim_arr_comp)
                    mean_season_FRSE_GWQ_comp    = self.mean_seasonal_cycle(date_val, FRSE_GWQ_sim_arr_comp)

                    return ((rch_sta_sim_dict, ET_sim_arr), (mean_season_rch_dict, mean_season_ET_dict,
                                                             [(mean_season_FRST_ET_comp, mean_season_FRST_PRECIP_comp, mean_season_FRST_SURQ_comp,
                                                               mean_season_FRST_LATQ_comp, mean_season_FRST_GWQ_comp),
                                                              (mean_season_FRSE_ET_comp, mean_season_FRSE_PRECIP_comp, mean_season_FRSE_SURQ_comp,
                                                               mean_season_FRSE_LATQ_comp, mean_season_FRSE_GWQ_comp)]),
                            (eva_metrics_rch_dict, eva_metrics_ET_dict))
                else:
                    return (rch_sta_sim_dict, ET_sim_arr), (mean_season_rch_dict, mean_season_ET_dict), (eva_metrics_rch_dict, eva_metrics_ET_dict)
        elif self.cal_vars_list == ['Streamflow', 'RZSW']:
            pass

        # Three Variables
        elif self.cal_vars_list == ['Streamflow', 'LAI', 'ET']:
            # Streamflow
            print('Streamflow:')
            rch_pop_obj_func_list = []
            for rch_sta_idx in sorted(self.hydro_stas.values()):
                print('rch_sta_idx:', rch_sta_idx)
                rch_obj_list = []
                for rch_pop_idx in rch_data_dict_PopList_sort:
                    pop_idx = rch_pop_idx[0]
                    # print('pop_idx:', pop_idx)
                    rch_sta_sim_data = rch_pop_idx[1][rch_sta_idx]
                    # print('rch_sta_sim_data:', rch_sta_sim_data.shape, rch_sta_sim_data)
                    rch_sta_obs_data = self.obs_sf_data[rch_sta_idx]
                    # print('rch_sta_obs_data:', rch_sta_obs_data.shape, rch_sta_obs_data)
                    if self.TVSA:
                        rch_obj_list_TVSA = []
                        winsize = 2 * self.half_win + 1
                        for win_idx in range(0, rch_sta_obs_data.shape[0], winsize):
                            rch_sta_obs_data_win = rch_sta_obs_data[win_idx:win_idx + winsize]
                            rch_sta_sim_data_win = rch_sta_sim_data[win_idx:win_idx + winsize]
                            rch_obj_list_TVSA.append(self.RMSE_Moving_Window(rch_sta_obs_data_win, rch_sta_sim_data_win, self.half_win))
                        # print('rch_obj_list_TVSA:', len(rch_obj_list_TVSA))
                        rch_obj_list.append(rch_obj_list_TVSA)
                    else:
                        # Objective Functions (NSE, KGE, R2, PBIAS, RMSE)
                        if self.objective_funs[0] == 'NSE':
                            rch_obj_list.append(self.NSE(rch_sta_obs_data, rch_sta_sim_data))
                        elif self.objective_funs[0] == 'KGE':
                            rch_obj_list.append(self.KGE(rch_sta_obs_data, rch_sta_sim_data))
                        elif self.objective_funs[0] == 'R2':
                            rch_obj_list.append(self.R2(rch_sta_obs_data, rch_sta_sim_data))
                        elif self.objective_funs[0] == 'PBIAS':
                            rch_obj_list.append(self.PBIAS(rch_sta_obs_data, rch_sta_sim_data))
                        elif self.objective_funs[0] == 'RMSE':
                            rch_obj_list.append(self.RMSE(rch_sta_obs_data, rch_sta_sim_data))
                rch_pop_obj_func_list.append(rch_obj_list)
                print('rch_pop_obj_func_list:', len(rch_pop_obj_func_list))
            print('\n')

            # LAI/ET
            print('LAI/ET:')
            LAI_pop_obj_func_list, ET_pop_obj_func_list = [], []
            # Objective Functions Mode (Average_Obj_Funcs/Area_Weighted_Avg)
            if self.objec_mode == 'Area_Weighted_Avg':
                ## Area weighted average LAI/ET simulation with Multiprocessing
                if self.forest_cal:
                    # LAI simulation
                    print('Area weighted average LAI simulation using multiprocessing:')
                    LAI_process_args = [(LAI_pop_idx, self.HRU_ID_Veg, self.HRU_veg_area_dict) for LAI_pop_idx in LAI_ET_HRU_data_dict_PopList_sort]
                    print('LAI_process_args:', len(LAI_process_args))
                    with Pool(processes=5) as p:
                        sim_LAI_avg = p.starmap(self.Veg_sim_area_weighted_avg, LAI_process_args)
                    LAI_data_dict_PopList_sort_w = [sim_lai_idx for sim_lai_idx in sorted(sim_LAI_avg, key=lambda x: x[0])]
                    print('LAI_data_dict_PopList_sort_w:', len(LAI_data_dict_PopList_sort_w))
                    #
                    for LAI_sim_pop_idx in LAI_data_dict_PopList_sort_w:
                        pop_idx = LAI_sim_pop_idx[0]
                        # print('pop_idx:', pop_idx)
                        LAI_sim_data_dict_area_w_pop = LAI_sim_pop_idx[1]
                        if self.TVSA:
                            LAI_obj_list_TVSA = []
                            winsize = 2 * self.half_win + 1
                            for win_idx in range(0, self.LAI_obs_data_dict_area_w.shape[0], winsize):
                                LAI_obs_data_win = self.LAI_obs_data_dict_area_w[win_idx:win_idx + winsize]
                                LAI_sim_data_win = LAI_sim_data_dict_area_w_pop[win_idx:win_idx + winsize]
                                LAI_obj_list_TVSA.append(self.RMSE_Moving_Window(LAI_obs_data_win, LAI_sim_data_win, self.half_win))
                            # print('LAI_obj_list_TVSA:', len(LAI_obj_list_TVSA))
                            LAI_pop_obj_func_list.append(LAI_obj_list_TVSA)
                        else:
                            # Objective Functions (NSE, KGE, R2, PBIAS, RMSE)
                            if self.objective_funs[0] == 'NSE':
                                LAI_pop_obj_func_list.append(self.NSE(self.LAI_obs_data_dict_area_w, LAI_sim_pop_idx[1]))
                            elif self.objective_funs[0] == 'KGE':
                                LAI_pop_obj_func_list.append(self.KGE(self.LAI_obs_data_dict_area_w, LAI_sim_pop_idx[1]))
                            elif self.objective_funs[0] == 'R2':
                                LAI_pop_obj_func_list.append(self.R2(self.LAI_obs_data_dict_area_w, LAI_sim_pop_idx[1]))
                            elif self.objective_funs[0] == 'PBIAS':
                                LAI_pop_obj_func_list.append(self.PBIAS(self.LAI_obs_data_dict_area_w, LAI_sim_pop_idx[1]))
                            elif self.objective_funs[0] == 'RMSE':
                                LAI_pop_obj_func_list.append(self.RMSE(self.LAI_obs_data_dict_area_w, LAI_sim_pop_idx[1]))
                    print('LAI_pop_obj_func_list:', len(LAI_pop_obj_func_list))

                    # ET simulation
                    print('Area weighted average ET simulation using multiprocessing:')
                    ET_process_args = [ET_pop_idx for ET_pop_idx in LAI_ET_HRU_data_dict_PopList_sort]
                    print('ET_process_args:', len(ET_process_args))
                    with Pool(processes=5) as p:
                        sim_ET_avg = p.map(self.RS_LAI_ET_sim_area_weighted_avg, ET_process_args)
                    ET_data_dict_PopList_sort_w = [sim_et_idx for sim_et_idx in sorted(sim_ET_avg, key=lambda x: x[0])]
                    print('ET_data_dict_PopList_sort_w:', len(ET_data_dict_PopList_sort_w))
                    #
                    for ET_sim_pop_idx in ET_data_dict_PopList_sort_w:
                        pop_idx = ET_sim_pop_idx[0]
                        # print('pop_idx:', pop_idx)
                        ET_sim_data_dict_area_w_pop = ET_sim_pop_idx[1]
                        if self.TVSA:
                            ET_obj_list_TVSA = []
                            winsize = 2 * self.half_win + 1
                            for win_idx in range(0, self.ET_obs_data_dict_area_w.shape[0], winsize):
                                ET_obs_data_win = self.ET_obs_data_dict_area_w[win_idx:win_idx + winsize]
                                ET_sim_data_win = ET_sim_data_dict_area_w_pop[win_idx:win_idx + winsize]
                                ET_obj_list_TVSA.append(self.RMSE_Moving_Window(ET_obs_data_win, ET_sim_data_win, self.half_win))
                            # print('ET_obj_list_TVSA:', len(ET_obj_list_TVSA))
                            ET_pop_obj_func_list.append(ET_obj_list_TVSA)
                        else:
                            # Objective Functions (NSE, KGE, R2, PBIAS, RMSE)
                            if self.objective_funs[0] == 'NSE':
                                ET_pop_obj_func_list.append(self.NSE(self.ET_obs_data_dict_area_w, ET_sim_pop_idx[1]))
                            elif self.objective_funs[0] == 'KGE':
                                ET_pop_obj_func_list.append(self.KGE(self.ET_obs_data_dict_area_w, ET_sim_pop_idx[1]))
                            elif self.objective_funs[0] == 'R2':
                                ET_pop_obj_func_list.append(self.R2(self.ET_obs_data_dict_area_w, ET_sim_pop_idx[1]))
                            elif self.objective_funs[0] == 'PBIAS':
                                ET_pop_obj_func_list.append(self.PBIAS(self.ET_obs_data_dict_area_w, ET_sim_pop_idx[1]))
                            elif self.objective_funs[0] == 'RMSE':
                                ET_pop_obj_func_list.append(self.RMSE(self.ET_obs_data_dict_area_w, ET_sim_pop_idx[1]))
                    print('ET_pop_obj_func_list:', len(ET_pop_obj_func_list))
                else:
                    # LAI simulation
                    print('Area weighted average LAI simulation using multiprocessing:')
                    LAI_process_args = [(LAI_pop_idx, 1) for LAI_pop_idx in LAI_ET_HRU_data_dict_PopList_sort]
                    print('LAI_process_args:', len(LAI_process_args))
                    with Pool(processes=self.cpu_worker_num) as p:
                        sim_LAI_avg = p.starmap(self.LAI_ET_sim_area_weighted_avg, LAI_process_args)
                    LAI_data_dict_PopList_sort_w = [sim_lai_idx for sim_lai_idx in sorted(sim_LAI_avg, key=lambda x: x[0])]
                    print('LAI_data_dict_PopList_sort_w:', len(LAI_data_dict_PopList_sort_w))
                    #
                    for LAI_sim_pop_idx in LAI_data_dict_PopList_sort_w:
                        pop_idx = LAI_sim_pop_idx[0]
                        # print('pop_idx:', pop_idx)
                        LAI_sim_data_dict_area_w_pop = LAI_sim_pop_idx[1]
                        if self.TVSA:
                            LAI_obj_list_TVSA = []
                            winsize = 2 * self.half_win + 1
                            for win_idx in range(0, self.LAI_obs_data_dict_area_w.shape[0], winsize):
                                LAI_obs_data_win = self.LAI_obs_data_dict_area_w[win_idx:win_idx + winsize]
                                LAI_sim_data_win = LAI_sim_data_dict_area_w_pop[win_idx:win_idx + winsize]
                                LAI_obj_list_TVSA.append(self.RMSE_Moving_Window(LAI_obs_data_win, LAI_sim_data_win, self.half_win))
                            # print('LAI_obj_list_TVSA:', len(LAI_obj_list_TVSA))
                            LAI_pop_obj_func_list.append(LAI_obj_list_TVSA)
                        else:
                            # Objective Functions (NSE, KGE, R2, PBIAS, RMSE)
                            if self.objective_funs[0] == 'NSE':
                                LAI_pop_obj_func_list.append(self.NSE(self.LAI_obs_data_dict_area_w, LAI_sim_pop_idx[1]))
                            elif self.objective_funs[0] == 'KGE':
                                LAI_pop_obj_func_list.append(self.KGE(self.LAI_obs_data_dict_area_w, LAI_sim_pop_idx[1]))
                            elif self.objective_funs[0] == 'R2':
                                LAI_pop_obj_func_list.append(self.R2(self.LAI_obs_data_dict_area_w, LAI_sim_pop_idx[1]))
                            elif self.objective_funs[0] == 'PBIAS':
                                LAI_pop_obj_func_list.append(self.PBIAS(self.LAI_obs_data_dict_area_w, LAI_sim_pop_idx[1]))
                            elif self.objective_funs[0] == 'RMSE':
                                LAI_pop_obj_func_list.append(self.RMSE(self.LAI_obs_data_dict_area_w, LAI_sim_pop_idx[1]))
                    print('LAI_pop_obj_func_list:', len(LAI_pop_obj_func_list))

                    # ET simulation
                    print('Area weighted average ET simulation using multiprocessing:')
                    ## Area weighted average LAI simulation with Multiprocessing
                    print('Area weighted average ET simulation using multiprocessing:')
                    ET_process_args = [(ET_pop_idx, 2) for ET_pop_idx in LAI_ET_HRU_data_dict_PopList_sort]
                    print('ET_process_args:', len(ET_process_args))
                    with Pool(processes=self.cpu_worker_num) as p:
                        sim_ET_avg = p.starmap(self.LAI_ET_sim_area_weighted_avg, ET_process_args)
                    ET_data_dict_PopList_sort_w = [sim_et_idx for sim_et_idx in sorted(sim_ET_avg, key=lambda x: x[0])]
                    print('ET_data_dict_PopList_sort_w:', len(ET_data_dict_PopList_sort_w))
                    #
                    for ET_sim_pop_idx in ET_data_dict_PopList_sort_w:
                        pop_idx = ET_sim_pop_idx[0]
                        # print('pop_idx:', pop_idx)
                        ET_sim_data_dict_area_w_pop = ET_sim_pop_idx[1]
                        if self.TVSA:
                            ET_obj_list_TVSA = []
                            winsize = 2 * self.half_win + 1
                            for win_idx in range(0, self.ET_obs_data_dict_area_w.shape[0], winsize):
                                ET_obs_data_win = self.ET_obs_data_dict_area_w[win_idx:win_idx + winsize]
                                ET_sim_data_win = ET_sim_data_dict_area_w_pop[win_idx:win_idx + winsize]
                                ET_obj_list_TVSA.append(self.RMSE_Moving_Window(ET_obs_data_win, ET_sim_data_win, self.half_win))
                            # print('ET_obj_list_TVSA:', len(ET_obj_list_TVSA))
                            ET_pop_obj_func_list.append(ET_obj_list_TVSA)
                        else:
                            # Objective Functions (NSE, KGE, R2, PBIAS, RMSE)
                            if self.objective_funs[0] == 'NSE':
                                ET_pop_obj_func_list.append(self.NSE(self.ET_obs_data_dict_area_w, ET_sim_pop_idx[1]))
                            elif self.objective_funs[0] == 'KGE':
                                ET_pop_obj_func_list.append(self.KGE(self.ET_obs_data_dict_area_w, ET_sim_pop_idx[1]))
                            elif self.objective_funs[0] == 'R2':
                                ET_pop_obj_func_list.append(self.R2(self.ET_obs_data_dict_area_w, ET_sim_pop_idx[1]))
                            elif self.objective_funs[0] == 'PBIAS':
                                ET_pop_obj_func_list.append(self.PBIAS(self.ET_obs_data_dict_area_w, ET_sim_pop_idx[1]))
                            elif self.objective_funs[0] == 'RMSE':
                                ET_pop_obj_func_list.append(self.RMSE(self.ET_obs_data_dict_area_w, ET_sim_pop_idx[1]))
                    print('ET_pop_obj_func_list:', len(ET_pop_obj_func_list))

            #
            if not self.SA_flag:
                self.para_obj_val_dict[self.n_gen] = np.hstack((SWATParas_Sampling,
                                                                np.array(rch_pop_obj_func_list).T,
                                                                np.array(LAI_pop_obj_func_list).reshape(-1, 1),
                                                                np.array(ET_pop_obj_func_list).reshape(-1, 1)))
                self.n_gen += 1
            cal_obj_end_time = time.time()
            print(f'Calculate Objective Functions Time: {round((cal_obj_end_time - cal_obj_start_time) / 60.0, 2)} min!')
            print('\n')
            return rch_pop_obj_func_list, LAI_pop_obj_func_list, ET_pop_obj_func_list
        elif self.cal_vars_list == ['Streamflow', 'ET', 'RZSW']:
            pass


    def objective_contra(self):
        obj_func_num, constraint_num = None, None
        # One Variables
        if self.cal_vars_list == ['Streamflow']:
            if self.cal_scheme == 'Multi-site':
                if len(self.hydro_stas) == 2:
                    if all(obj in ['NSE', 'KGE', 'R2'] for obj in self.objective_funs):
                        obj_func_num   = 2
                        constraint_num = 2
                    elif all(obj in ['RMSE', 'PBIAS'] for obj in self.objective_funs):
                        obj_func_num   = 2
                        constraint_num = 0
                elif len(self.hydro_stas) == 3:
                    if all(obj in ['NSE', 'KGE', 'R2'] for obj in self.objective_funs):
                        obj_func_num   = 3
                        constraint_num = 0
                    elif all(obj in ['RMSE', 'PBIAS'] for obj in self.objective_funs):
                        obj_func_num   = 3
                        constraint_num = 0
            elif self.cal_scheme == 'Multi-objective':
                if len(self.hydro_stas) == 1:
                    if len(self.objective_funs) == 2:  # Need to consider the input order (Step 8)
                        if self.objective_funs == ['NSE', 'KGE']:
                            obj_func_num   = 2
                            constraint_num = 2
                        elif self.objective_funs == ['NSE', 'R2']:
                            obj_func_num   = 2
                            constraint_num = 2
                        elif self.objective_funs == ['NSE', 'PBIAS']:
                            obj_func_num   = 2
                            constraint_num = 1
                        elif self.objective_funs == ['NSE', 'RMSE']:
                            obj_func_num   = 2
                            constraint_num = 1
                        elif self.objective_funs == ['KGE', 'R2']:
                            obj_func_num   = 2
                            constraint_num = 2
                        elif self.objective_funs == ['KGE', 'PBIAS']:
                            obj_func_num   = 2
                            constraint_num = 1
                        elif self.objective_funs == ['KGE', 'RMSE']:
                            obj_func_num   = 2
                            constraint_num = 1
                        elif self.objective_funs == ['R2', 'PBIAS']:
                            obj_func_num   = 2
                            constraint_num = 0
                        elif self.objective_funs == ['R2', 'RMSE']:
                            obj_func_num   = 2
                            constraint_num = 1
                        elif self.objective_funs == ['PBIAS', 'RMSE']:
                            obj_func_num   = 2
                            constraint_num = 0
                    elif len(self.objective_funs) == 3:  # Need to consider the input order (Step 8)
                        if self.objective_funs == ['NSE', 'KGE', 'R2']:
                            obj_func_num   = 3
                            constraint_num = 3
                        elif self.objective_funs == ['NSE', 'KGE', 'PBIAS']:
                            obj_func_num   = 3
                            constraint_num = 2
                        elif self.objective_funs == ['NSE', 'KGE', 'RMSE']:
                            obj_func_num   = 3
                            constraint_num = 2
                        elif self.objective_funs == ['NSE', 'R2', 'PBIAS']:
                            obj_func_num   = 3
                            constraint_num = 2
                        elif self.objective_funs == ['NSE', 'R2', 'RMSE']:
                            obj_func_num   = 3
                            constraint_num = 2
                        elif self.objective_funs == ['NSE', 'PBIAS', 'RMSE']:
                            obj_func_num   = 3
                            constraint_num = 1
                        elif self.objective_funs == ['KGE', 'R2', 'PBIAS']:
                            obj_func_num   = 3
                            constraint_num = 2
                        elif self.objective_funs == ['KGE', 'R2', 'RMSE']:
                            obj_func_num   = 3
                            constraint_num = 2
                        elif self.objective_funs == ['KGE', 'PBIAS', 'RMSE']:
                            obj_func_num   = 3
                            constraint_num = 1
                        elif self.objective_funs == ['R2', 'PBIAS', 'RMSE']:
                            obj_func_num   = 3
                            constraint_num = 1
                elif len(self.hydro_stas) == 2:
                    if len(self.objective_funs) == 2:  # Need to consider the input order (Step 8)
                        if self.objective_funs == ['NSE', 'KGE']:
                            obj_func_num   = 4
                            constraint_num = 4
                        elif self.objective_funs == ['NSE', 'R2']:
                            obj_func_num   = 4
                            constraint_num = 4
                        elif self.objective_funs == ['NSE', 'PBIAS']:
                            obj_func_num   = 4
                            constraint_num = 2
                        elif self.objective_funs == ['NSE', 'RMSE']:
                            obj_func_num   = 4
                            constraint_num = 2
                        elif self.objective_funs == ['KGE', 'R2']:
                            obj_func_num   = 4
                            constraint_num = 4
                        elif self.objective_funs == ['KGE', 'PBIAS']:
                            obj_func_num   = 4
                            constraint_num = 2
                        elif self.objective_funs == ['KGE', 'RMSE']:
                            obj_func_num   = 4
                            constraint_num = 2
                        elif self.objective_funs == ['R2', 'PBIAS']:
                            obj_func_num   = 4
                            constraint_num = 2
                        elif self.objective_funs == ['R2', 'RMSE']:
                            obj_func_num   = 4
                            constraint_num = 2
                        elif self.objective_funs == ['PBIAS', 'RMSE']:
                            obj_func_num   = 4
                            constraint_num = 0
                    elif len(self.objective_funs) == 3:  # Need to consider the input order (Step 8)
                        if self.objective_funs == ['NSE', 'KGE', 'R2']:
                            obj_func_num   = 6
                            constraint_num = 6
                        elif self.objective_funs == ['NSE', 'KGE', 'PBIAS']:
                            obj_func_num   = 6
                            constraint_num = 4
                        elif self.objective_funs == ['NSE', 'KGE', 'RMSE']:
                            obj_func_num   = 6
                            constraint_num = 4
                        elif self.objective_funs == ['NSE', 'R2', 'PBIAS']:
                            obj_func_num   = 6
                            constraint_num = 4
                        elif self.objective_funs == ['NSE', 'R2', 'RMSE']:
                            obj_func_num   = 6
                            constraint_num = 4
                        elif self.objective_funs == ['NSE', 'PBIAS', 'RMSE']:
                            obj_func_num   = 6
                            constraint_num = 2
                        elif self.objective_funs == ['KGE', 'R2', 'PBIAS']:
                            obj_func_num   = 6
                            constraint_num = 4
                        elif self.objective_funs == ['KGE', 'R2', 'RMSE']:
                            obj_func_num   = 6
                            constraint_num = 4
                        elif self.objective_funs == ['KGE', 'PBIAS', 'RMSE']:
                            obj_func_num   = 6
                            constraint_num = 2
                        elif self.objective_funs == ['R2', 'PBIAS', 'RMSE']:
                            obj_func_num   = 6
                            constraint_num = 2
                elif len(self.hydro_stas) == 3:
                    if len(self.objective_funs) == 2:    # Need to consider the input order (Step 8)
                        if self.objective_funs == ['NSE', 'KGE']:
                            obj_func_num   = 6
                            constraint_num = 6
                        elif self.objective_funs == ['NSE', 'R2']:
                            obj_func_num   = 6
                            constraint_num = 6
                        elif self.objective_funs == ['NSE', 'PBIAS']:
                            obj_func_num   = 6
                            constraint_num = 3
                        elif self.objective_funs == ['NSE', 'RMSE']:
                            obj_func_num   = 6
                            constraint_num = 3
                        elif self.objective_funs == ['KGE', 'R2']:
                            obj_func_num   = 6
                            constraint_num = 6
                        elif self.objective_funs == ['KGE', 'PBIAS']:
                            obj_func_num   = 6
                            constraint_num = 3
                        elif self.objective_funs == ['KGE', 'RMSE']:
                            obj_func_num   = 6
                            constraint_num = 3
                        elif self.objective_funs == ['R2', 'PBIAS']:
                            obj_func_num   = 6
                            constraint_num = 3
                        elif self.objective_funs == ['R2', 'RMSE']:
                            obj_func_num   = 6
                            constraint_num = 3
                        elif self.objective_funs == ['PBIAS', 'RMSE']:
                            obj_func_num   = 6
                            constraint_num = 0
                    elif len(self.objective_funs) == 3:  # Need to consider the input order (Step 8)
                        if self.objective_funs == ['NSE', 'KGE', 'R2']:
                            obj_func_num   = 9
                            constraint_num = 9
                        elif self.objective_funs == ['NSE', 'KGE', 'PBIAS']:
                            obj_func_num   = 9
                            constraint_num = 6
                        elif self.objective_funs == ['NSE', 'KGE', 'RMSE']:
                            obj_func_num   = 9
                            constraint_num = 6
                        elif self.objective_funs == ['NSE', 'R2', 'PBIAS']:
                            obj_func_num   = 9
                            constraint_num = 6
                        elif self.objective_funs == ['NSE', 'R2', 'RMSE']:
                            obj_func_num   = 9
                            constraint_num = 6
                        elif self.objective_funs == ['NSE', 'PBIAS', 'RMSE']:
                            obj_func_num   = 9
                            constraint_num = 3
                        elif self.objective_funs == ['KGE', 'R2', 'PBIAS']:
                            obj_func_num   = 9
                            constraint_num = 6
                        elif self.objective_funs == ['KGE', 'R2', 'RMSE']:
                            obj_func_num   = 9
                            constraint_num = 6
                        elif self.objective_funs == ['KGE', 'PBIAS', 'RMSE']:
                            obj_func_num   = 9
                            constraint_num = 3
                        elif self.objective_funs == ['R2', 'PBIAS', 'RMSE']:
                            obj_func_num   = 9
                            constraint_num = 3
        elif self.cal_vars_list == ['LAI']:
            if self.cal_scheme == 'Multi-objective':
                if len(self.objective_funs) == 2:  # Need to consider the input order (Step 8)
                    if self.objective_funs == ['NSE', 'KGE']:
                        obj_func_num   = 2
                        constraint_num = 2
                    elif self.objective_funs == ['NSE', 'R2']:
                        obj_func_num   = 2
                        constraint_num = 2
                    elif self.objective_funs == ['NSE', 'PBIAS']:
                        obj_func_num   = 2
                        constraint_num = 1
                    elif self.objective_funs == ['NSE', 'RMSE']:
                        obj_func_num   = 2
                        constraint_num = 1
                    elif self.objective_funs == ['KGE', 'R2']:
                        obj_func_num   = 2
                        constraint_num = 2
                    elif self.objective_funs == ['KGE', 'PBIAS']:
                        obj_func_num   = 2
                        constraint_num = 1
                    elif self.objective_funs == ['KGE', 'RMSE']:
                        obj_func_num   = 2
                        constraint_num = 1
                    elif self.objective_funs == ['R2', 'PBIAS']:
                        obj_func_num   = 2
                        constraint_num = 1
                    elif self.objective_funs == ['R2', 'RMSE']:
                        obj_func_num   = 2
                        constraint_num = 1
                    elif self.objective_funs == ['PBIAS', 'RMSE']:
                        obj_func_num   = 2
                        constraint_num = 0
                elif len(self.objective_funs) == 3:  # Need to consider the input order (Step 8)
                    if self.objective_funs == ['NSE', 'KGE', 'R2']:
                        obj_func_num   = 3
                        constraint_num = 3
                    elif self.objective_funs == ['NSE', 'KGE', 'PBIAS']:
                        obj_func_num   = 3
                        constraint_num = 2
                    elif self.objective_funs == ['NSE', 'KGE', 'RMSE']:
                        obj_func_num   = 3
                        constraint_num = 2
                    elif self.objective_funs == ['NSE', 'R2', 'PBIAS']:
                        obj_func_num   = 3
                        constraint_num = 2
                    elif self.objective_funs == ['NSE', 'R2', 'RMSE']:
                        obj_func_num   = 3
                        constraint_num = 2
                    elif self.objective_funs == ['NSE', 'PBIAS', 'RMSE']:
                        obj_func_num   = 3
                        constraint_num = 1
                    elif self.objective_funs == ['KGE', 'R2', 'PBIAS']:
                        obj_func_num   = 3
                        constraint_num = 2
                    elif self.objective_funs == ['KGE', 'R2', 'RMSE']:
                        obj_func_num   = 3
                        constraint_num = 2
                    elif self.objective_funs == ['KGE', 'PBIAS', 'RMSE']:
                        obj_func_num   = 3
                        constraint_num = 1
                    elif self.objective_funs == ['R2', 'PBIAS', 'RMSE']:
                        obj_func_num   = 3
                        constraint_num = 1
            elif self.cal_scheme == 'Single-objective':
                if all(obj in ['NSE', 'KGE', 'R2'] for obj in self.objective_funs):
                    obj_func_num   = 1
                    constraint_num = 1
                elif all(obj in ['RMSE', 'PBIAS'] for obj in self.objective_funs):
                    obj_func_num   = 1
                    constraint_num = 0
        elif self.cal_vars_list == ['BIOM']:
            if self.cal_scheme == 'Multi-objective':
                if len(self.objective_funs) == 2:  # Need to consider the input order (Step 8)
                    if self.objective_funs == ['NSE', 'KGE']:
                        obj_func_num   = 2
                        constraint_num = 0
                    elif self.objective_funs == ['NSE', 'R2']:
                        obj_func_num   = 2
                        constraint_num = 0
                    elif self.objective_funs == ['NSE', 'PBIAS']:
                        obj_func_num   = 2
                        constraint_num = 0
                    elif self.objective_funs == ['NSE', 'RMSE']:
                        obj_func_num   = 2
                        constraint_num = 0
                    elif self.objective_funs == ['KGE', 'R2']:
                        obj_func_num   = 2
                        constraint_num = 0
                    elif self.objective_funs == ['KGE', 'PBIAS']:
                        obj_func_num   = 2
                        constraint_num = 0
                    elif self.objective_funs == ['KGE', 'RMSE']:
                        obj_func_num   = 2
                        constraint_num = 0
                    elif self.objective_funs == ['R2', 'PBIAS']:
                        obj_func_num   = 2
                        constraint_num = 0
                    elif self.objective_funs == ['R2', 'RMSE']:
                        obj_func_num   = 2
                        constraint_num = 0
                    elif self.objective_funs == ['PBIAS', 'RMSE']:
                        obj_func_num   = 2
                        constraint_num = 0
                elif len(self.objective_funs) == 3:  # Need to consider the input order (Step 8)
                    if self.objective_funs == ['NSE', 'KGE', 'R2']:
                        obj_func_num   = 3
                        constraint_num = 0
                    elif self.objective_funs == ['NSE', 'KGE', 'PBIAS']:
                        obj_func_num   = 3
                        constraint_num = 0
                    elif self.objective_funs == ['NSE', 'KGE', 'RMSE']:
                        obj_func_num   = 3
                        constraint_num = 0
                    elif self.objective_funs == ['NSE', 'R2', 'PBIAS']:
                        obj_func_num   = 3
                        constraint_num = 0
                    elif self.objective_funs == ['NSE', 'R2', 'RMSE']:
                        obj_func_num   = 3
                        constraint_num = 0
                    elif self.objective_funs == ['NSE', 'PBIAS', 'RMSE']:
                        obj_func_num   = 3
                        constraint_num = 0
                    elif self.objective_funs == ['KGE', 'R2', 'PBIAS']:
                        obj_func_num   = 3
                        constraint_num = 0
                    elif self.objective_funs == ['KGE', 'R2', 'RMSE']:
                        obj_func_num   = 3
                        constraint_num = 0
                    elif self.objective_funs == ['KGE', 'PBIAS', 'RMSE']:
                        obj_func_num   = 3
                        constraint_num = 0
                    elif self.objective_funs == ['R2', 'PBIAS', 'RMSE']:
                        obj_func_num   = 3
                        constraint_num = 0
            elif self.cal_scheme == 'Single-objective':
                if all(obj in ['NSE', 'KGE', 'R2'] for obj in self.objective_funs):
                    obj_func_num   = 1
                    constraint_num = 0
                elif all(obj in ['RMSE', 'PBIAS'] for obj in self.objective_funs):
                    obj_func_num   = 1
                    constraint_num = 0
        elif self.cal_vars_list == ['ET']:
            if self.cal_scheme == 'Multi-objective':
                if len(self.objective_funs) == 2:  # Need to consider the input order (Step 8)
                    if self.objective_funs == ['NSE', 'KGE']:
                        obj_func_num   = 2
                        constraint_num = 2
                    elif self.objective_funs == ['NSE', 'R2']:
                        obj_func_num   = 2
                        constraint_num = 2
                    elif self.objective_funs == ['NSE', 'PBIAS']:
                        obj_func_num   = 2
                        constraint_num = 1
                    elif self.objective_funs == ['NSE', 'RMSE']:
                        obj_func_num   = 2
                        constraint_num = 1
                    elif self.objective_funs == ['KGE', 'R2']:
                        obj_func_num   = 2
                        constraint_num = 2
                    elif self.objective_funs == ['KGE', 'PBIAS']:
                        obj_func_num   = 2
                        constraint_num = 1
                    elif self.objective_funs == ['KGE', 'RMSE']:
                        obj_func_num   = 2
                        constraint_num = 1
                    elif self.objective_funs == ['R2', 'PBIAS']:
                        obj_func_num   = 2
                        constraint_num = 1
                    elif self.objective_funs == ['R2', 'RMSE']:
                        obj_func_num   = 2
                        constraint_num = 1
                    elif self.objective_funs == ['PBIAS', 'RMSE']:
                        obj_func_num   = 2
                        constraint_num = 0
                elif len(self.objective_funs) == 3:  # Need to consider the input order (Step 8)
                    if self.objective_funs == ['NSE', 'KGE', 'R2']:
                        obj_func_num   = 3
                        constraint_num = 3
                    elif self.objective_funs == ['NSE', 'KGE', 'PBIAS']:
                        obj_func_num   = 3
                        constraint_num = 2
                    elif self.objective_funs == ['NSE', 'KGE', 'RMSE']:
                        obj_func_num   = 3
                        constraint_num = 2
                    elif self.objective_funs == ['NSE', 'R2', 'PBIAS']:
                        obj_func_num   = 3
                        constraint_num = 2
                    elif self.objective_funs == ['NSE', 'R2', 'RMSE']:
                        obj_func_num   = 3
                        constraint_num = 2
                    elif self.objective_funs == ['NSE', 'PBIAS', 'RMSE']:
                        obj_func_num   = 3
                        constraint_num = 1
                    elif self.objective_funs == ['KGE', 'R2', 'PBIAS']:
                        obj_func_num   = 3
                        constraint_num = 2
                    elif self.objective_funs == ['KGE', 'R2', 'RMSE']:
                        obj_func_num   = 3
                        constraint_num = 2
                    elif self.objective_funs == ['KGE', 'PBIAS', 'RMSE']:
                        obj_func_num   = 3
                        constraint_num = 1
                    elif self.objective_funs == ['R2', 'PBIAS', 'RMSE']:
                        obj_func_num   = 3
                        constraint_num = 1
            elif self.cal_scheme == 'Single-objective':
                if all(obj in ['NSE', 'KGE', 'R2'] for obj in self.objective_funs):
                    obj_func_num   = 1
                    constraint_num = 1
                elif all(obj in ['RMSE', 'PBIAS'] for obj in self.objective_funs):
                    obj_func_num   = 1
                    constraint_num = 0

        # Two Variables
        elif self.cal_vars_list == ['Streamflow', 'ET']:
            if len(self.hydro_stas) == 1:
                if all(obj in ['NSE', 'KGE', 'R2'] for obj in self.objective_funs):
                    obj_func_num   = 2
                    constraint_num = 2
                elif all(obj in ['RMSE', 'PBIAS'] for obj in self.objective_funs):
                    obj_func_num   = 2
                    constraint_num = 0
            elif len(self.hydro_stas) == 2:
                if all(obj in ['NSE', 'KGE', 'R2'] for obj in self.objective_funs):
                    obj_func_num   = 3
                    constraint_num = 3
                elif all(obj in ['RMSE', 'PBIAS'] for obj in self.objective_funs):
                    obj_func_num   = 3
                    constraint_num = 0
            elif len(self.hydro_stas) == 3:
                if all(obj in ['NSE', 'KGE', 'R2'] for obj in self.objective_funs):
                    obj_func_num   = 4
                    constraint_num = 0
                elif all(obj in ['RMSE', 'PBIAS'] for obj in self.objective_funs):
                    obj_func_num   = 4
                    constraint_num = 0
        elif self.cal_vars_list == ['Streamflow', 'RZSW']:
            if all(obj in ['NSE', 'KGE', 'R2'] for obj in self.objective_funs):
                obj_func_num   = 4
                constraint_num = 4
            elif all(obj in ['RMSE', 'PBIAS'] for obj in self.objective_funs):
                obj_func_num   = 4
                constraint_num = 1

        # Three Variables
        elif self.cal_vars_list == ['Streamflow', 'LAI', 'ET']:
            if all(obj in ['NSE', 'KGE', 'R2'] for obj in self.objective_funs):
                obj_func_num   = 5
                constraint_num = 5
            elif all(obj in ['RMSE', 'PBIAS'] for obj in self.objective_funs):
                obj_func_num   = 5
                constraint_num = 0
        elif self.cal_vars_list == ['Streamflow', 'ET', 'RZSW']:
            if all(obj in ['NSE', 'KGE', 'R2'] for obj in self.objective_funs):
                obj_func_num   = 5
                constraint_num = 5
            elif all(obj in ['RMSE', 'PBIAS'] for obj in self.objective_funs):
                obj_func_num   = 5
                constraint_num = 0
        return obj_func_num, constraint_num


    # Read SWAT Parameters From swat_par.def
    def read_swat_para(self):
        swat_para = np.loadtxt(fname=os.path.join(self.swat_nsga_in, 'SWAT_Par.def'), dtype=str, skiprows=1, usecols=(0, 1, 2))
        if swat_para.ndim > 1:
            swat_para = swat_para.tolist()
            for para_idx in swat_para:
                para_name = para_idx[0].split('__')[1]
                para_change_type = para_idx[0].split('__')[0]
                if not para_name.endswith('.plant.dat'):
                    self.swat_parameter.append([para_name.split('.')[0], (float(para_idx[1]), float(para_idx[2])), para_change_type, '.' + para_name.split('.')[1]])
                else:
                    self.swat_parameter.append([para_name.split('.')[0], (float(para_idx[1]), float(para_idx[2])), para_change_type, '.plant.dat'])
                para_suffix_list = ['.mgt', '.gw', '.sol', '.rte', '.hru', '.bsn', '.sub', '.plant.dat']
                for suff_idx in para_suffix_list:
                    if para_name.endswith(suff_idx):
                        self.swat_para_catg.setdefault(suff_idx, []).append(para_name.split('.')[0])
                    #
                    if suff_idx == '.mgt' or suff_idx == '.sol':
                        self.swat_para_prec[suff_idx] = 2
                    elif suff_idx == '.rte' or suff_idx == '.hru' or suff_idx == '.bsn' or suff_idx == '.sub':
                        self.swat_para_prec[suff_idx] = 3
                    elif suff_idx == '.gw':
                        self.swat_para_prec[suff_idx] = 4
        else:
            para_idx = swat_para.tolist()
            para_name = para_idx[0].split('__')[1]
            para_change_type = para_idx[0].split('__')[0]
            if not para_name.endswith('.plant.dat'):
                self.swat_parameter.append([para_name.split('.')[0], (float(para_idx[1]), float(para_idx[2])), para_change_type, '.' + para_name.split('.')[1]])
            else:
                self.swat_parameter.append([para_name.split('.')[0], (float(para_idx[1]), float(para_idx[2])), para_change_type, '.plant.dat'])
            para_suffix_list = ['.mgt', '.gw', '.sol', '.rte', '.hru', '.bsn', '.sub', '.plant.dat']
            for suff_idx in para_suffix_list:
                if para_name.endswith(suff_idx):
                    self.swat_para_catg.setdefault(suff_idx, []).append(para_name.split('.')[0])
                #
                if suff_idx == '.mgt' or suff_idx == '.sol':
                    self.swat_para_prec[suff_idx] = 2
                elif suff_idx == '.rte' or suff_idx == '.hru' or suff_idx == '.bsn' or suff_idx == '.sub':
                    self.swat_para_prec[suff_idx] = 3
                elif suff_idx == '.gw':
                    self.swat_para_prec[suff_idx] = 4

        # Soil Parameters
        self.swat_para_sol['SOL_Z']   = 8   # Depth                [mm]
        self.swat_para_sol['SOL_BD']  = 9   # Bulk Density Moist [g/cc]
        self.swat_para_sol['SOL_AWC'] = 10  # Ave. AW Incl. Rock Frag
        self.swat_para_sol['SOL_K']   = 11  # Ksat. (est.)      [mm/hr]
        self.swat_para_sol['SOL_ALB'] = 17  # Soil Albedo (Moist)
        self.swat_para_sol['USLE_K']  = 18  # Erosion K
        self.swat_para_sol['SOL_EC']  = 19  # Salinity (EC, Form 5)

        # Plant Parameters Base Position (plant.dat)
        self.swat_plant_pos['BIO_E']     = (2, 1, 8)  # row, column, precision
        self.swat_plant_pos['HVSTI']     = (2, 2, 8)  # row, column, precision
        self.swat_plant_pos['BLAI']      = (2, 3, 8)  # row, column, precision
        self.swat_plant_pos['FRGRW1']    = (2, 4, 8)  # row, column, precision
        self.swat_plant_pos['LAIMX1']    = (2, 5, 8)  # row, column, precision
        self.swat_plant_pos['FRGRW2']    = (2, 6, 8)  # row, column, precision
        self.swat_plant_pos['LAIMX2']    = (2, 7, 8)  # row, column, precision
        self.swat_plant_pos['DLAI']      = (2, 8, 8)  # row, column, precision
        self.swat_plant_pos['CHTMX']     = (2, 9, 8)  # row, column, precision
        self.swat_plant_pos['T_OPT']     = (3, 1, 8)  # row, column, precision
        self.swat_plant_pos['T_BASE']    = (3, 2, 8)  # row, column, precision
        self.swat_plant_pos['GSI']       = (4, 3, 8)  # row, column, precision
        self.swat_plant_pos['VPDFR']     = (4, 4, 8)  # row, column, precision
        self.swat_plant_pos['FRGMAX']    = (4, 5, 8)  # row, column, precision
        self.swat_plant_pos['ALAI_MIN']  = (4, 10, 8) # row, column, precision
        self.swat_plant_pos['BIO_LEAF']  = (5, 1, 3)  # row, column, precision
        self.swat_plant_pos['MAT_YRS']   = (5, 2, 0)  # row, column, precision
        self.swat_plant_pos['BMX_TREES'] = (5, 3, 3)  # row, column, precision
        self.swat_plant_pos['EXT_COEF']  = (5, 4, 3)  # row, column, precision
        print('self.swat_parameter:', len(self.swat_parameter))
        for para_name_idx in self.swat_parameter:
            print(para_name_idx)
        print('self.swat_para_catg:', len(self.swat_para_catg))
        # print('self.swat_para_sol:', len(self.swat_para_sol))
        # print('self.swat_plant_id:', len(self.swat_plant_id))
        # print('self.swat_plant_pos:', len(self.swat_plant_pos))


    # Search Parameter Files With Specified Suffix
    def search_para_files(self, input_path, pcholder, para_suffix):
        return [os.path.basename(para_file) for para_file in glob.glob(input_path + f'\\{pcholder}{para_suffix}')]


    # Copy all files to a specific target folder
    def copy_to_folder(self, src_file_list, dest_dir):
        for file_name in src_file_list:
            # Overwrite the file in the target folder
            shutil.copy2(f'{self.swat_TxtInOut}\\{file_name}', dest_dir)


    # Modify file.cio File
    def modify_cio_file_v1(self, input_path, input_file, val_dict):
        with open(f'{input_path}/{input_file}', 'r') as f_r, open(f'{input_path}/{input_file[:-4]}.bak', 'w') as f_w:
            txt_lins = f_r.readlines()
            for line in txt_lins:
                if 'NBYR' in line:
                    line_new = line.replace(line,f"              {val_dict['NBYR']}    | NBYR : Number of years simulated\n")
                elif 'IYR' in line:
                    line_new = line.replace(line,f"            {val_dict['IYR']}    | IYR : Beginning year of simulation\n")
                elif 'IPRINT' in line:
                    line_new = line.replace(line,f"               {val_dict['IPRINT']}    | IPRINT: print code (month, day, year)\n")
                elif 'NYSKIP' in line:
                    line_new = line.replace(line,f"               {val_dict['NYSKIP']}    | NYSKIP: number of years to skip output printing/summarization\n")
                else:
                    line_new = line
                f_w.write(line_new)
        os.remove(f'{input_path}/{input_file}')
        os.rename(f'{input_path}/{input_file[:-4]}.bak', f'{input_path}/{input_file}')


    # Modify file.cio File
    def modify_cio_file(self, args):
        input_path = args[0]
        val_dict   = args[1]
        out_print  = args[2]
        with open(f'{input_path}/file.cio', 'r') as f_r, open(f'{input_path}/file.bak', 'w') as f_w:
            txt_lins = f_r.readlines()
            txt_lins[7] = f"               {val_dict['NBYR']}    | NBYR : Number of years simulated\n"
            txt_lins[8] = f"            {val_dict['IYR']}    | IYR : Beginning year of simulation\n"
            txt_lins[58] = f"               {val_dict['IPRINT']}    | IPRINT: print code (month, day, year)\n"
            txt_lins[59] = f"               {val_dict['NYSKIP']}    | NYSKIP: number of years to skip output printing/summarization\n"

            # Print Only Calibration Variables
            if out_print == self.output_var_print[0]:
                # output.rch print
                txt_lins[64] = f'   1   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0\n'

                # output.sub print
                txt_lins[66] = f'   4   5   7   8   0   0   0   0   0   0   0   0   0   0   0\n'

                # output.hru print
                txt_lins[68] = f'   6   8   64   65   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0\n'
            # Print Main Output Variables
            elif out_print == self.output_var_print[1]:
                # output.rch print
                txt_lins[64] = f'   1   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0\n'

                # output.sub print
                txt_lins[66] = f'   4   5   7   8   0   0   0   0   0   0   0   0   0   0   0\n'

                # output.hru print
                txt_lins[68] = f'   1   6   8   18   19   20   21   22   64   65   0   0   0   0   0   0   0   0   0   0\n'
            # Print All Output Variables
            elif out_print == self.output_var_print[2]:
                # output.rch print
                txt_lins[64] = f'   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0\n'

                # output.sub print
                txt_lins[66] = f'   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0\n'

                # output.hru print
                txt_lins[68] = f'   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0\n'

            # Daily Print
            txt_lins[84] = f"               {val_dict['ICALEN']}    | ICALEN: Code for printing out calendar or julian dates to .rch, .sub and .hru files\n"
            f_w.writelines(txt_lins)
        os.remove(f'{input_path}/file.cio')
        os.rename(f'{input_path}/file.bak', f'{input_path}/file.cio')


    # Modify Parameter File
    def modify_para_file_v1(self, input_path, input_file, para_samp_dict, suff_idx_len, para_prec):
        modify_lins_list = {}
        with open(f'{input_path}/{input_file}', 'r') as f_r:
            txt_lins = f_r.readlines()
            for val_key in para_samp_dict.keys():
                for line in txt_lins:
                    if val_key in line.strip().split(':')[0].split():
                        modify_lins_list[val_key] = txt_lins.index(line)
                        break
        with open(f'{input_path}/{input_file}', 'r') as f_r, open(f'{input_path}/{input_file[:-suff_idx_len]}.bak', 'w') as f_w:
            txt_lins = f_r.readlines()
            for line_idx, line in enumerate(txt_lins):
                for val_key in para_samp_dict.keys():
                    line_num = modify_lins_list[val_key]
                    if line_idx == line_num:
                        line_des = '    |' + line.split('|')[1]
                        txt_lins[line_num] = f'{para_samp_dict[val_key]:.{para_prec}f}'.rjust(16, ' ') + line_des
            f_w.writelines(txt_lins)
        os.remove(f'{input_path}/{input_file}')
        os.rename(f'{input_path}/{input_file[:-suff_idx_len]}.bak', f'{input_path}/{input_file}')


    # Modify Parameter File
    def modify_para_file(self, input_path, input_file, para_samp_dict, suff_idx_len, para_prec, para_suff):
        modify_lins_list = {}
        with open(f'{input_path}/{input_file}', 'r') as f_r:
            txt_lins = f_r.readlines()
            for val_key in para_samp_dict.keys():
                for line in txt_lins:
                    if val_key in line.strip().split(':')[0].split():
                        modify_lins_list[val_key] = txt_lins.index(line)
                        break

        with open(f'{input_path}/{input_file}', 'r') as f_r, open(f'{input_path}/{input_file[:-suff_idx_len]}.bak', 'w') as f_w:
            txt_lins = f_r.readlines()
            for line_idx, line in enumerate(txt_lins):
                for val_key in para_samp_dict.keys():
                    line_num = modify_lins_list[val_key]
                    if line_idx == line_num:
                        para_cha_type = para_samp_dict[val_key][0]
                        line_des = '    |' + line.split('|')[1]
                        new_val = 0
                        ori_val = float(line.split('|')[0].strip())
                        if para_cha_type == 'r':
                            new_val = ori_val * (1 + para_samp_dict[val_key][1])
                        elif para_cha_type == 'v':
                            new_val = para_samp_dict[val_key][1]
                        elif para_cha_type == 'a':
                            new_val = para_samp_dict[val_key][1] + ori_val
                        val_length = 16
                        if para_suff == '.rte':
                            val_length = 14
                        txt_lins[line_num] = f'{new_val:.{para_prec}f}'.rjust(val_length, ' ') + line_des
            f_w.writelines(txt_lins)
        os.remove(f'{input_path}/{input_file}')
        os.rename(f'{input_path}/{input_file[:-suff_idx_len]}.bak', f'{input_path}/{input_file}')


    # Modify Soil Parameter File
    def modify_sol_para_file_v1(self, input_path, input_file, para_key_samp_dict, para_prec):
        with open(f'{input_path}/{input_file}', 'r') as f_r, open(f'{input_path}/{input_file[:-3]}.bak', 'w') as f_w:
            txt_lins = f_r.readlines()
            for line_idx, line in enumerate(txt_lins):
                for line_key in para_key_samp_dict.keys():
                    line_num = self.swat_para_sol[line_key] - 1
                    if line_idx == line_num:
                        line_des = line.split(':')[0] + ':'
                        val_layers = line.split(':')[1].split()
                        for sol_para_val in para_key_samp_dict[line_key]:
                            sol_layer_idx = sol_para_val[0][1]
                            if sol_layer_idx == 'All':
                                val_layers = [f'{sol_para_val[1]:.{para_prec}f}' for _ in range(len(val_layers))]
                            else:
                                val_layers[sol_layer_idx - 1] = f'{sol_para_val[1]:.{para_prec}f}'
                        txt_lins_new = ''
                        for val_lay_idx in val_layers:
                            txt_lins_new += val_lay_idx.rjust(12, ' ')
                        txt_lins[line_num] = line_des + txt_lins_new + '\n'
            f_w.writelines(txt_lins)
        os.remove(f'{input_path}/{input_file}')
        os.rename(f'{input_path}/{input_file[:-3]}.bak', f'{input_path}/{input_file}')


    # Modify Soil Parameter File
    def modify_sol_para_file(self, input_path, input_file, para_key_samp_dict, para_prec):
        with open(f'{input_path}/{input_file}', 'r') as f_r, open(f'{input_path}/{input_file[:-3]}.bak', 'w') as f_w:
            txt_lins = f_r.readlines()
            for line_idx, line in enumerate(txt_lins):
                for line_key in para_key_samp_dict.keys():
                    line_num = self.swat_para_sol[line_key] - 1
                    if line_idx == line_num:
                        line_des = line.split(':')[0] + ':'
                        val_layers = np.array([float(lay_idx) for lay_idx in line.split(':')[1].split()])
                        for sol_para_val in para_key_samp_dict[line_key]:
                            sol_layer_idx = sol_para_val[0][1]
                            para_sol_type = sol_para_val[0][2]
                            new_val = np.zeros(shape=val_layers.shape)
                            if para_sol_type == 'r':
                                new_val = val_layers * (1 + sol_para_val[1])
                            elif para_sol_type == 'v':
                                new_val = sol_para_val[1] + new_val
                            elif para_sol_type == 'a':
                                new_val = sol_para_val[1] + val_layers
                            new_val = new_val.tolist()
                            if sol_layer_idx == 'All':
                                val_layers = [new_idx for new_idx in new_val]
                            else:
                                val_layers[sol_layer_idx - 1] = new_val[sol_layer_idx - 1]
                        val_layers = [f'{val_idx:.{para_prec}f}' for val_idx in val_layers]
                        txt_lins_new = ''
                        for val_lay_idx in val_layers:
                            txt_lins_new += val_lay_idx.rjust(12, ' ')
                        txt_lins[line_num] = line_des + txt_lins_new + '\n'
            f_w.writelines(txt_lins)
        os.remove(f'{input_path}/{input_file}')
        os.rename(f'{input_path}/{input_file[:-3]}.bak', f'{input_path}/{input_file}')


    # Modify Plant Parameter File
    def modify_plant_para_file_v1(self, input_path, input_file, para_key_samp_dict):
        with open(f'{input_path}/{input_file}', 'r') as f_r, open(f'{input_path}/{input_file[:-3]}.bak', 'w') as f_w:
            txt_lins = f_r.readlines()
            for line_idx, line in enumerate(txt_lins):
                for line_key in para_key_samp_dict.keys():
                    if line_idx == (line_key - 1):
                        val_paras = line.split()
                        for plant_para_val in para_key_samp_dict[line_key]:
                            val_paras[plant_para_val[0][2][1] - 1] = f'{plant_para_val[1][0]:.{plant_para_val[1][1]}f}'
                        txt_lins_new = ''
                        for val_para_idx in val_paras:
                            txt_lins_new += val_para_idx + '   '
                        txt_lins[line_key - 1] = '  ' + txt_lins_new.strip() + '\n'
            f_w.writelines(txt_lins)
        os.remove(f'{input_path}/{input_file}')
        os.rename(f'{input_path}/{input_file[:-3]}.bak', f'{input_path}/{input_file}')


    # Modify Plant Parameter File
    def modify_plant_para_file(self, input_path, input_file, para_key_samp_dict):
        with open(f'{input_path}/{input_file}', 'r') as f_r, open(f'{input_path}/{input_file[:-3]}.bak', 'w') as f_w:
            txt_lins = f_r.readlines()
            for line_idx, line in enumerate(txt_lins):
                for line_key in para_key_samp_dict.keys():
                    if line_idx == (line_key - 1):
                        val_paras = [val_idx for val_idx in line.split()]
                        for plant_para_val in para_key_samp_dict[line_key]:
                            para_plant_type = plant_para_val[1][0]
                            plant_col = plant_para_val[0][2][1] - 1
                            plant_preci = plant_para_val[1][2]
                            if para_plant_type == 'r':
                                new_val = float(val_paras[plant_col]) * (1 + plant_para_val[1][1])
                            elif para_plant_type == 'v':
                                new_val = plant_para_val[1][1]
                            elif para_plant_type == 'a':
                                new_val = float(val_paras[plant_col]) + plant_para_val[1][1]
                            ## 如果存在MAT_YRS，则进行四舍五入转换为整数值
                            if plant_para_val[0][0] == 'MAT_YRS':
                                new_val = round(new_val, 0)
                            val_paras[plant_col] = f'{new_val:.{plant_preci}f}'  # 保留指定小数位数
                        txt_lins_new = ''
                        for val_para_idx in val_paras:
                            txt_lins_new += val_para_idx + '   '
                        txt_lins[line_key - 1] = '  ' + txt_lins_new.strip() + '\n'
            f_w.writelines(txt_lins)
        os.remove(f'{input_path}/{input_file}')
        os.rename(f'{input_path}/{input_file[:-3]}.bak', f'{input_path}/{input_file}')


    # Main function for Modifying SWAT Project Parameter Files
    def modify_SWAT_para_multiprocessing(self, pop_idx, pop_path, para_sampling_dict):
        print(f'Modify parameters at Population_{pop_idx}')
        for suff_idx in self.swat_para_catg.keys():
            para_name_list = self.swat_para_catg[suff_idx]
            para_file_list = self.search_para_files(pop_path, '?????????', suff_idx)
            para_val_dict = {}
            if self.para_fix_mode or (self.cal_val_state == 'Validation' and self.pop_size == 1):
                for para_name_idx in para_name_list:
                    para_val_dict[para_name_idx] = [para_sampling_dict[para_name_idx][0], para_sampling_dict[para_name_idx][1]]
            else:
                for para_name_idx in para_name_list:
                    para_val_dict[para_name_idx] = [para_sampling_dict[para_name_idx][0], para_sampling_dict[para_name_idx][1][pop_idx - 1]]

            # Parameter Category
            if suff_idx == '.sol':
                para_val_sol_dict = {}
                for para_key in para_val_dict.keys():
                    sol_key = para_key.split('(')[0]
                    if para_key.split('(')[1][:-1] == '':
                        sol_layer = 'All'
                    else:
                        sol_layer = int(para_key.split('(')[1][:-1])
                    para_val_sol_dict[(sol_key, sol_layer, para_val_dict[para_key][0])] = para_val_dict[para_key][1]

                # Aggregating soil parameters from the same row
                para_val_sol_key = set(val_sol_idx[0] for val_sol_idx in para_val_sol_dict.keys())
                para_key_sol_dict = {}
                for key_idx in para_val_sol_key:
                    for sol_key_idx in para_val_sol_dict.keys():
                        if sol_key_idx[0] == key_idx:
                            para_key_sol_dict.setdefault(key_idx, []).append([sol_key_idx, para_val_sol_dict[sol_key_idx]])

                for para_file_idx in para_file_list:
                    self.modify_sol_para_file(pop_path, para_file_idx, para_key_sol_dict, self.swat_para_prec[suff_idx])
            elif suff_idx == '.plant.dat':
                para_file_plant = 'plant.dat'
                para_val_plant_key = set()
                para_val_plant_dict = {}
                for para_key in para_val_dict.keys():
                    plant_key = para_key.split('{')[0]
                    plant_id = int(para_key.split('{')[1][:-1])
                    plant_para_row = self.swat_plant_pos[plant_key][0] + (plant_id - 1) * 5
                    plant_pos = (plant_para_row, self.swat_plant_pos[plant_key][1])
                    plant_precision = self.swat_plant_pos[plant_key][2]
                    para_val_plant_key.add(plant_para_row)
                    para_val_plant_dict[(plant_key, plant_id, plant_pos)] = [para_val_dict[para_key][0], para_val_dict[para_key][1], plant_precision]

                # Aggregating plant parameters from the same row
                para_val_plant_key_list = sorted(list(para_val_plant_key))
                para_key_plant_dict = {}
                for key_idx in para_val_plant_key_list:
                    for plant_key_idx in para_val_plant_dict.keys():
                        if plant_key_idx[2][0] == key_idx:
                            para_key_plant_dict.setdefault(key_idx, []).append([plant_key_idx, para_val_plant_dict[plant_key_idx]])

                self.modify_plant_para_file(pop_path, para_file_plant, para_key_plant_dict)
            elif suff_idx == '.bsn':
                para_file_bsn = 'basins.bsn'
                self.modify_para_file(pop_path, para_file_bsn, para_val_dict, len(suff_idx),
                                      self.swat_para_prec[suff_idx], suff_idx)
            elif (suff_idx == '.mgt' or suff_idx == '.gw' or suff_idx == '.rte' or
                  suff_idx == '.hru' or suff_idx == '.sub'):
                # PHU_PLT和CANMX参数均存在
                if any(para_idx.startswith('PHU_PLT') for para_idx in para_name_list) and any(para_idx.startswith('CANMX') for para_idx in para_name_list):
                    print('This part not finished!')
                # 存在PHU_PLT参数
                elif any(para_idx.startswith('PHU_PLT') for para_idx in para_name_list):
                    # 全部为PHU_PLT参数
                    if all(para_idx.startswith('PHU_PLT') for para_idx in para_name_list):
                        # CANMX参数修改
                        para_name_Land_num = [int(para_idx.split('{')[1][:-1]) for para_idx in [elem for elem in para_name_list if elem.startswith('PHU_PLT')]]
                        para_file_list_FRST_mgt = [item + '.mgt' for item in self.HRU_GIS_Forest[para_name_Land_num[0]]]
                        para_file_list_FRSE_mgt = [item + '.mgt' for item in self.HRU_GIS_Forest[para_name_Land_num[1]]]
                        for para_file_idx in para_file_list:
                            if para_file_idx in para_file_list_FRST_mgt:
                                para_val_dict_FRST = {'PHU_PLT': para_val_dict[f'PHU_PLT{{{para_name_Land_num[0]}}}']}
                                self.modify_para_file(pop_path, para_file_idx, para_val_dict_FRST, len(suff_idx), self.swat_para_prec[suff_idx], suff_idx)
                            elif para_file_idx in para_file_list_FRSE_mgt:
                                para_val_dict_FRSE = {'PHU_PLT': para_val_dict[f'PHU_PLT{{{para_name_Land_num[1]}}}']}
                                self.modify_para_file(pop_path, para_file_idx, para_val_dict_FRSE, len(suff_idx), self.swat_para_prec[suff_idx], suff_idx)
                    # 存在非PHU_PLT参数得.mgt参数
                    else:
                        # CANMX参数修改
                        para_name_Land_num = [int(para_idx.split('{')[1][:-1]) for para_idx in [elem for elem in para_name_list if elem.startswith('PHU_PLT')]]
                        para_file_list_FRST_mgt = [item + '.mgt' for item in self.HRU_GIS_Forest[para_name_Land_num[0]]]
                        para_file_list_FRSE_mgt = [item + '.mgt' for item in self.HRU_GIS_Forest[para_name_Land_num[1]]]
                        for para_file_idx in para_file_list:
                            if para_file_idx in para_file_list_FRST_mgt:
                                para_val_dict_FRST = {'PHU_PLT': para_val_dict[f'PHU_PLT{{{para_name_Land_num[0]}}}']}
                                self.modify_para_file(pop_path, para_file_idx, para_val_dict_FRST, len(suff_idx), self.swat_para_prec[suff_idx], suff_idx)
                            elif para_file_idx in para_file_list_FRSE_mgt:
                                para_val_dict_FRSE = {'PHU_PLT': para_val_dict[f'PHU_PLT{{{para_name_Land_num[1]}}}']}
                                self.modify_para_file(pop_path, para_file_idx, para_val_dict_FRSE, len(suff_idx), self.swat_para_prec[suff_idx], suff_idx)
                        # 非CANMX的.hru参数修改
                        for para_file_idx in para_file_list:
                            para_val_dict_filter = {k:v for k,v in para_val_dict.items() if not k.startswith('PHU_PLT')}
                            self.modify_para_file(pop_path, para_file_idx, para_val_dict_filter, len(suff_idx), self.swat_para_prec[suff_idx], suff_idx)
                # 存在CANMX参数
                elif any(para_idx.startswith('CANMX{') for para_idx in para_name_list):
                    # 全部为CANMX参数
                    if all(para_idx.startswith('CANMX') for para_idx in para_name_list):
                        # CANMX参数修改
                        para_name_Land_num = [int(para_idx.split('{')[1][:-1]) for para_idx in [elem for elem in para_name_list if elem.startswith('CANMX')]]
                        para_file_list_FRST_hru = [item + '.hru' for item in self.HRU_GIS_Forest[para_name_Land_num[0]]]
                        para_file_list_FRSE_hru = [item + '.hru' for item in self.HRU_GIS_Forest[para_name_Land_num[1]]]
                        for para_file_idx in para_file_list:
                            if para_file_idx in para_file_list_FRST_hru:
                                para_val_dict_FRST = {'CANMX': para_val_dict[f'CANMX{{{para_name_Land_num[0]}}}']}
                                self.modify_para_file(pop_path, para_file_idx, para_val_dict_FRST, len(suff_idx), self.swat_para_prec[suff_idx], suff_idx)
                            elif para_file_idx in para_file_list_FRSE_hru:
                                para_val_dict_FRSE = {'CANMX': para_val_dict[f'CANMX{{{para_name_Land_num[1]}}}']}
                                self.modify_para_file(pop_path, para_file_idx, para_val_dict_FRSE, len(suff_idx), self.swat_para_prec[suff_idx], suff_idx)
                    # 存在非CANMX参数的.hru数
                    else:
                        # CANMX参数修改
                        para_name_Land_num = [int(para_idx.split('{')[1][:-1]) for para_idx in [elem for elem in para_name_list if elem.startswith('CANMX')]]
                        para_file_list_FRST_hru = [item + '.hru' for item in self.HRU_GIS_Forest[para_name_Land_num[0]]]
                        para_file_list_FRSE_hru = [item + '.hru' for item in self.HRU_GIS_Forest[para_name_Land_num[1]]]
                        for para_file_idx in para_file_list:
                            if para_file_idx in para_file_list_FRST_hru:
                                para_val_dict_FRST = {'CANMX': para_val_dict[f'CANMX{{{para_name_Land_num[0]}}}']}
                                self.modify_para_file(pop_path, para_file_idx, para_val_dict_FRST, len(suff_idx), self.swat_para_prec[suff_idx], suff_idx)
                            elif para_file_idx in para_file_list_FRSE_hru:
                                para_val_dict_FRSE = {'CANMX': para_val_dict[f'CANMX{{{para_name_Land_num[1]}}}']}
                                self.modify_para_file(pop_path, para_file_idx, para_val_dict_FRSE, len(suff_idx), self.swat_para_prec[suff_idx], suff_idx)
                        # 非CANMX的.hru参数修改
                        for para_file_idx in para_file_list:
                            para_val_dict_filter = {k:v for k,v in para_val_dict.items() if not k.startswith('CANMX')}
                            self.modify_para_file(pop_path, para_file_idx, para_val_dict_filter, len(suff_idx), self.swat_para_prec[suff_idx], suff_idx)
                # 不存在CANMX或者PHU_PLT参数
                else:
                    for para_file_idx in para_file_list:
                        self.modify_para_file(pop_path, para_file_idx, para_val_dict, len(suff_idx), self.swat_para_prec[suff_idx], suff_idx)


    # Main function for Modifying SWAT Project Parameter Files
    def modify_SWAT_para_multiprocessing_v2(self, pop_idx, pop_path, para_sampling_dict):
        print(f'Modify parameters at Population_{pop_idx}')
        for suff_idx in self.swat_para_catg.keys():
            # print('suff_idx:', suff_idx)
            para_name_list = self.swat_para_catg[suff_idx]
            para_file_list = self.search_para_files(pop_path, '?????????', suff_idx)
            para_val_dict = {}
            if self.para_fix_mode or (self.cal_val_state == 'Validation' and self.pop_size == 1):
                for para_name_idx in para_name_list:
                    para_val_dict[para_name_idx] = [para_sampling_dict[para_name_idx][0], para_sampling_dict[para_name_idx][1]]
            else:
                for para_name_idx in para_name_list:
                    para_val_dict[para_name_idx] = [para_sampling_dict[para_name_idx][0], para_sampling_dict[para_name_idx][1][pop_idx - 1]]

            # Parameter Category
            if suff_idx == '.sol':
                para_val_sol_dict = {}
                for para_key in para_val_dict.keys():
                    sol_key = para_key.split('(')[0]
                    if para_key.split('(')[1][:-1] == '':
                        sol_layer = 'All'
                    else:
                        sol_layer = int(para_key.split('(')[1][:-1])
                    para_val_sol_dict[(sol_key, sol_layer, para_val_dict[para_key][0])] = para_val_dict[para_key][1]

                # Aggregating soil parameters from the same row
                para_val_sol_key = set(val_sol_idx[0] for val_sol_idx in para_val_sol_dict.keys())
                para_key_sol_dict = {}
                for key_idx in para_val_sol_key:
                    for sol_key_idx in para_val_sol_dict.keys():
                        if sol_key_idx[0] == key_idx:
                            para_key_sol_dict.setdefault(key_idx, []).append([sol_key_idx, para_val_sol_dict[sol_key_idx]])

                for para_file_idx in para_file_list:
                    self.modify_sol_para_file(pop_path, para_file_idx, para_key_sol_dict, self.swat_para_prec[suff_idx])
            elif suff_idx == '.plant.dat':
                para_file_plant = 'plant.dat'
                para_val_plant_key = set()
                para_val_plant_dict = {}
                for para_key in para_val_dict.keys():
                    plant_key = para_key.split('{')[0]
                    plant_id = int(para_key.split('{')[1][:-1])
                    plant_para_row = self.swat_plant_pos[plant_key][0] + (plant_id - 1) * 5
                    plant_pos = (plant_para_row, self.swat_plant_pos[plant_key][1])
                    plant_precision = self.swat_plant_pos[plant_key][2]
                    para_val_plant_key.add(plant_para_row)
                    para_val_plant_dict[(plant_key, plant_id, plant_pos)] = [para_val_dict[para_key][0], para_val_dict[para_key][1], plant_precision]

                # Aggregating plant parameters from the same row
                para_val_plant_key_list = sorted(list(para_val_plant_key))
                para_key_plant_dict = {}
                for key_idx in para_val_plant_key_list:
                    for plant_key_idx in para_val_plant_dict.keys():
                        if plant_key_idx[2][0] == key_idx:
                            para_key_plant_dict.setdefault(key_idx, []).append([plant_key_idx, para_val_plant_dict[plant_key_idx]])

                self.modify_plant_para_file(pop_path, para_file_plant, para_key_plant_dict)
            elif suff_idx == '.bsn':
                para_file_bsn = 'basins.bsn'
                self.modify_para_file(pop_path, para_file_bsn, para_val_dict, len(suff_idx),
                                      self.swat_para_prec[suff_idx], suff_idx)
            elif (suff_idx == '.mgt' or suff_idx == '.gw' or suff_idx == '.rte' or suff_idx == '.hru' or suff_idx == '.sub'):
                if suff_idx == '.mgt':
                    # 存在PHU_PLT参数（分植被类型）
                    if any(para_idx.startswith('PHU_PLT{') for para_idx in para_name_list):
                        # PHU_PLT参数修改
                        para_Veg_num = [int(para_idx.split('{')[1][:-1]) for para_idx in [elem for elem in para_name_list if elem.startswith('PHU_PLT')]]
                        PHU_PLT_hru_gis_dict = self.get_hru_gis_dict(self.Plant_ID, self.HRU_GIS_ID, para_Veg_num, suff_idx)
                        for PHU_PLT_Veg_id, PHU_PLT_para_files in PHU_PLT_hru_gis_dict.items():
                            for para_file_idx in PHU_PLT_para_files:
                                para_val_dict_PHU_PLT = {'PHU_PLT': para_val_dict[f'PHU_PLT{{{self.Plant_ID_re[PHU_PLT_Veg_id]}}}']}
                                self.modify_para_file(pop_path, para_file_idx, para_val_dict_PHU_PLT, len(suff_idx), self.swat_para_prec[suff_idx], suff_idx)
                        # 非PHU_PLT参数修改
                        if any(not para_idx.startswith('PHU_PLT') for para_idx in para_name_list):
                            for para_file_idx in para_file_list:
                                para_val_dict_filter = {k: v for k, v in para_val_dict.items() if not k.startswith('PHU_PLT')}
                                self.modify_para_file(pop_path, para_file_idx, para_val_dict_filter, len(suff_idx), self.swat_para_prec[suff_idx], suff_idx)
                    else:
                        for para_file_idx in para_file_list:
                            self.modify_para_file(pop_path, para_file_idx, para_val_dict, len(suff_idx), self.swat_para_prec[suff_idx], suff_idx)

                elif suff_idx == '.hru':
                    # 存在CANMX参数（分植被类型）
                    if any(para_idx.startswith('CANMX{') for para_idx in para_name_list):
                        # CANMX参数修改
                        para_Veg_num = [int(para_idx.split('{')[1][:-1]) for para_idx in [elem for elem in para_name_list if elem.startswith('CANMX')]]
                        CANMX_hru_gis_dict = self.get_hru_gis_dict(self.Plant_ID, self.HRU_GIS_ID, para_Veg_num, suff_idx)
                        for CANMX_Veg_id, CANMX_para_files in CANMX_hru_gis_dict.items():
                            for para_file_idx in CANMX_para_files:
                                para_val_dict_CANMX = {'CANMX': para_val_dict[f'CANMX{{{self.Plant_ID_re[CANMX_Veg_id]}}}']}
                                self.modify_para_file(pop_path, para_file_idx, para_val_dict_CANMX, len(suff_idx), self.swat_para_prec[suff_idx], suff_idx)
                        # 非CANMX参数修改
                        if any(not para_idx.startswith('CANMX') for para_idx in para_name_list):
                            for para_file_idx in para_file_list:
                                para_val_dict_filter = {k: v for k, v in para_val_dict.items() if not k.startswith('CANMX')}
                                self.modify_para_file(pop_path, para_file_idx, para_val_dict_filter, len(suff_idx), self.swat_para_prec[suff_idx], suff_idx)
                    else:
                        for para_file_idx in para_file_list:
                            self.modify_para_file(pop_path, para_file_idx, para_val_dict, len(suff_idx), self.swat_para_prec[suff_idx], suff_idx)

                # 不存在分植被类型的CANMX或者PHU_PLT参数
                else:
                    for para_file_idx in para_file_list:
                        self.modify_para_file(pop_path, para_file_idx, para_val_dict, len(suff_idx), self.swat_para_prec[suff_idx], suff_idx)


    # 根据给定的landuse_id获取对应的HRU_GIS，以字典形式返回，并加上指定后缀
    def get_hru_gis_dict(self, plant_database, hru_id_dict, landuse_id_list, suffix):
        hru_gis_dict = {}
        for landuse_idx in landuse_id_list:
            if landuse_idx in plant_database:
                landuse_name = plant_database[landuse_idx]
                if landuse_name in hru_id_dict:
                    # 为HRU_GIS添加后缀
                    hru_gis_dict[landuse_name] = [hru + suffix for hru in hru_id_dict[landuse_name]]
        return hru_gis_dict


    # Run SWAT Model
    def run_SWAT_model(self, run_args):
        pop_idx = run_args[0]
        model_dir = run_args[1]
        print_infor = run_args[3]
        exe_path = ''
        if print_infor == 'Curr_screen':
            exe_path = f'{model_dir}\\{run_args[2]}'
        elif print_infor == 'Pop_folder':
            exe_path  = f'{model_dir}\\{run_args[2]} > Population_{pop_idx}.txt'
        print(f'{os.path.split(model_dir)[-1]} Start Executing:')
        subprocess.run(exe_path, cwd=model_dir, check=True, shell=True)


    # Run SWAT Model
    def run_SWAT_model_v2(self, cmd):
        # model_dir = run_args[0]
        # cmd = run_args[1]
        # print(f'{os.path.split(model_dir)[-1]} Start Executing:')
        # subprocess.run(['cmd', '/c', cmd])
        os.system(cmd)


    # Read Reach Simulation Data
    def read_rch_sim_data_v1(self, input_path, timestep='day'):
        print('Read output.rch:')
        rch_data_dict = {}
        if timestep == 'day':
            rch_data = np.loadtxt(fname=f'{input_path}', skiprows=9, usecols=(1, 3, 4, 5, 7, 8))
            RCH      = rch_data[:, 0]
            MO_rch   = rch_data[:, 1]
            DA_rch   = rch_data[:, 2]
            YR_rch   = rch_data[:, 3]
            FLOW_OUT = rch_data[:, 5]
            for outlet in self.hydro_stas.values():
                print('outlet:', outlet)
                outlet_rch   = RCH[(outlet - 1)::self.reach_num]
                outlet_day   = DA_rch[(outlet - 1)::self.reach_num]
                outlet_month = MO_rch[(outlet - 1)::self.reach_num]
                outlet_year  = YR_rch[(outlet - 1)::self.reach_num]
                outlet_data  = FLOW_OUT[(outlet - 1)::self.reach_num]
                outlet_rch_date = []
                for year_idx, month_idx, day_idx in zip(outlet_year, outlet_month, outlet_day):
                    outlet_rch_date.append(f'{int(year_idx)}-{int(month_idx):02}-{int(day_idx):02}')
                print('outlet_rch_date:', len(outlet_rch_date), outlet_rch_date)

                # Determine whether the outlet value is correct
                if np.all(outlet_rch == outlet):
                    rch_data_dict[outlet] = [outlet_rch_date, outlet_data]
                else:
                    print('Wrong extracting outlet values!')
                    break
        elif timestep == 'month':
            pass
        return rch_data_dict


    # Read Reach Simulation Data
    def read_rch_sim_data(self, args):
        # print('Read output.rch:')
        pop_idx    = args[0]
        input_path = args[1]
        rch_data_dict = {}
        if self.print_key == 'day':
            rch_data = np.loadtxt(fname=f'{input_path}', skiprows=9, usecols=(1, 3, 4, 5, 7, 8))
            RCH      = rch_data[:, 0]
            FLOW_OUT = rch_data[:, 5]
            for outlet in self.hydro_stas.values():
                # print('outlet:', outlet)
                outlet_rch_id   = RCH[(outlet - 1)::self.reach_num]
                outlet_rch_data = FLOW_OUT[(outlet - 1)::self.reach_num]
                # Determine whether the outlet value is correct
                if np.all(outlet_rch_id == outlet):
                    rch_data_dict[outlet] = outlet_rch_data
                else:
                    print('Wrong extracting outlet values!')
                    break
        elif self.print_key == 'month':
            rch_data = np.loadtxt(fname=f'{input_path}', skiprows=9, usecols=(1, 5, 6))
            RCH      = rch_data[:, 0]
            FLOW_OUT = rch_data[:, 2]
            for outlet in self.hydro_stas.values():
                # print('outlet:', outlet)
                outlet_rch_id_list   = []
                outlet_rch_data_list = []
                for row in range(0, RCH.shape[0] - self.reach_num, self.reach_num * 13):
                    outlet_rch_id_year   = RCH[row:row + self.reach_num * 12][(outlet - 1)::self.reach_num]
                    outlet_rch_data_year = FLOW_OUT[row:row + self.reach_num * 12][(outlet - 1)::self.reach_num]
                    outlet_rch_id_list.append(outlet_rch_id_year)
                    outlet_rch_data_list.append(outlet_rch_data_year)
                outlet_rch_id   = np.hstack(outlet_rch_id_list)
                outlet_rch_data = np.hstack(outlet_rch_data_list)

                # Determine whether the outlet value is correct
                if np.all(outlet_rch_id == outlet):
                    rch_data_dict[outlet] = outlet_rch_data
                else:
                    print('Wrong extracting outlet values!')
                    break
        # print('rch_data_dict:', len(rch_data_dict))
        return pop_idx, rch_data_dict


    # Read Subbasin Simulation Data
    def read_sub_sim_data_v1(self, input_path, timestep='day'):
        print('Read output.sub:')
        sub_data_dict = {}
        if timestep == 'day':
            sub_data = np.loadtxt(fname=f'{input_path}', skiprows=9, usecols=(1, 3, 4, 5, 7, 8, 9, 10))
            SUB      = sub_data[:, 0]
            MO_sub   = sub_data[:, 1]
            DA_sub   = sub_data[:, 2]
            YR_sub   = sub_data[:, 3]
            ET_sub   = sub_data[:, 4]
            SW_sub   = sub_data[:, 5]
            SURQ_sub = sub_data[:, 6]
            GWQ_sub  = sub_data[:, 7]
            print('SUB:', SUB)
            print('MO_sub:', MO_sub)
            print('DA_sub:', DA_sub)
            print('YR_sub:', YR_sub)
            print('ET_sub:', ET_sub)
            print('SW_sub:', SW_sub)
            print('SURQ_sub:', SURQ_sub)
            print('GWQ_sub:', GWQ_sub)
            print('\n')
            subbasin_id = set(SUB.astype(int).tolist())
            print('subbasin_id:', subbasin_id)
            for sub_idx in subbasin_id:
                print('sub_idx:', sub_idx)
                sub_id        = SUB[(sub_idx - 1)::self.reach_num]
                sub_day       = DA_sub[(sub_idx - 1)::self.reach_num]
                sub_month     = MO_sub[(sub_idx - 1)::self.reach_num]
                sub_year      = YR_sub[(sub_idx - 1)::self.reach_num]
                ET_sub_data   = ET_sub[(sub_idx - 1)::self.reach_num]
                SW_sub_data   = SW_sub[(sub_idx - 1)::self.reach_num]
                SURQ_sub_data = SURQ_sub[(sub_idx - 1)::self.reach_num]
                GWQ_sub_data  = GWQ_sub[(sub_idx - 1)::self.reach_num]
                print('sub_id:', sub_id)
                print('sub_day:', sub_day)
                print('sub_month:', sub_month)
                print('sub_year:', sub_year)
                print('ET_sub_data:', ET_sub_data)
                print('SW_sub_data:', SW_sub_data)
                print('SURQ_sub_data:', SURQ_sub_data)
                print('GWQ_sub_data:', GWQ_sub_data)
                sub_date = []
                for year_idx, month_idx, day_idx in zip(sub_year, sub_month, sub_day):
                    sub_date.append(f'{int(year_idx)}-{int(month_idx):02}-{int(day_idx):02}')
                print('sub_date:', len(sub_date), sub_date)

                # Determine whether the subbasin value is correct
                if np.all(sub_id == sub_idx):
                    sub_data_dict[sub_idx] = [sub_date, ET_sub_data, SW_sub_data, SURQ_sub_data, GWQ_sub_data]
                else:
                    print('Wrong extracting outlet values!')
                    break

                sys.exit()
        elif timestep == 'month':
            pass


    # Read Subbasin Simulation Data
    def read_sub_sim_data(self, args):
        # print('Read output.sub:')
        pop_idx    = args[0]
        input_path = args[1]
        ET_data_dict = {}
        SW_data_dict = {}
        SURQ_data_dict = {}
        GWQ_data_dict  = {}
        if self.print_key == 'day':
            sub_data = np.loadtxt(fname=f'{input_path}', skiprows=9, usecols=(1, 3, 4, 5, 7, 8, 9, 10))
            SUB      = sub_data[:, 0]
            ET_sub   = sub_data[:, 4]
            SW_sub   = sub_data[:, 5]
            SURQ_sub = sub_data[:, 6]
            GWQ_sub  = sub_data[:, 7]
            subbasin_list = set(SUB.astype(int).tolist())
            # print('subbasin_list:', subbasin_list)
            for sub_idx in subbasin_list:
                sub_id        = SUB[(sub_idx - 1)::self.reach_num]
                ET_sub_data   = ET_sub[(sub_idx - 1)::self.reach_num]
                SW_sub_data   = SW_sub[(sub_idx - 1)::self.reach_num]
                SURQ_sub_data = SURQ_sub[(sub_idx - 1)::self.reach_num]
                GWQ_sub_data  = GWQ_sub[(sub_idx - 1)::self.reach_num]
                # Select Subbasin
                if sub_idx in self.Subbasin_ID:
                    # Determine whether the subbasin value is correct
                    if np.all(sub_id == sub_idx):
                        # ET, SW, SURQ, GWQ
                        ET_data_dict[sub_idx] = ET_sub_data
                        SW_data_dict[sub_idx] = SW_sub_data
                        SURQ_data_dict[sub_idx] = SURQ_sub_data
                        GWQ_data_dict[sub_idx]  = GWQ_sub_data
                    else:
                        print('Wrong extracting subbasin values!')
                        break
        elif self.print_key == 'month':
            sub_data = np.loadtxt(fname=f'{input_path}', skiprows=9, usecols=(1, 4, 5, 6, 7))
            SUB      = sub_data[:, 0]
            ET_sub   = sub_data[:, 1]
            SW_sub   = sub_data[:, 2]
            SURQ_sub = sub_data[:, 3]
            GWQ_sub  = sub_data[:, 4]
            subbasin_list = set(SUB.astype(int).tolist())
            # print('subbasin_list:', subbasin_list)
            for sub_idx in subbasin_list:
                sub_data_id_list   = []
                ET_sub_data_list   = []
                SW_sub_data_list   = []
                SURQ_sub_data_list = []
                GWQ_sub_data_list  = []
                for row in range(0, SUB.shape[0] - self.reach_num, self.reach_num * 13):
                    sub_id_year        = SUB[row:row + self.reach_num * 12][(sub_idx - 1)::self.reach_num]
                    ET_sub_data_year   = ET_sub[row:row + self.reach_num * 12][(sub_idx - 1)::self.reach_num]
                    SW_sub_data_year   = SW_sub[row:row + self.reach_num * 12][(sub_idx - 1)::self.reach_num]
                    SURQ_sub_data_year = SURQ_sub[row:row + self.reach_num * 12][(sub_idx - 1)::self.reach_num]
                    GWQ_sub_data_year  = GWQ_sub[row:row + self.reach_num * 12][(sub_idx - 1)::self.reach_num]
                    sub_data_id_list.append(sub_id_year)
                    ET_sub_data_list.append(ET_sub_data_year)
                    SW_sub_data_list.append(SW_sub_data_year)
                    SURQ_sub_data_list.append(SURQ_sub_data_year)
                    GWQ_sub_data_list.append(GWQ_sub_data_year)
                sub_data_id   = np.hstack(sub_data_id_list)
                ET_sub_data   = np.hstack(ET_sub_data_list)
                SW_sub_data   = np.hstack(SW_sub_data_list)
                SURQ_sub_data = np.hstack(SURQ_sub_data_list)
                GWQ_sub_data  = np.hstack(GWQ_sub_data_list)
                # Select Subbasin
                if sub_idx in self.Subbasin_ID:
                    # Determine whether the subbasin value is correct
                    if np.all(sub_data_id == sub_idx):
                        # ET, SW, SURQ, GWQ
                        ET_data_dict[sub_idx] = ET_sub_data
                        SW_data_dict[sub_idx] = SW_sub_data
                        SURQ_data_dict[sub_idx] = SURQ_sub_data
                        GWQ_data_dict[sub_idx]  = GWQ_sub_data
                    else:
                        print('Wrong extracting subbasin values!')
                        break
        # print('ET_data_dict:', len(ET_data_dict))
        # print('SW_data_dict:', len(SW_data_dict))
        # print('SURQ_data_dict:', len(SURQ_data_dict))
        # print('GWQ_data_dict:', len(GWQ_data_dict))
        return pop_idx, [ET_data_dict, SW_data_dict, SURQ_data_dict, GWQ_data_dict]


    # Read HRU Simulation Data
    def read_hru_sim_data(self, args):
        # print('Read output.hru:')
        pop_idx    = args[0]
        input_path = args[1]
        PRECIP_data_dict = {}
        ET_data_dict   = {}
        SW_data_dict   = {}
        SURQ_data_dict = {}
        LATQ_data_dict = {}
        GWQ_data_dict  = {}
        BIOM_data_dict = {}
        LAI_data_dict  = {}
        if self.print_key == 'day':
            if self.water_budget:
                # HRU(1), PRECIP(9), ET(10), SW(11), SURQ(12), LATQ(14), GWQ(15), WYLD(16), BIOM(17), LAI(18)
                hru_data = np.loadtxt(fname=f'{input_path}', skiprows=9, usecols=(1, 9, 10, 11, 12, 14, 15, 16, 17, 18))
                HRU        = hru_data[:, 0]
                PRECIP_hru = hru_data[:, 1]
                ET_hru     = hru_data[:, 2]
                SW_hru     = hru_data[:, 3]
                SURQ_hru   = hru_data[:, 4]
                LATQ_hru   = hru_data[:, 5]
                GWQ_hru    = hru_data[:, 6]
                BIOM_hru   = hru_data[:, 8]
                LAI_hru    = hru_data[:, 9]
                hru_list   = set(HRU.astype(int).tolist())
                # print('hru_list:', hru_list)
                for hru_idx in hru_list:
                    hru_id = HRU[(hru_idx - 1)::self.hru_num]
                    PRECIP_hru_data = PRECIP_hru[(hru_idx - 1)::self.hru_num]
                    ET_hru_data = ET_hru[(hru_idx - 1)::self.hru_num]
                    SW_hru_data = SW_hru[(hru_idx - 1)::self.hru_num]
                    SURQ_hru_data = SURQ_hru[(hru_idx - 1)::self.hru_num]
                    LATQ_hru_data = LATQ_hru[(hru_idx - 1)::self.hru_num]
                    GWQ_hru_data  = GWQ_hru[(hru_idx - 1)::self.hru_num]
                    BIOM_hru_data = BIOM_hru[(hru_idx - 1)::self.hru_num]
                    LAI_hru_data  = LAI_hru[(hru_idx - 1)::self.hru_num]
                    # Select HRU
                    if hru_idx in self.HRU_ID:
                        # Determine whether the HRU value is correct
                        if np.all(hru_id == hru_idx):
                            # ET, SW, BIOM, LAI
                            PRECIP_data_dict[hru_idx] = PRECIP_hru_data
                            ET_data_dict[hru_idx]   = ET_hru_data
                            SW_data_dict[hru_idx]   = SW_hru_data
                            SURQ_data_dict[hru_idx] = SURQ_hru_data
                            LATQ_data_dict[hru_idx]      = LATQ_hru_data
                            GWQ_data_dict[hru_idx]  = GWQ_hru_data
                            BIOM_data_dict[hru_idx] = BIOM_hru_data
                            LAI_data_dict[hru_idx]  = LAI_hru_data
                        else:
                            print('Wrong extracting HRU values!')
                            break
            else:
                hru_data = np.loadtxt(fname=f'{input_path}', skiprows=9, usecols=(1, 5, 6, 7, 9, 10, 11, 12))
                HRU      = hru_data[:, 0]
                ET_hru   = hru_data[:, 4]
                SW_hru   = hru_data[:, 5]
                BIOM_hru = hru_data[:, 6]
                LAI_hru  = hru_data[:, 7]
                hru_list = set(HRU.astype(int).tolist())
                # print('hru_list:', hru_list)
                for hru_idx in hru_list:
                    hru_id        = HRU[(hru_idx - 1)::self.hru_num]
                    ET_hru_data   = ET_hru[(hru_idx - 1)::self.hru_num]
                    SW_hru_data   = SW_hru[(hru_idx - 1)::self.hru_num]
                    BIOM_hru_data = BIOM_hru[(hru_idx - 1)::self.hru_num]
                    LAI_hru_data  = LAI_hru[(hru_idx - 1)::self.hru_num]
                    # Select HRU
                    if hru_idx in self.HRU_ID:
                        # Determine whether the HRU value is correct
                        if np.all(hru_id == hru_idx):
                            # ET, SW, BIOM, LAI
                            ET_data_dict[hru_idx]   = ET_hru_data
                            SW_data_dict[hru_idx]   = SW_hru_data
                            BIOM_data_dict[hru_idx] = BIOM_hru_data
                            LAI_data_dict[hru_idx]  = LAI_hru_data
                        else:
                            print('Wrong extracting HRU values!')
                            break
        elif self.print_key == 'month':
            hru_data = np.loadtxt(fname=f'{input_path}', skiprows=9, usecols=(1, 6, 7, 8, 9))
            HRU      = hru_data[:, 0]
            ET_hru   = hru_data[:, 1]
            SW_hru   = hru_data[:, 2]
            BIOM_hru = hru_data[:, 3]
            LAI_hru  = hru_data[:, 4]
            hru_list = set(HRU.astype(int).tolist())
            # print('hru_list:', hru_list)
            for hru_idx in hru_list:
                hru_data_id_list   = []
                ET_hru_data_list   = []
                SW_hru_data_list   = []
                BIOM_hru_data_list = []
                LAI_hru_data_list  = []
                for row in range(0, HRU.shape[0] - self.hru_num, self.hru_num * 13):
                    hru_id_year        = HRU[row:row + self.hru_num * 12][(hru_idx - 1)::self.hru_num].astype(int)
                    ET_hru_data_year   = ET_hru[row:row + self.hru_num * 12][(hru_idx - 1)::self.hru_num]
                    SW_hru_data_year   = SW_hru[row:row + self.hru_num * 12][(hru_idx - 1)::self.hru_num]
                    BIOM_hru_data_year = BIOM_hru[row:row + self.hru_num * 12][(hru_idx - 1)::self.hru_num]
                    LAI_hru_data_year  = LAI_hru[row:row + self.hru_num * 12][(hru_idx - 1)::self.hru_num]
                    hru_data_id_list.append(hru_id_year)
                    ET_hru_data_list.append(ET_hru_data_year)
                    SW_hru_data_list.append(SW_hru_data_year)
                    BIOM_hru_data_list.append(BIOM_hru_data_year)
                    LAI_hru_data_list.append(LAI_hru_data_year)
                hru_data_id   = np.hstack(hru_data_id_list)
                ET_hru_data   = np.hstack(ET_hru_data_list)
                SW_hru_data   = np.hstack(SW_hru_data_list)
                BIOM_hru_data = np.hstack(BIOM_hru_data_list)
                LAI_hru_data  = np.hstack(LAI_hru_data_list)
                # Select HRU
                if hru_idx in self.HRU_ID:
                    # Determine whether the HRU value is correct
                    if np.all(hru_data_id == hru_idx):
                        # ET, SW, BIOM, LAI
                        ET_data_dict[hru_idx]   = ET_hru_data
                        SW_data_dict[hru_idx]   = SW_hru_data
                        BIOM_data_dict[hru_idx] = BIOM_hru_data
                        LAI_data_dict[hru_idx]  = LAI_hru_data
                    else:
                        print('Wrong extracting HRU values!')
                        break
        # BIOM
        elif self.print_key == 'year':
            hru_data = np.loadtxt(fname=f'{input_path}', skiprows=9, usecols=(1, 6, 7, 8, 9))
            HRU      = hru_data[:, 0]
            ET_hru   = hru_data[:, 1]
            SW_hru   = hru_data[:, 2]
            BIOM_hru = hru_data[:, 3]
            LAI_hru  = hru_data[:, 4]
            hru_list = set(HRU.astype(int).tolist())
            # print('hru_list:', hru_list)
            for hru_idx in hru_list:
                hru_data_id_list   = []
                ET_hru_data_list   = []
                SW_hru_data_list   = []
                BIOM_hru_data_list = []
                LAI_hru_data_list  = []
                for row in range(0, HRU.shape[0] - self.hru_num, self.hru_num):
                    hru_id_year        = HRU[row:row + self.hru_num][(hru_idx - 1)].astype(int)
                    ET_hru_data_year   = ET_hru[row:row + self.hru_num][(hru_idx - 1)]
                    SW_hru_data_year   = SW_hru[row:row + self.hru_num][(hru_idx - 1)]
                    BIOM_hru_data_year = BIOM_hru[row:row + self.hru_num][(hru_idx - 1)]
                    LAI_hru_data_year  = LAI_hru[row:row + self.hru_num][(hru_idx - 1)]
                    hru_data_id_list.append(hru_id_year)
                    ET_hru_data_list.append(ET_hru_data_year)
                    SW_hru_data_list.append(SW_hru_data_year)
                    BIOM_hru_data_list.append(BIOM_hru_data_year)
                    LAI_hru_data_list.append(LAI_hru_data_year)
                hru_data_id   = np.hstack(hru_data_id_list)
                ET_hru_data   = np.hstack(ET_hru_data_list)
                SW_hru_data   = np.hstack(SW_hru_data_list)
                BIOM_hru_data = np.hstack(BIOM_hru_data_list)
                LAI_hru_data  = np.hstack(LAI_hru_data_list)
                # Select HRU
                if hru_idx in self.HRU_ID:
                    # Determine whether the HRU value is correct
                    if np.all(hru_data_id == hru_idx):
                        # ET, SW, BIOM, LAI
                        ET_data_dict[hru_idx]   = ET_hru_data
                        SW_data_dict[hru_idx]   = SW_hru_data
                        BIOM_data_dict[hru_idx] = BIOM_hru_data
                        LAI_data_dict[hru_idx]  = LAI_hru_data
                    else:
                        print('Wrong extracting HRU values!')
                        break
        # print('ET_data_dict:', len(ET_data_dict))
        # print('SW_data_dict:', len(SW_data_dict))
        # print('BIOM_data_dict:', len(BIOM_data_dict))
        # print('LAI_data_dict:', len(LAI_data_dict))
        if self.water_budget:
            return pop_idx, [ET_data_dict, SW_data_dict, PRECIP_data_dict, SURQ_data_dict, LATQ_data_dict, GWQ_data_dict, BIOM_data_dict, LAI_data_dict]
        else:
            return pop_idx, [ET_data_dict, SW_data_dict, BIOM_data_dict, LAI_data_dict]


    # Read Simulation Multiprocessing (跟模拟数据读入模块重复，已弃用)
    def read_sim_data_multiprocessing(self, input_path):
        rch_path = f'{input_path}\\output.rch'
        sub_path = f'{input_path}\\output.sub'
        hru_path = f'{input_path}\\output.hru'

        # One Variables
        if self.cal_vars_list == ['Streamflow']:
            # output.rch
            rch_data_dict = self.read_rch_sim_data(rch_path)
            return rch_data_dict
        elif self.cal_vars_list == ['LAI']:
            # output.hru
            LAI_HRU_data_dict = self.read_hru_sim_data(hru_path)[3]
            return LAI_HRU_data_dict
        elif self.cal_vars_list == ['BIOM']:
            # output.hru
            BIOM_HRU_data_dict = self.read_hru_sim_data(hru_path)[2]
            return BIOM_HRU_data_dict

        # Two Variables
        elif self.cal_vars_list == ['Streamflow', 'ET']:
            # output.rch
            rch_data_dict = self.read_rch_sim_data(rch_path)
            if self.spatial_unit == 'HRU':
                # output.hru
                ET_HRU_data_dict = self.read_hru_sim_data(hru_path)[0]
                return [rch_data_dict, ET_HRU_data_dict]
            elif self.spatial_unit == 'Subbasin':
                # output.sub
                ET_Sub_data_dict = self.read_sub_sim_data(sub_path)[0]
                return [rch_data_dict, ET_Sub_data_dict]
        elif self.cal_vars_list == ['Streamflow', 'RZSW']:
            # output.rch
            rch_data_dict = self.read_rch_sim_data(rch_path)
            if self.spatial_unit == 'HRU':
                # output.hru
                SW_HRU_data_dict = self.read_hru_sim_data(hru_path)[1]
                return [rch_data_dict, SW_HRU_data_dict]
            elif self.spatial_unit == 'Subbasin':
                # output.sub
                SW_Sub_data_dict = self.read_sub_sim_data(sub_path)[1]
                return [rch_data_dict, SW_Sub_data_dict]

        # Three Variables
        elif self.cal_vars_list == ['Streamflow', 'ET', 'RZSW']:
            # output.rch
            rch_data_dict = self.read_rch_sim_data(rch_path)
            if self.spatial_unit == 'HRU':
                # output.hru
                pop_idx, [ET_HRU_data_dict, SW_HRU_data_dict, BIOM_HRU_data_dict, LAI_HRU_data_dict] = \
                    self.read_hru_sim_data(hru_path)
                return [rch_data_dict, ET_HRU_data_dict, SW_HRU_data_dict]
            elif self.spatial_unit == 'Subbasin':
                # output.sub
                pop_idx, [ET_Sub_data_dict, SW_Sub_data_dict, SURQ_Sub_data_dict, GWQ_Sub_data_dict] = \
                    self.read_sub_sim_data(sub_path)
                return [rch_data_dict, ET_Sub_data_dict, SW_Sub_data_dict]


    # Read Observed Streamflow Data
    def read_obs_streamflow_data(self):
        print('Read Streamflow:')
        rch_obs_data_dict = {}
        NingDu_station_cal, ShiCheng_station_cal, FenKeng_station_cal = [], [], []
        NingDu_station_val, ShiCheng_station_val, FenKeng_station_val = [], [], []
        if self.cal_val_state == 'Calibration':
            if not os.path.exists(f'{self.swat_nsga_in}\\observed_SF_cal_{self.print_key}.txt'):
                if self.print_key == 'day':
                    # Calibration
                    for excel_year_cal in range(self.cal_period[0], self.cal_period[1] + 1):
                        print('excel_year_cal:', excel_year_cal)
                        streamflow_data_cal = pd.read_excel(io=f'{self.sf_obs_path}\\flow_by_day_{excel_year_cal}.xls',
                                                            sheet_name=list(self.hydro_stas.keys()), header=0, index_col=0, nrows=31)
                        NingDu_streamflow_data_cal   = streamflow_data_cal['NingDu'].melt(var_name='day').dropna()['value'].tolist()
                        ShiCheng_streamflow_data_cal = streamflow_data_cal['ShiCheng'].melt(var_name='day').dropna()['value'].tolist()
                        FenKeng_streamflow_data_cal  = streamflow_data_cal['FenKeng'].melt(var_name='day').dropna()['value'].tolist()
                        NingDu_station_cal.extend(NingDu_streamflow_data_cal)
                        ShiCheng_station_cal.extend(ShiCheng_streamflow_data_cal)
                        FenKeng_station_cal.extend(FenKeng_streamflow_data_cal)
                    # Write Observation
                    with open(f'{self.swat_nsga_in}\\observed_SF_cal_{self.print_key}.txt', 'w') as sf_f_cal:
                        sf_f_cal.write('Date'.ljust(14) + 'NingDu'.ljust(12) + 'ShiCheng'.ljust(12) + 'FenKeng'.ljust(12) + '\n')
                        for sf_cal_idx in zip(self.obs_date_day_cal, NingDu_station_cal, ShiCheng_station_cal, FenKeng_station_cal):
                            sf_f_cal.write(f'{sf_cal_idx[0]:<10}    {sf_cal_idx[1]:<8}    {sf_cal_idx[2]:<8}    {sf_cal_idx[3]:<8}\n')
                    return np.array(NingDu_station_cal), np.array(ShiCheng_station_cal), np.array(FenKeng_station_cal)
                elif self.print_key == 'month':
                    # Calibration
                    for excel_year_cal in range(self.cal_period[0], self.cal_period[1] + 1):
                        print('excel_year_cal:', excel_year_cal)
                        streamflow_data_cal = pd.read_excel(io=f'{self.sf_obs_path}\\flow_by_day_{excel_year_cal}.xls',
                                                            sheet_name=list(self.hydro_stas.keys()), header=0, index_col=0, nrows=31)
                        # 计算每列的平均值（排除NaN值）
                        NingDu_streamflow_data_cal   = streamflow_data_cal['NingDu'].apply(lambda col:col.dropna().mean(), axis=0).tolist()
                        ShiCheng_streamflow_data_cal = streamflow_data_cal['ShiCheng'].apply(lambda col:col.dropna().mean(), axis=0).tolist()
                        FenKeng_streamflow_data_cal  = streamflow_data_cal['FenKeng'].apply(lambda col:col.dropna().mean(), axis=0).tolist()
                        NingDu_station_cal.extend(NingDu_streamflow_data_cal)
                        ShiCheng_station_cal.extend(ShiCheng_streamflow_data_cal)
                        FenKeng_station_cal.extend(FenKeng_streamflow_data_cal)
                    # Write Observation
                    with open(f'{self.swat_nsga_in}\\observed_SF_cal_{self.print_key}.txt', 'w') as sf_f_cal:
                        sf_f_cal.write('Date'.ljust(14) + 'NingDu'.ljust(14) + 'ShiCheng'.ljust(14) + 'FenKeng'.ljust(16) + '\n')
                        for sf_cal_idx in zip(self.obs_date_mon_cal, NingDu_station_cal, ShiCheng_station_cal, FenKeng_station_cal):
                            format_x = [f'{num:.6f}' for num in sf_cal_idx if isinstance(num, float)]
                            sf_f_cal.write(f'{sf_cal_idx[0]:<10}    {format_x[0]:<10}    {format_x[1]:<10}    {format_x[2]:<10}\n')
                    return np.array(NingDu_station_cal), np.array(ShiCheng_station_cal), np.array(FenKeng_station_cal)
            else:
                print('Streamflow observation data already exist!')
                if len(self.hydro_stas) == 1:
                    streamflow_data = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_SF_cal_{self.print_key}.txt', skiprows=1, usecols=1)
                    for rch_k, rch_v in zip(self.hydro_stas.values(), [streamflow_data]):
                        rch_obs_data_dict[rch_k] = rch_v
                elif len(self.hydro_stas) == 2:
                    streamflow_data = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_SF_cal_{self.print_key}.txt', skiprows=1, usecols=(1, 2))
                    NingDu_station_cal  = streamflow_data[:, 0]
                    FenKeng_station_cal = streamflow_data[:, 1]
                    for rch_k, rch_v in zip(sorted(self.hydro_stas.values()), [NingDu_station_cal, FenKeng_station_cal]):
                        rch_obs_data_dict[rch_k] = rch_v
                elif len(self.hydro_stas) == 3:
                    streamflow_data = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_SF_cal_{self.print_key}.txt', skiprows=1, usecols=(1, 2, 3))
                    NingDu_station_cal   = streamflow_data[:, 0]
                    ShiCheng_station_cal = streamflow_data[:, 1]
                    FenKeng_station_cal  = streamflow_data[:, 2]
                    for rch_k, rch_v in zip(sorted(self.hydro_stas.values()), [NingDu_station_cal, ShiCheng_station_cal, FenKeng_station_cal]):
                        rch_obs_data_dict[rch_k] = rch_v
                return rch_obs_data_dict
        elif self.cal_val_state == 'Validation':
            if not os.path.exists(f'{self.swat_nsga_in}\\observed_SF_val_{self.print_key}.txt'):
                if self.print_key == 'day':
                    # Validation
                    for excel_year_val in range(self.val_period[0], self.val_period[1] + 1):
                        print('excel_year_val:', excel_year_val)
                        streamflow_data_val = pd.read_excel(io=f'{self.sf_obs_path}\\flow_by_day_{excel_year_val}.xls',
                                                            sheet_name=list(self.hydro_stas.keys()), header=0, index_col=0, nrows=31)
                        NingDu_streamflow_data_val   = streamflow_data_val['NingDu'].melt(var_name='day').dropna()['value'].tolist()
                        ShiCheng_streamflow_data_val = streamflow_data_val['ShiCheng'].melt(var_name='day').dropna()['value'].tolist()
                        FenKeng_streamflow_data_val  = streamflow_data_val['FenKeng'].melt(var_name='day').dropna()['value'].tolist()
                        NingDu_station_val.extend(NingDu_streamflow_data_val)
                        ShiCheng_station_val.extend(ShiCheng_streamflow_data_val)
                        FenKeng_station_val.extend(FenKeng_streamflow_data_val)
                    # Write Observation
                    with open(f'{self.swat_nsga_in}\\observed_SF_val_{self.print_key}.txt', 'w') as sf_f_val:
                        sf_f_val.write('Date'.ljust(14) + 'NingDu'.ljust(12) + 'ShiCheng'.ljust(12) + 'FenKeng'.ljust(12) + '\n')
                        for sf_val_idx in zip(self.obs_date_day_val, NingDu_station_val, ShiCheng_station_val, FenKeng_station_val):
                            sf_f_val.write(f'{sf_val_idx[0]:<10}    {sf_val_idx[1]:<8}    {sf_val_idx[2]:<8}    {sf_val_idx[3]:<8}\n')
                    return np.array(NingDu_station_val), np.array(ShiCheng_station_val), np.array(FenKeng_station_val)
                elif self.print_key == 'month':
                    # Validation
                    for excel_year_val in range(self.val_period[0], self.val_period[1] + 1):
                        print('excel_year_val:', excel_year_val)
                        streamflow_data_val = pd.read_excel(io=f'{self.sf_obs_path}\\flow_by_day_{excel_year_val}.xls',
                                                            sheet_name=list(self.hydro_stas.keys()), header=0, index_col=0, nrows=31)
                        # 计算每列的平均值（排除NaN值）
                        NingDu_streamflow_data_val   = streamflow_data_val['NingDu'].apply(lambda col:col.dropna().mean(), axis=0).tolist()
                        ShiCheng_streamflow_data_val = streamflow_data_val['ShiCheng'].apply(lambda col:col.dropna().mean(), axis=0).tolist()
                        FenKeng_streamflow_data_val  = streamflow_data_val['FenKeng'].apply(lambda col:col.dropna().mean(), axis=0).tolist()
                        NingDu_station_val.extend(NingDu_streamflow_data_val)
                        ShiCheng_station_val.extend(ShiCheng_streamflow_data_val)
                        FenKeng_station_val.extend(FenKeng_streamflow_data_val)
                    # Write Observation
                    with open(f'{self.swat_nsga_in}\\observed_SF_val_{self.print_key}.txt', 'w') as sf_f_val:
                        sf_f_val.write('Date'.ljust(14) + 'NingDu'.ljust(14) + 'ShiCheng'.ljust(14) + 'FenKeng'.ljust(16) + '\n')
                        for sf_val_idx in zip(self.obs_date_mon_val, NingDu_station_val, ShiCheng_station_val, FenKeng_station_val):
                            format_x = [f'{num:.6f}' for num in sf_val_idx if isinstance(num, float)]
                            sf_f_val.write(f'{sf_val_idx[0]:<10}    {format_x[0]:<10}    {format_x[1]:<10}    {format_x[2]:<10}\n')
                    return np.array(NingDu_station_val), np.array(ShiCheng_station_val), np.array(FenKeng_station_val)
            else:
                print('Streamflow observation data already exist!')
                if len(self.hydro_stas) == 1:
                    streamflow_data = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_SF_val_{self.print_key}.txt', skiprows=1, usecols=1)
                    for rch_k, rch_v in zip(self.hydro_stas.values(), [streamflow_data]):
                        rch_obs_data_dict[rch_k] = rch_v
                elif len(self.hydro_stas) == 2:
                    streamflow_data = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_SF_val_{self.print_key}.txt', skiprows=1, usecols=(1, 2))
                    NingDu_station_val  = streamflow_data[:, 0]
                    FenKeng_station_val = streamflow_data[:, 1]
                    for rch_k, rch_v in zip(sorted(self.hydro_stas.values()), [NingDu_station_val, FenKeng_station_val]):
                        rch_obs_data_dict[rch_k] = rch_v
                elif len(self.hydro_stas) == 3:
                    streamflow_data = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_SF_val_{self.print_key}.txt', skiprows=1, usecols=(1, 2, 3))
                    NingDu_station_val   = streamflow_data[:, 0]
                    ShiCheng_station_val = streamflow_data[:, 1]
                    FenKeng_station_val  = streamflow_data[:, 2]
                    for rch_k, rch_v in zip(sorted(self.hydro_stas.values()), [NingDu_station_val, ShiCheng_station_val, FenKeng_station_val]):
                        rch_obs_data_dict[rch_k] = rch_v
                return rch_obs_data_dict


    # Read Observed Surface Runoff/Baseflow Data
    def read_obs_surface_runoff_baseflow_data(self):
        print('Read Surface Runoff/Baseflow:')
        NingDu_surface_runoff_cal, ShiCheng_surface_runoff_cal, FenKeng_surface_runoff_cal = [], [], []
        NingDu_surface_runoff_val, ShiCheng_surface_runoff_val, FenKeng_surface_runoff_val = [], [], []
        NingDu_baseflow_cal, ShiCheng_baseflow_cal, FenKeng_baseflow_cal = [], [], []
        NingDu_baseflow_val, ShiCheng_baseflow_val, FenKeng_baseflow_val = [], [], []
        if self.cal_val_state == 'Calibration':
            # Calibration
            if not (os.path.exists(f'{self.swat_nsga_in}\\observed_SR_cal_{self.print_key}.txt') or
                    os.path.exists(f'{self.swat_nsga_in}\\observed_BF_cal_{self.print_key}.txt')):
                for hydro_sta_idx in self.hydro_stas.keys():
                    Streamflow_obs = []
                    Bflow_Pass1 = []
                    Bflow_Pass2 = []
                    Bflow_Pass3 = []
                    with open(f'{self.bf_obs_path}\\{hydro_sta_idx}.out', 'r') as base_f:
                        base_sep_data = base_f.readlines()[2:6941]
                        base_sep_data_slice = [line[9:] for line in base_sep_data]
                        for base_line in base_sep_data_slice:
                            base_line_val = base_line.split()
                            Streamflow_obs.append(float(base_line_val[0]))
                            Bflow_Pass1.append(float(base_line_val[1]))
                            Bflow_Pass2.append(float(base_line_val[2]))
                            Bflow_Pass3.append(float(base_line_val[3]))
                    Baseflow_data_cal = [(p1 + p2) / 2 for p1, p2 in zip(Bflow_Pass1[730:4017], Bflow_Pass2[730:4017])]
                    Surface_runoff_data_cal = [sf - bf for sf, bf in zip(Streamflow_obs[730:4017], Baseflow_data_cal)]
                    # Hydrological stations
                    if hydro_sta_idx == 'NingDu':
                        NingDu_surface_runoff_cal = Surface_runoff_data_cal
                        NingDu_baseflow_cal = Baseflow_data_cal
                    elif hydro_sta_idx == 'ShiCheng':
                        ShiCheng_surface_runoff_cal = Surface_runoff_data_cal
                        ShiCheng_baseflow_cal = Baseflow_data_cal
                    elif hydro_sta_idx == 'FenKeng':
                        FenKeng_surface_runoff_cal = Surface_runoff_data_cal
                        FenKeng_baseflow_cal = Baseflow_data_cal

                if self.print_key == 'day':
                    # Write Surface Runoff Observation
                    with open(f'{self.swat_nsga_in}\\observed_SR_cal_{self.print_key}.txt', 'w') as sr_f_cal:
                        sr_f_cal.write('Date'.ljust(14) + 'NingDu'.ljust(14) + 'ShiCheng'.ljust(14) + 'FenKeng'.ljust(16) + '\n')
                        for sr_cal_idx in zip(self.obs_date_day_cal, NingDu_surface_runoff_cal, ShiCheng_surface_runoff_cal, FenKeng_surface_runoff_cal):
                            format_x = [f'{num:.6f}' for num in sr_cal_idx if isinstance(num, float)]
                            sr_f_cal.write(f'{sr_cal_idx[0]:<10}    {format_x[0]:<10}    {format_x[1]:<10}    {format_x[2]:<10}\n')
                    # Write Baseflow Observation
                    with open(f'{self.swat_nsga_in}\\observed_BF_cal_{self.print_key}.txt', 'w') as bf_f_cal:
                        bf_f_cal.write('Date'.ljust(14) + 'NingDu'.ljust(14) + 'ShiCheng'.ljust(14) + 'FenKeng'.ljust(16) + '\n')
                        for bf_cal_idx in zip(self.obs_date_day_cal, NingDu_baseflow_cal, ShiCheng_baseflow_cal, FenKeng_baseflow_cal):
                            format_x = [f'{num:.6f}' for num in bf_cal_idx if isinstance(num, float)]
                            bf_f_cal.write(f'{bf_cal_idx[0]:<10}    {format_x[0]:<10}    {format_x[1]:<10}    {format_x[2]:<10}\n')

                    return ([np.array(NingDu_surface_runoff_cal), np.array(NingDu_baseflow_cal)],
                            [np.array(ShiCheng_surface_runoff_cal), np.array(ShiCheng_baseflow_cal)],
                            [np.array(FenKeng_surface_runoff_cal), np.array(FenKeng_baseflow_cal)])
                elif self.print_key == 'month':
                    # Monthly Average
                    obs_date_day_cal = pd.date_range(start=f'{self.cal_period[0]}-01-01', end=f'{self.cal_period[1]}-12-31', freq='D')
                    NingDu_surface_runoff_mon   = pd.DataFrame({'date': obs_date_day_cal, 'value': NingDu_surface_runoff_cal})
                    ShiCheng_surface_runoff_mon = pd.DataFrame({'date': obs_date_day_cal, 'value': ShiCheng_surface_runoff_cal})
                    FenKeng_surface_runoff_mon  = pd.DataFrame({'date': obs_date_day_cal, 'value': FenKeng_surface_runoff_cal})
                    # 按月计算平均值
                    NingDu_surface_runoff_mon_avg   = NingDu_surface_runoff_mon.groupby(pd.Grouper(key='date', freq='M')).mean()['value'].tolist()
                    ShiCheng_surface_runoff_mon_avg = ShiCheng_surface_runoff_mon.groupby(pd.Grouper(key='date', freq='M')).mean()['value'].tolist()
                    FenKeng_surface_runoff_mon_avg  = FenKeng_surface_runoff_mon.groupby(pd.Grouper(key='date', freq='M')).mean()['value'].tolist()

                    NingDu_baseflow_mon   = pd.DataFrame({'date': obs_date_day_cal, 'value': NingDu_baseflow_cal})
                    ShiCheng_baseflow_mon = pd.DataFrame({'date': obs_date_day_cal, 'value': ShiCheng_baseflow_cal})
                    FenKeng_baseflow_mon  = pd.DataFrame({'date': obs_date_day_cal, 'value': FenKeng_baseflow_cal})
                    # 按月计算平均值
                    NingDu_baseflow_mon_avg   = NingDu_baseflow_mon.groupby(pd.Grouper(key='date', freq='M')).mean()['value'].tolist()
                    ShiCheng_baseflow_mon_avg = ShiCheng_baseflow_mon.groupby(pd.Grouper(key='date', freq='M')).mean()['value'].tolist()
                    FenKeng_baseflow_mon_avg  = FenKeng_baseflow_mon.groupby(pd.Grouper(key='date', freq='M')).mean()['value'].tolist()

                    # Write Surface Runoff Observation
                    with open(f'{self.swat_nsga_in}\\observed_SR_cal_{self.print_key}.txt', 'w') as sr_f_cal:
                        sr_f_cal.write('Date'.ljust(14) + 'NingDu'.ljust(14) + 'ShiCheng'.ljust(14) + 'FenKeng'.ljust(16) + '\n')
                        for sr_cal_idx in zip(self.obs_date_mon_cal, NingDu_surface_runoff_mon_avg, ShiCheng_surface_runoff_mon_avg, FenKeng_surface_runoff_mon_avg):
                            format_x = [f'{num:.6f}' for num in sr_cal_idx if isinstance(num, float)]
                            sr_f_cal.write(f'{sr_cal_idx[0]:<10}    {format_x[0]:<10}    {format_x[1]:<10}    {format_x[2]:<10}\n')
                    # Write Baseflow Observation
                    with open(f'{self.swat_nsga_in}\\observed_BF_cal_{self.print_key}.txt', 'w') as bf_f_cal:
                        bf_f_cal.write('Date'.ljust(14) + 'NingDu'.ljust(14) + 'ShiCheng'.ljust(14) + 'FenKeng'.ljust(16) + '\n')
                        for bf_cal_idx in zip(self.obs_date_mon_cal, NingDu_baseflow_mon_avg, ShiCheng_baseflow_mon_avg, FenKeng_baseflow_mon_avg):
                            format_x = [f'{num:.6f}' for num in bf_cal_idx if isinstance(num, float)]
                            bf_f_cal.write(f'{bf_cal_idx[0]:<10}    {format_x[0]:<10}    {format_x[1]:<10}    {format_x[2]:<10}\n')

                    return ([np.array(NingDu_surface_runoff_mon_avg), np.array(NingDu_baseflow_mon_avg)],
                            [np.array(ShiCheng_surface_runoff_mon_avg), np.array(ShiCheng_baseflow_mon_avg)],
                            [np.array(FenKeng_surface_runoff_mon_avg), np.array(FenKeng_baseflow_mon_avg)])

            else:
                print('Surface Runoff/Baseflow observation data already exist!')
                surface_runoff_data_cal = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_SR_cal_{self.print_key}.txt', skiprows=1, usecols=(1, 2, 3))
                NingDu_sta_surface_runoff_cal   = surface_runoff_data_cal[:, 0]
                ShiCheng_sta_surface_runoff_cal = surface_runoff_data_cal[:, 1]
                FenKeng_sta_surface_runoff_cal  = surface_runoff_data_cal[:, 2]

                baseflow_data_cal = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_BF_cal_{self.print_key}.txt', skiprows=1, usecols=(1, 2, 3))
                NingDu_sta_baseflow_cal   = baseflow_data_cal[:, 0]
                ShiCheng_sta_baseflow_cal = baseflow_data_cal[:, 1]
                FenKeng_sta_baseflow_cal  = baseflow_data_cal[:, 2]

                return ([NingDu_sta_surface_runoff_cal, NingDu_sta_baseflow_cal],
                        [ShiCheng_sta_surface_runoff_cal, ShiCheng_sta_baseflow_cal],
                        [FenKeng_sta_surface_runoff_cal, FenKeng_sta_baseflow_cal])
        elif self.cal_val_state == 'Validation':
            # Validation
            if not (os.path.exists(f'{self.swat_nsga_in}\\observed_SR_val_{self.print_key}.txt') or
                    os.path.exists(f'{self.swat_nsga_in}\\observed_BF_val_{self.print_key}.txt')):
                for hydro_sta_idx in self.hydro_stas.keys():
                    Streamflow_obs = []
                    Bflow_Pass1 = []
                    Bflow_Pass2 = []
                    Bflow_Pass3 = []
                    with open(f'{self.bf_obs_path}\\{hydro_sta_idx}.out', 'r') as base_f:
                        base_sep_data = base_f.readlines()[2:6941]
                        base_sep_data_slice = [line[9:] for line in base_sep_data]
                        for base_line in base_sep_data_slice:
                            base_line_val = base_line.split()
                            Streamflow_obs.append(float(base_line_val[0]))
                            Bflow_Pass1.append(float(base_line_val[1]))
                            Bflow_Pass2.append(float(base_line_val[2]))
                            Bflow_Pass3.append(float(base_line_val[3]))
                    Baseflow_data_val = [(p1 + p2) / 2 for p1, p2 in zip(Bflow_Pass1[4017:], Bflow_Pass2[4017:])]
                    Surface_runoff_data_val = [sf - bf for sf, bf in zip(Streamflow_obs[4017:], Baseflow_data_val)]
                    # Hydrological stations
                    if hydro_sta_idx == 'NingDu':
                        NingDu_surface_runoff_val = Surface_runoff_data_val
                        NingDu_baseflow_val = Baseflow_data_val
                    elif hydro_sta_idx == 'ShiCheng':
                        ShiCheng_surface_runoff_val = Surface_runoff_data_val
                        ShiCheng_baseflow_val = Baseflow_data_val
                    elif hydro_sta_idx == 'FenKeng':
                        FenKeng_surface_runoff_val = Surface_runoff_data_val
                        FenKeng_baseflow_val = Baseflow_data_val

                if self.print_key == 'day':
                    # Write Surface Runoff Observation
                    with open(f'{self.swat_nsga_in}\\observed_SR_val_{self.print_key}.txt', 'w') as sr_f_val:
                        sr_f_val.write('Date'.ljust(14) + 'NingDu'.ljust(14) + 'ShiCheng'.ljust(14) + 'FenKeng'.ljust(16) + '\n')
                        for sr_val_idx in zip(self.obs_date_day_val, NingDu_surface_runoff_val, ShiCheng_surface_runoff_val, FenKeng_surface_runoff_val):
                            format_x = [f'{num:.6f}' for num in sr_val_idx if isinstance(num, float)]
                            sr_f_val.write(f'{sr_val_idx[0]:<10}    {format_x[0]:<10}    {format_x[1]:<10}    {format_x[2]:<10}\n')
                    # Write Baseflow Observation
                    with open(f'{self.swat_nsga_in}\\observed_BF_val_{self.print_key}.txt', 'w') as bf_f_val:
                        bf_f_val.write('Date'.ljust(14) + 'NingDu'.ljust(14) + 'ShiCheng'.ljust(14) + 'FenKeng'.ljust(16) + '\n')
                        for bf_val_idx in zip(self.obs_date_day_val, NingDu_baseflow_val, ShiCheng_baseflow_val, FenKeng_baseflow_val):
                            format_x = [f'{num:.6f}' for num in bf_val_idx if isinstance(num, float)]
                            bf_f_val.write(f'{bf_val_idx[0]:<10}    {format_x[0]:<10}    {format_x[1]:<10}    {format_x[2]:<10}\n')

                    return ([np.array(NingDu_surface_runoff_val), np.array(NingDu_baseflow_val)],
                            [np.array(ShiCheng_surface_runoff_val), np.array(ShiCheng_baseflow_val)],
                            [np.array(FenKeng_surface_runoff_val), np.array(FenKeng_baseflow_val)])
                elif self.print_key == 'month':
                    # Monthly Average
                    obs_date_day_val = pd.date_range(start=f'{self.val_period[0]}-01-01', end=f'{self.val_period[1]}-12-31', freq='D')
                    NingDu_surface_runoff_mon   = pd.DataFrame({'date': obs_date_day_val, 'value': NingDu_surface_runoff_val})
                    ShiCheng_surface_runoff_mon = pd.DataFrame({'date': obs_date_day_val, 'value': ShiCheng_surface_runoff_val})
                    FenKeng_surface_runoff_mon  = pd.DataFrame({'date': obs_date_day_val, 'value': FenKeng_surface_runoff_val})
                    # 按月计算平均值
                    NingDu_surface_runoff_mon_avg   = NingDu_surface_runoff_mon.groupby(pd.Grouper(key='date', freq='M')).mean()['value'].tolist()
                    ShiCheng_surface_runoff_mon_avg = ShiCheng_surface_runoff_mon.groupby(pd.Grouper(key='date', freq='M')).mean()['value'].tolist()
                    FenKeng_surface_runoff_mon_avg  = FenKeng_surface_runoff_mon.groupby(pd.Grouper(key='date', freq='M')).mean()['value'].tolist()

                    NingDu_baseflow_mon   = pd.DataFrame({'date': obs_date_day_val, 'value': NingDu_baseflow_val})
                    ShiCheng_baseflow_mon = pd.DataFrame({'date': obs_date_day_val, 'value': ShiCheng_baseflow_val})
                    FenKeng_baseflow_mon  = pd.DataFrame({'date': obs_date_day_val, 'value': FenKeng_baseflow_val})
                    # 按月计算平均值
                    NingDu_baseflow_mon_avg   = NingDu_baseflow_mon.groupby(pd.Grouper(key='date', freq='M')).mean()['value'].tolist()
                    ShiCheng_baseflow_mon_avg = ShiCheng_baseflow_mon.groupby(pd.Grouper(key='date', freq='M')).mean()['value'].tolist()
                    FenKeng_baseflow_mon_avg  = FenKeng_baseflow_mon.groupby(pd.Grouper(key='date', freq='M')).mean()['value'].tolist()

                    # Write Surface Runoff Observation
                    with open(f'{self.swat_nsga_in}\\observed_SR_val_{self.print_key}.txt', 'w') as sr_f_val:
                        sr_f_val.write('Date'.ljust(14) + 'NingDu'.ljust(14) + 'ShiCheng'.ljust(14) + 'FenKeng'.ljust(16) + '\n')
                        for sr_val_idx in zip(self.obs_date_mon_val, NingDu_surface_runoff_mon_avg, ShiCheng_surface_runoff_mon_avg, FenKeng_surface_runoff_mon_avg):
                            format_x = [f'{num:.6f}' for num in sr_val_idx if isinstance(num, float)]
                            sr_f_val.write(f'{sr_val_idx[0]:<10}    {format_x[0]:<10}    {format_x[1]:<10}    {format_x[2]:<10}\n')
                    # Write Baseflow Observation
                    with open(f'{self.swat_nsga_in}\\observed_BF_val_{self.print_key}.txt', 'w') as bf_f_val:
                        bf_f_val.write('Date'.ljust(14) + 'NingDu'.ljust(14) + 'ShiCheng'.ljust(14) + 'FenKeng'.ljust(16) + '\n')
                        for bf_val_idx in zip(self.obs_date_mon_val, NingDu_baseflow_mon_avg, ShiCheng_baseflow_mon_avg, FenKeng_baseflow_mon_avg):
                            format_x = [f'{num:.6f}' for num in bf_val_idx if isinstance(num, float)]
                            bf_f_val.write(f'{bf_val_idx[0]:<10}    {format_x[0]:<10}    {format_x[1]:<10}    {format_x[2]:<10}\n')

                    return ([np.array(NingDu_surface_runoff_mon_avg), np.array(NingDu_baseflow_mon_avg)],
                            [np.array(ShiCheng_surface_runoff_mon_avg), np.array(ShiCheng_baseflow_mon_avg)],
                            [np.array(FenKeng_surface_runoff_mon_avg), np.array(FenKeng_baseflow_mon_avg)])
            else:
                print('Surface Runoff/Baseflow observation data already exist!')
                surface_runoff_data_val = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_SR_val_{self.print_key}.txt', skiprows=1, usecols=(1, 2, 3))
                NingDu_sta_surface_runoff_val   = surface_runoff_data_val[:, 0]
                ShiCheng_sta_surface_runoff_val = surface_runoff_data_val[:, 1]
                FenKeng_sta_surface_runoff_val  = surface_runoff_data_val[:, 2]

                baseflow_data_val = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_BF_val_{self.print_key}.txt', skiprows=1, usecols=(1, 2, 3))
                NingDu_sta_baseflow_val   = baseflow_data_val[:, 0]
                ShiCheng_sta_baseflow_val = baseflow_data_val[:, 1]
                FenKeng_sta_baseflow_val  = baseflow_data_val[:, 2]

                return ([NingDu_sta_surface_runoff_val, NingDu_sta_baseflow_val],
                        [ShiCheng_sta_surface_runoff_val, ShiCheng_sta_baseflow_val],
                        [FenKeng_sta_surface_runoff_val, FenKeng_sta_baseflow_val])


    # Zonal Statistics
    def zonal_sta_v1(self, zonal_shp, shp_field, in_raster_arr, sta_type='mean'):
        # 读取矢量数据和栅格数据
        vector_data = gpd.read_file(zonal_shp)
        # 按照矢量数据的shp_field字段对栅格进行分区统计
        stats = zonal_stats(vectors=vector_data, raster=in_raster_arr, nodata=-99, stats=[sta_type], categorical=True, copy_properties=True, geojson_out=True)
        # 将统计结果转换为数据框
        df_zonal = gpd.GeoDataFrame.from_features(stats)
        # 将分区统计结果与所使用的矢量字段值进行连接，并按字段排序
        zonal_sta_field = vector_data[[shp_field]].merge(df_zonal[[shp_field, 'mean']], on=shp_field).sort_values(shp_field).to_numpy()
        print('zonal_sta_field:', type(zonal_sta_field), zonal_sta_field)
        return zonal_sta_field


    # Zonal Statistics
    def zonal_sta(self, zonal_shp, shp_field, ras_date, in_raster_arr, ds_affine, sta_type='mean'):
        # 读取矢量数据和栅格数据，并按矢量字段排序
        vector_data = gpd.read_file(zonal_shp)[[shp_field, 'geometry']].sort_values(shp_field)
        # 按照矢量数据的shp_field字段对栅格进行分区统计
        stats = zonal_stats(vectors=vector_data, raster=in_raster_arr, affine=ds_affine, nodata=-99, stats=[sta_type])
        # 将所选择的字段值与统计结果分别作为2列输出为numpy格式
        zonal_sta_field = np.column_stack((vector_data[shp_field], [s['mean'] for s in stats]))
        ras_datetime = datetime.strptime(ras_date, '%Y_%m_%d')
        # ras_datetime = datetime.strptime(ras_date, '%Y_%m_%d').date()
        # print('zonal_sta_field:', zonal_sta_field)
        return [ras_datetime, zonal_sta_field]


    # Read Observed ET Data
    def read_obs_et_data(self):
        print('Read ET:')
        if self.cal_val_state == 'Calibration':
            # Calibration
            if not os.path.exists(f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_ET_cal_{self.print_key}.txt'):
                # Zonal Statistics
                ET_raster_list_cal = []
                ET_raster_list = glob.glob(f'{self.et_obs_path}\\*.tif')
                print('ET_raster_list:', len(ET_raster_list))
                for ET_idx in ET_raster_list:
                    img_year = int(os.path.basename(ET_idx)[:4])
                    if self.cal_period[0] <= img_year <= self.cal_period[1]:
                        ET_raster_list_cal.append(ET_idx)
                ET_raster_list_cal.sort()
                print('ET_raster_list_cal:', len(ET_raster_list_cal))

                # Subbasin/HRU
                vector_shp  = self.HRU_shp
                zonal_field = 'HRU_ID'
                format_len = 8
                if self.spatial_unit == 'HRU':
                    vector_shp = self.HRU_shp
                    zonal_field = 'HRU_ID'
                    format_len = 8
                elif self.spatial_unit == 'Subbasin':
                    vector_shp = self.Subbasin_shp
                    zonal_field = 'Subbasin'
                    format_len = 3
                ds_affine = rasterio.open(ET_raster_list_cal[0]).transform

                ## Multiprocessing
                print('Zonal statistics using multiprocessing:')
                zonal_sta_field_dict = {}
                process_args = [(vector_shp, zonal_field, os.path.basename(ET_cal_idx)[:10], rasterio.open(ET_cal_idx).read(1), ds_affine)
                                for ET_cal_idx in ET_raster_list_cal]
                with Pool(self.cpu_worker_num) as p:
                    zonal_sta_res = p.starmap(self.zonal_sta, process_args)
                for zon_idx in zonal_sta_res:
                    raster_date = zon_idx[0]
                    zon_sta_arr = zon_idx[1]
                    zonal_sta_field_dict[raster_date] = zon_sta_arr
                zonal_sta_field_dict_list = sorted(zonal_sta_field_dict.items(), key=lambda d:d[0], reverse=False)
                print('zonal_sta_field_dict_list:', len(zonal_sta_field_dict_list))

                format_id = [f'{self.spatial_unit}_{int(sp_idx):<{format_len}}' for sp_idx in zonal_sta_field_dict_list[0][1][:, 0]]
                header = f'{"Date":<12}' + ''.join(format_id) + '\n'
                if self.print_key == 'day':
                    with open(f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_ET_cal_{self.print_key}.txt', 'w') as day_f:
                        day_f.write(header)
                        # Values
                        for img_zon_idx in zonal_sta_field_dict_list:
                            date_idx = img_zon_idx[0].date()
                            format_zon_val = [f'{zon_idx:.6f}'.ljust(12) for zon_idx in img_zon_idx[1][:, 1].tolist()]
                            format_zon_val_row = f'{str(date_idx):<12}' + ''.join(format_zon_val)  + '\n'
                            day_f.write(format_zon_val_row)
                elif self.print_key == 'month':
                    zonal_sta_dict_list = []
                    for zon_sta_idx in zonal_sta_field_dict_list:
                        zonal_date = zon_sta_idx[0]
                        zonal_val = zon_sta_idx[1][:, 1].T.tolist()
                        zonal_val.insert(0, zonal_date)
                        zonal_sta_dict_list.append(zonal_val)
                    # print('zonal_sta_dict_list:', len(zonal_sta_dict_list))
                    zonal_sta_dict_df = pd.DataFrame(zonal_sta_dict_list).sort_values(by=0)
                    zonal_sta_dict_df[0] = pd.to_datetime(zonal_sta_dict_df[0])
                    zonal_sta_dict_df.set_index(0, inplace=True)
                    # Monthly Sum
                    zonal_sta_dict_df_mon_sum = zonal_sta_dict_df.resample('M').sum()
                    # 将index列的日期格式设置为只保留年月
                    zonal_sta_dict_df_mon_sum.index = zonal_sta_dict_df_mon_sum.index.strftime('%Y-%m')
                    # print('zonal_sta_dict_df_mon_sum:', zonal_sta_dict_df_mon_sum)

                    # 将DataFrame转换为字符串，并设置列的最小空间宽度
                    format_zonal_sta_mon_sum_df = zonal_sta_dict_df_mon_sum.to_string(index=True, col_space=11, header=False,
                                                                                      float_format='%.6f', index_names=False, justify='left')
                    # 将数据写入到文本文件中
                    with open(f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_ET_cal_{self.print_key}.txt', 'w') as mon_f:
                        mon_f.write(header)
                        mon_f.write(format_zonal_sta_mon_sum_df + '\n')
            else:
                print('ET observation data already exist!')
                et_data_cal_dict = {}
                if self.spatial_unit == 'HRU':
                    et_data_cal = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_ET_cal_{self.print_key}.txt', skiprows=1,
                                             usecols=self.HRU_ID)
                    for hru_idx, hru_val in enumerate(self.HRU_ID):
                        # print(hru_idx, hru_val)
                        et_data_cal_dict[hru_val] = et_data_cal[:, hru_idx]
                elif self.spatial_unit == 'Subbasin':
                    et_data_cal = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_ET_cal_{self.print_key}.txt', skiprows=1,
                                             usecols=self.Subbasin_ID)
                    for sub_idx, sub_val in enumerate(self.Subbasin_ID):
                        # print(sub_idx, sub_val)
                        et_data_cal_dict[sub_val] = et_data_cal[:, sub_idx]
                # print('et_data_cal_dict:', len(et_data_cal_dict))
                return et_data_cal_dict
        elif self.cal_val_state == 'Validation':
            # Validation
            if not os.path.exists(f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_ET_val_{self.print_key}.txt'):
                # Zonal Statistics
                ET_raster_list_val = []
                ET_raster_list = glob.glob(f'{self.et_obs_path}\\*.tif')
                print('ET_raster_list:', len(ET_raster_list))
                for ET_idx in ET_raster_list:
                    img_year = int(os.path.basename(ET_idx)[:4])
                    if self.val_period[0] <= img_year <= self.val_period[1]:
                        ET_raster_list_val.append(ET_idx)
                ET_raster_list_val.sort()
                print('ET_raster_list_val:', len(ET_raster_list_val))

                # Subbasin/HRU
                vector_shp  = self.HRU_shp
                zonal_field = 'HRU_ID'
                format_len = 8
                if self.spatial_unit == 'HRU':
                    vector_shp = self.HRU_shp
                    zonal_field = 'HRU_ID'
                    format_len = 8
                elif self.spatial_unit == 'Subbasin':
                    vector_shp = self.Subbasin_shp
                    zonal_field = 'Subbasin'
                    format_len = 3
                ds_affine = rasterio.open(ET_raster_list_val[0]).transform

                ## Multiprocessing
                print('Zonal statistics using multiprocessing:')
                zonal_sta_field_dict = {}
                process_args = [(vector_shp, zonal_field, os.path.basename(ET_val_idx)[:10], rasterio.open(ET_val_idx).read(1), ds_affine)
                                for ET_val_idx in ET_raster_list_val]
                with Pool(self.cpu_worker_num) as p:
                    zonal_sta_res = p.starmap(self.zonal_sta, process_args)
                for zon_idx in zonal_sta_res:
                    raster_date = zon_idx[0]
                    zon_sta_arr = zon_idx[1]
                    zonal_sta_field_dict[raster_date] = zon_sta_arr
                zonal_sta_field_dict_list = sorted(zonal_sta_field_dict.items(), key=lambda d:d[0], reverse=False)
                print('zonal_sta_field_dict_list:', len(zonal_sta_field_dict_list))

                format_id = [f'{self.spatial_unit}_{int(sp_idx):<{format_len}}' for sp_idx in zonal_sta_field_dict_list[0][1][:, 0]]
                header = f'{"Date":<12}' + ''.join(format_id) + '\n'
                if self.print_key == 'day':
                    with open(f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_ET_val_{self.print_key}.txt', 'w') as day_f:
                        day_f.write(header)
                        # Values
                        for img_zon_idx in zonal_sta_field_dict_list:
                            date_idx = img_zon_idx[0].date()
                            format_zon_val = [f'{zon_idx:.6f}'.ljust(12) for zon_idx in img_zon_idx[1][:, 1].tolist()]
                            format_zon_val_row = f'{str(date_idx):<12}' + ''.join(format_zon_val)  + '\n'
                            day_f.write(format_zon_val_row)
                elif self.print_key == 'month':
                    zonal_sta_dict_list = []
                    for zon_sta_idx in zonal_sta_field_dict_list:
                        zonal_date = zon_sta_idx[0]
                        zonal_val = zon_sta_idx[1][:, 1].T.tolist()
                        zonal_val.insert(0, zonal_date)
                        zonal_sta_dict_list.append(zonal_val)
                    # print('zonal_sta_dict_list:', len(zonal_sta_dict_list))
                    zonal_sta_dict_df = pd.DataFrame(zonal_sta_dict_list).sort_values(by=0)
                    zonal_sta_dict_df[0] = pd.to_datetime(zonal_sta_dict_df[0])
                    zonal_sta_dict_df.set_index(0, inplace=True)
                    # Monthly Sum
                    zonal_sta_dict_df_mon_sum = zonal_sta_dict_df.resample('M').sum()
                    # 将index列的日期格式设置为只保留年月
                    zonal_sta_dict_df_mon_sum.index = zonal_sta_dict_df_mon_sum.index.strftime('%Y-%m')
                    # print('zonal_sta_dict_df_mon_sum:', zonal_sta_dict_df_mon_sum)

                    # 将DataFrame转换为字符串，并设置列的最小空间宽度
                    format_zonal_sta_mon_sum_df = zonal_sta_dict_df_mon_sum.to_string(index=True, col_space=11, header=False,
                                                                                      float_format='%.6f', index_names=False, justify='left')
                    # 将数据写入到文本文件中
                    with open(f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_ET_val_{self.print_key}.txt', 'w') as mon_f:
                        mon_f.write(header)
                        mon_f.write(format_zonal_sta_mon_sum_df + '\n')
            else:
                print('ET observation data already exist!')
                et_data_val_dict = {}
                if self.spatial_unit == 'HRU':
                    et_data_val = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_ET_val_{self.print_key}.txt', skiprows=1,
                                             usecols=self.HRU_ID)
                    for hru_idx, hru_val in enumerate(self.HRU_ID):
                        # print(hru_idx, hru_val)
                        et_data_val_dict[hru_val] = et_data_val[:, hru_idx]
                elif self.spatial_unit == 'Subbasin':
                    et_data_val = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_ET_val_{self.print_key}.txt', skiprows=1,
                                             usecols=self.Subbasin_ID)
                    for sub_idx, sub_val in enumerate(self.Subbasin_ID):
                        # print(sub_idx, sub_val)
                        et_data_val_dict[sub_val] = et_data_val[:, sub_idx]
                print('et_data_val_dict:', len(et_data_val_dict))
                return et_data_val_dict


    # Read Observed LAI Data
    def read_obs_lai_data(self):
        print('Read LAI:')
        if self.cal_val_state == 'Calibration':
            # Calibration
            if not os.path.exists(f'{self.swat_nsga_in}\\observed_HRU_LAI_cal_{self.print_key}.txt'):
                # Zonal Statistics
                LAI_raster_list_cal = []
                LAI_raster_list = glob.glob(f'{self.lai_obs_path}\\*.tif')
                print('LAI_raster_list:', len(LAI_raster_list))
                for LAI_idx in LAI_raster_list:
                    img_year = int(os.path.basename(LAI_idx)[:4])
                    if self.cal_period[0] <= img_year <= self.cal_period[1]:
                        LAI_raster_list_cal.append(LAI_idx)
                LAI_raster_list_cal.sort()
                print('LAI_raster_list_cal:', len(LAI_raster_list_cal))

                # Subbasin/HRU
                vector_shp  = self.HRU_shp
                zonal_field = 'HRU_ID'
                format_len = 8
                ds_affine = rasterio.open(LAI_raster_list_cal[0]).transform

                ## Multiprocessing
                print('Zonal statistics using multiprocessing:')
                zonal_sta_field_dict = {}
                process_args = [(vector_shp, zonal_field, os.path.basename(LAI_cal_idx)[:10], rasterio.open(LAI_cal_idx).read(1), ds_affine)
                                for LAI_cal_idx in LAI_raster_list_cal]
                with Pool(self.cpu_worker_num) as p:
                    zonal_sta_res = p.starmap(self.zonal_sta, process_args)
                for zon_idx in zonal_sta_res:
                    raster_date = zon_idx[0]
                    zon_sta_arr = zon_idx[1]
                    zonal_sta_field_dict[raster_date] = zon_sta_arr
                zonal_sta_field_dict_list = sorted(zonal_sta_field_dict.items(), key=lambda d:d[0], reverse=False)
                print('zonal_sta_field_dict_list:', len(zonal_sta_field_dict_list))

                format_id = [f'HRU_{int(sp_idx):<{format_len}}' for sp_idx in zonal_sta_field_dict_list[0][1][:, 0]]
                header = f'{"Date":<12}' + ''.join(format_id) + '\n'
                if self.print_key == 'day':
                    with open(f'{self.swat_nsga_in}\\observed_HRU_LAI_cal_{self.print_key}.txt', 'w') as day_f:
                        day_f.write(header)
                        # Values
                        for img_zon_idx in zonal_sta_field_dict_list:
                            date_idx = img_zon_idx[0].date()
                            format_zon_val = [f'{zon_idx:.6f}'.ljust(12) for zon_idx in img_zon_idx[1][:, 1].tolist()]
                            format_zon_val_row = f'{str(date_idx):<12}' + ''.join(format_zon_val)  + '\n'
                            day_f.write(format_zon_val_row)
                elif self.print_key == 'month':
                    zonal_sta_dict_list = []
                    for zon_sta_idx in zonal_sta_field_dict_list:
                        zonal_date = zon_sta_idx[0]
                        zonal_val = zon_sta_idx[1][:, 1].T.tolist()
                        zonal_val.insert(0, zonal_date)
                        zonal_sta_dict_list.append(zonal_val)
                    # print('zonal_sta_dict_list:', len(zonal_sta_dict_list))
                    zonal_sta_dict_df = pd.DataFrame(zonal_sta_dict_list).sort_values(by=0)
                    zonal_sta_dict_df[0] = pd.to_datetime(zonal_sta_dict_df[0])
                    zonal_sta_dict_df.set_index(0, inplace=True)
                    # Monthly Average
                    zonal_sta_dict_df_mon_avg = zonal_sta_dict_df.resample('M').mean()
                    # 将index列的日期格式设置为只保留年月
                    zonal_sta_dict_df_mon_avg.index = zonal_sta_dict_df_mon_avg.index.strftime('%Y-%m')
                    # print('zonal_sta_dict_df_mon_avg:', zonal_sta_dict_df_mon_avg)

                    # 将DataFrame转换为字符串，并设置列的最小空间宽度
                    format_zonal_sta_mon_avg_df = zonal_sta_dict_df_mon_avg.to_string(index=True, col_space=11, header=False,
                                                                                      float_format='%.6f', index_names=False, justify='left')
                    # 将数据写入到文本文件中
                    with open(f'{self.swat_nsga_in}\\observed_HRU_LAI_cal_{self.print_key}.txt', 'w') as mon_f:
                        mon_f.write(header)
                        mon_f.write(format_zonal_sta_mon_avg_df + '\n')
            else:
                print('LAI observation data already exist!')
                lai_data_cal_dict = {}
                lai_data_cal = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_HRU_LAI_cal_{self.print_key}.txt', skiprows=1, usecols=self.HRU_ID)
                for hru_idx, hru_val in enumerate(self.HRU_ID):
                    # print(hru_idx, hru_val)
                    lai_data_cal_dict[hru_val] = lai_data_cal[:, hru_idx]
                # print('lai_data_cal_dict:', len(lai_data_cal_dict))
                return lai_data_cal_dict
        elif self.cal_val_state == 'Validation':
            # Validation
            if not os.path.exists(f'{self.swat_nsga_in}\\observed_HRU_LAI_val_{self.print_key}.txt'):
                # Zonal Statistics
                LAI_raster_list_val = []
                LAI_raster_list = glob.glob(f'{self.lai_obs_path}\\*.tif')
                print('LAI_raster_list:', len(LAI_raster_list))
                for LAI_idx in LAI_raster_list:
                    img_year = int(os.path.basename(LAI_idx)[:4])
                    if self.val_period[0] <= img_year <= self.val_period[1]:
                        LAI_raster_list_val.append(LAI_idx)
                LAI_raster_list_val.sort()
                print('LAI_raster_list_val:', len(LAI_raster_list_val))

                # Subbasin/HRU
                vector_shp  = self.HRU_shp
                zonal_field = 'HRU_ID'
                format_len = 8
                ds_affine = rasterio.open(LAI_raster_list_val[0]).transform

                ## Multiprocessing
                print('Zonal statistics using multiprocessing:')
                zonal_sta_field_dict = {}
                process_args = [(vector_shp, zonal_field, os.path.basename(LAI_val_idx)[:10], rasterio.open(LAI_val_idx).read(1), ds_affine)
                                for LAI_val_idx in LAI_raster_list_val]
                with Pool(self.cpu_worker_num) as p:
                    zonal_sta_res = p.starmap(self.zonal_sta, process_args)
                for zon_idx in zonal_sta_res:
                    raster_date = zon_idx[0]
                    zon_sta_arr = zon_idx[1]
                    zonal_sta_field_dict[raster_date] = zon_sta_arr
                zonal_sta_field_dict_list = sorted(zonal_sta_field_dict.items(), key=lambda d:d[0], reverse=False)
                print('zonal_sta_field_dict_list:', len(zonal_sta_field_dict_list))

                format_id = [f'HRU_{int(sp_idx):<{format_len}}' for sp_idx in zonal_sta_field_dict_list[0][1][:, 0]]
                header = f'{"Date":<12}' + ''.join(format_id) + '\n'
                if self.print_key == 'day':
                    with open(f'{self.swat_nsga_in}\\observed_HRU_LAI_val_{self.print_key}.txt', 'w') as day_f:
                        day_f.write(header)
                        # Values
                        for img_zon_idx in zonal_sta_field_dict_list:
                            date_idx = img_zon_idx[0].date()
                            format_zon_val = [f'{zon_idx:.6f}'.ljust(12) for zon_idx in img_zon_idx[1][:, 1].tolist()]
                            format_zon_val_row = f'{str(date_idx):<12}' + ''.join(format_zon_val)  + '\n'
                            day_f.write(format_zon_val_row)
                elif self.print_key == 'month':
                    zonal_sta_dict_list = []
                    for zon_sta_idx in zonal_sta_field_dict_list:
                        zonal_date = zon_sta_idx[0]
                        zonal_val = zon_sta_idx[1][:, 1].T.tolist()
                        zonal_val.insert(0, zonal_date)
                        zonal_sta_dict_list.append(zonal_val)
                    # print('zonal_sta_dict_list:', len(zonal_sta_dict_list))
                    zonal_sta_dict_df = pd.DataFrame(zonal_sta_dict_list).sort_values(by=0)
                    zonal_sta_dict_df[0] = pd.to_datetime(zonal_sta_dict_df[0])
                    zonal_sta_dict_df.set_index(0, inplace=True)
                    # Monthly Average
                    zonal_sta_dict_df_mon_avg = zonal_sta_dict_df.resample('M').mean()
                    # 将index列的日期格式设置为只保留年月
                    zonal_sta_dict_df_mon_avg.index = zonal_sta_dict_df_mon_avg.index.strftime('%Y-%m')
                    # print('zonal_sta_dict_df_mon_sum:', zonal_sta_dict_df_mon_sum)

                    # 将DataFrame转换为字符串，并设置列的最小空间宽度
                    format_zonal_sta_mon_avg_df = zonal_sta_dict_df_mon_avg.to_string(index=True, col_space=11, header=False,
                                                                                      float_format='%.6f', index_names=False, justify='left')
                    # 将数据写入到文本文件中
                    with open(f'{self.swat_nsga_in}\\observed_HRU_LAI_val_{self.print_key}.txt', 'w') as mon_f:
                        mon_f.write(header)
                        mon_f.write(format_zonal_sta_mon_avg_df + '\n')
            else:
                print('LAI observation data already exist!')
                lai_data_val_dict = {}
                lai_data_val = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_HRU_LAI_val_{self.print_key}.txt', skiprows=1, usecols=self.HRU_ID)
                for hru_idx, hru_val in enumerate(self.HRU_ID):
                    # print(hru_idx, hru_val)
                    lai_data_val_dict[hru_val] = lai_data_val[:, hru_idx]
                print('lai_data_val_dict:', len(lai_data_val_dict))
                return lai_data_val_dict


    # Read Observed Biomass Data
    def read_obs_biom_data(self):
        print('Read Biomass:')
        if self.cal_val_state == 'Calibration':
            # Calibration
            if not os.path.exists(f'{self.swat_nsga_in}\\observed_HRU_BIOM_cal_year.txt'):
                print('Biomass observation data do not exist!')
            else:
                print('Biomass observation data already exist!')
                bio_data_cal_dict = {}
                HRU_list = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_HRU_BIOM_cal_year.txt', dtype=str, max_rows=1).tolist()[1:]
                HRU_list = [int(hur_idx[4:]) for hur_idx in HRU_list]
                # print('HRU_list:', len(HRU_list))
                bio_data_cal = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_HRU_BIOM_cal_year.txt', skiprows=1, usecols=range(1, len(HRU_list) + 1))
                for hru_idx, hru_val in enumerate(HRU_list):
                    # print(hru_idx, hru_val)
                    bio_data_cal_dict[hru_val] = bio_data_cal[:, hru_idx]
                print('bio_data_cal_dict:', len(bio_data_cal_dict))
                return bio_data_cal_dict
        elif self.cal_val_state == 'Validation':
            # Validation
            if not os.path.exists(f'{self.swat_nsga_in}\\observed_HRU_BIOM_val_year.txt'):
                print('Biomass observation data do not exist!')
            else:
                print('Biomass observation data already exist!')
                bio_data_val_dict = {}
                HRU_list = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_HRU_BIOM_val_year.txt', dtype=str, max_rows=1).tolist()[1:]
                HRU_list = [int(hur_idx[4:]) for hur_idx in HRU_list]
                print('HRU_list:', len(HRU_list))
                bio_data_val = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_HRU_BIOM_val_year.txt', skiprows=1, usecols=range(1, len(HRU_list) + 1))
                for hru_idx, hru_val in enumerate(HRU_list):
                    # print(hru_idx, hru_val)
                    bio_data_val_dict[hru_val] = bio_data_val[:, hru_idx]
                print('bio_data_val_dict:', len(bio_data_val_dict))
                return bio_data_val_dict


    # Read Observed Root Zone Soil Moisture
    def read_obs_rzsw_data(self):
        print('Read RZSW:')
        if self.cal_val_state == 'Calibration':
            # Calibration
            if not os.path.exists(f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_RZSW_cal_{self.print_key}.txt'):
                # Zonal Statistics
                RZSW_raster_list_cal = []
                RZSW_raster_list = glob.glob(f'{self.rzsw_obs_path}\\*.tif')
                print('RZSW_raster_list:', len(RZSW_raster_list))
                for RZSW_idx in RZSW_raster_list:
                    img_year = int(os.path.basename(RZSW_idx)[:4])
                    if self.cal_period[0] <= img_year <= self.cal_period[1]:
                        RZSW_raster_list_cal.append(RZSW_idx)
                RZSW_raster_list_cal.sort()
                print('RZSW_raster_list_cal:', len(RZSW_raster_list_cal))

                # Subbasin/HRU
                vector_shp  = self.HRU_shp
                zonal_field = 'HRU_ID'
                format_len = 8
                if self.spatial_unit == 'HRU':
                    vector_shp = self.HRU_shp
                    zonal_field = 'HRU_ID'
                    format_len = 8
                elif self.spatial_unit == 'Subbasin':
                    vector_shp = self.Subbasin_shp
                    zonal_field = 'Subbasin'
                    format_len = 3
                ds_affine = rasterio.open(RZSW_raster_list_cal[0]).transform

                ## Multiprocessing
                print('Zonal statistics using multiprocessing:')
                zonal_sta_field_dict = {}
                process_args = [(vector_shp, zonal_field, os.path.basename(RZSW_cal_idx)[:10], rasterio.open(RZSW_cal_idx).read(1), ds_affine)
                                for RZSW_cal_idx in RZSW_raster_list_cal]
                with Pool(self.cpu_worker_num) as p:
                    zonal_sta_res = p.starmap(self.zonal_sta, process_args)
                for zon_idx in zonal_sta_res:
                    raster_date = zon_idx[0]
                    zon_sta_arr = zon_idx[1]
                    zonal_sta_field_dict[raster_date] = zon_sta_arr
                zonal_sta_field_dict_list = sorted(zonal_sta_field_dict.items(), key=lambda d:d[0], reverse=False)
                print('zonal_sta_field_dict_list:', len(zonal_sta_field_dict_list))

                format_id = [f'{self.spatial_unit}_{int(sp_idx):<{format_len}}' for sp_idx in zonal_sta_field_dict_list[0][1][:, 0]]
                header = f'{"Date":<12}' + ''.join(format_id) + '\n'
                if self.print_key == 'day':
                    with open(f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_RZSW_cal_{self.print_key}.txt', 'w') as day_f:
                        day_f.write(header)
                        # Values
                        for img_zon_idx in zonal_sta_field_dict_list:
                            date_idx = img_zon_idx[0].date()
                            format_zon_val = [f'{zon_idx:.6f}'.ljust(12) for zon_idx in img_zon_idx[1][:, 1].tolist()]
                            format_zon_val_row = f'{str(date_idx):<12}' + ''.join(format_zon_val)  + '\n'
                            day_f.write(format_zon_val_row)
                elif self.print_key == 'month':
                    zonal_sta_dict_list = []
                    for zon_sta_idx in zonal_sta_field_dict_list:
                        zonal_date = zon_sta_idx[0]
                        zonal_val = zon_sta_idx[1][:, 1].T.tolist()
                        zonal_val.insert(0, zonal_date)
                        zonal_sta_dict_list.append(zonal_val)
                    # print('zonal_sta_dict_list:', len(zonal_sta_dict_list))
                    zonal_sta_dict_df = pd.DataFrame(zonal_sta_dict_list).sort_values(by=0)
                    zonal_sta_dict_df[0] = pd.to_datetime(zonal_sta_dict_df[0])
                    zonal_sta_dict_df.set_index(0, inplace=True)
                    # Monthly Sum
                    zonal_sta_dict_df_mon_sum = zonal_sta_dict_df.resample('M').sum()
                    # 将index列的日期格式设置为只保留年月
                    zonal_sta_dict_df_mon_sum.index = zonal_sta_dict_df_mon_sum.index.strftime('%Y-%m')
                    # print('zonal_sta_dict_df_mon_sum:', zonal_sta_dict_df_mon_sum)

                    # 将DataFrame转换为字符串，并设置列的最小空间宽度
                    format_zonal_sta_mon_sum_df = zonal_sta_dict_df_mon_sum.to_string(index=True, col_space=11, header=False,
                                                                                      float_format='%.6f', index_names=False, justify='left')
                    # 将数据写入到文本文件中
                    with open(f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_RZSW_cal_{self.print_key}.txt', 'w') as mon_f:
                        mon_f.write(header)
                        mon_f.write(format_zonal_sta_mon_sum_df + '\n')
            else:
                print('RZSW observation data already exist!')
                rzsw_data_cal_dict = {}
                if self.spatial_unit == 'HRU':
                    rzsw_data_cal = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_RZSW_cal_{self.print_key}.txt', skiprows=1,
                                               usecols=self.HRU_ID)
                    for hru_idx, hru_val in enumerate(self.HRU_ID):
                        # print(hru_idx, hru_val)
                        rzsw_data_cal_dict[hru_val] = rzsw_data_cal[:, hru_idx]
                elif self.spatial_unit == 'Subbasin':
                    rzsw_data_cal = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_RZSW_cal_{self.print_key}.txt', skiprows=1,
                                               usecols=self.Subbasin_ID)
                    for sub_idx, sub_val in enumerate(self.Subbasin_ID):
                        # print(sub_idx, sub_val)
                        rzsw_data_cal_dict[sub_val] = rzsw_data_cal[:, sub_idx]
                print('rzsw_data_cal_dict:', len(rzsw_data_cal_dict))
                return rzsw_data_cal_dict
        elif self.cal_val_state == 'Validation':
            # Validation
            if not os.path.exists(f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_RZSW_val_{self.print_key}.txt'):
                # Zonal Statistics
                RZSW_raster_list_val = []
                RZSW_raster_list = glob.glob(f'{self.et_obs_path}\\*.tif')
                print('RZSW_raster_list:', len(RZSW_raster_list))
                for RZSW_idx in RZSW_raster_list:
                    img_year = int(os.path.basename(RZSW_idx)[:4])
                    if self.val_period[0] <= img_year <= self.val_period[1]:
                        RZSW_raster_list_val.append(RZSW_idx)
                RZSW_raster_list_val.sort()
                print('RZSW_raster_list_val:', len(RZSW_raster_list_val))

                # Subbasin/HRU
                vector_shp  = self.HRU_shp
                zonal_field = 'HRU_ID'
                format_len = 8
                if self.spatial_unit == 'HRU':
                    vector_shp = self.HRU_shp
                    zonal_field = 'HRU_ID'
                    format_len = 8
                elif self.spatial_unit == 'Subbasin':
                    vector_shp = self.Subbasin_shp
                    zonal_field = 'Subbasin'
                    format_len = 3
                ds_affine = rasterio.open(RZSW_raster_list_val[0]).transform

                ## Multiprocessing
                print('Zonal statistics using multiprocessing:')
                zonal_sta_field_dict = {}
                process_args = [(vector_shp, zonal_field, os.path.basename(RZSW_val_idx)[:10], rasterio.open(RZSW_val_idx).read(1), ds_affine)
                                for RZSW_val_idx in RZSW_raster_list_val]
                with Pool(self.cpu_worker_num) as p:
                    zonal_sta_res = p.starmap(self.zonal_sta, process_args)
                for zon_idx in zonal_sta_res:
                    raster_date = zon_idx[0]
                    zon_sta_arr = zon_idx[1]
                    zonal_sta_field_dict[raster_date] = zon_sta_arr
                zonal_sta_field_dict_list = sorted(zonal_sta_field_dict.items(), key=lambda d:d[0], reverse=False)
                print('zonal_sta_field_dict_list:', len(zonal_sta_field_dict_list))

                format_id = [f'{self.spatial_unit}_{int(sp_idx):<{format_len}}'
                             for sp_idx in zonal_sta_field_dict_list[0][1][:, 0]]
                header = f'{"Date":<12}' + ''.join(format_id) + '\n'
                if self.print_key == 'day':
                    with open(f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_RZSW_val_{self.print_key}.txt', 'w') as day_f:
                        day_f.write(header)
                        # Values
                        for img_zon_idx in zonal_sta_field_dict_list:
                            date_idx = img_zon_idx[0].date()
                            format_zon_val = [f'{zon_idx:.6f}'.ljust(12) for zon_idx in img_zon_idx[1][:, 1].tolist()]
                            format_zon_val_row = f'{str(date_idx):<12}' + ''.join(format_zon_val)  + '\n'
                            day_f.write(format_zon_val_row)
                elif self.print_key == 'month':
                    zonal_sta_dict_list = []
                    for zon_sta_idx in zonal_sta_field_dict_list:
                        zonal_date = zon_sta_idx[0]
                        zonal_val = zon_sta_idx[1][:, 1].T.tolist()
                        zonal_val.insert(0, zonal_date)
                        zonal_sta_dict_list.append(zonal_val)
                    # print('zonal_sta_dict_list:', len(zonal_sta_dict_list))
                    zonal_sta_dict_df = pd.DataFrame(zonal_sta_dict_list).sort_values(by=0)
                    zonal_sta_dict_df[0] = pd.to_datetime(zonal_sta_dict_df[0])
                    zonal_sta_dict_df.set_index(0, inplace=True)
                    # Monthly Sum
                    zonal_sta_dict_df_mon_sum = zonal_sta_dict_df.resample('M').sum()
                    # 将index列的日期格式设置为只保留年月
                    zonal_sta_dict_df_mon_sum.index = zonal_sta_dict_df_mon_sum.index.strftime('%Y-%m')
                    # print('zonal_sta_dict_df_mon_sum:', zonal_sta_dict_df_mon_sum)

                    # 将DataFrame转换为字符串，并设置列的最小空间宽度
                    format_zonal_sta_mon_sum_df = zonal_sta_dict_df_mon_sum.to_string(index=True, col_space=11, header=False,
                                                                                      float_format='%.6f', index_names=False, justify='left')
                    # 将数据写入到文本文件中
                    with open(f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_RZSW_val_{self.print_key}.txt', 'w') as mon_f:
                        mon_f.write(header)
                        mon_f.write(format_zonal_sta_mon_sum_df + '\n')
            else:
                print('RZSW observation data already exist!')
                rzsw_data_val_dict = {}
                if self.spatial_unit == 'HRU':
                    rzsw_data_val = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_RZSW_val_{self.print_key}.txt', skiprows=1,
                                               usecols=self.HRU_ID)
                    for hru_idx, hru_val in enumerate(self.HRU_ID):
                        # print(hru_idx, hru_val)
                        rzsw_data_val_dict[hru_val] = rzsw_data_val[:, hru_idx]
                elif self.spatial_unit == 'Subbasin':
                    rzsw_data_val = np.loadtxt(fname=f'{self.swat_nsga_in}\\observed_{self.spatial_unit}_RZSW_val_{self.print_key}.txt', skiprows=1,
                                               usecols=self.Subbasin_ID)
                    for sub_idx, sub_val in enumerate(self.Subbasin_ID):
                        # print(sub_idx, sub_val)
                        rzsw_data_val_dict[sub_val] = rzsw_data_val[:, sub_idx]
                print('rzsw_data_val_dict:', len(rzsw_data_val_dict))
                return rzsw_data_val_dict


    def NSE(self, obs_data, sim_data):
        if obs_data.shape == sim_data.shape:
            return 1 - (np.sum(np.square(obs_data - sim_data)) / np.sum(np.square(obs_data - np.mean(obs_data))))
        else:
            print('Inconsistent shape between observation and simulation data!')
            sys.exit()


    def KGE(self, obs_data, sim_data):
        if obs_data.shape == sim_data.shape:
            # slope, intercept, rvalue, pvalue, stderr, intercept_stderr
            r     = stats.linregress(sim_data, obs_data)[2]
            alpha = np.std(sim_data) / np.std(obs_data)
            beta  = np.mean(sim_data) / np.mean(obs_data)
            return 1 - np.sqrt(np.square(r - 1) + np.square(alpha - 1) + np.square(beta - 1))
        else:
            print('Inconsistent shape between observation and simulation data!')
            sys.exit()


    def R2(self, obs_data, sim_data):
        if obs_data.shape == sim_data.shape:
            return (np.square(np.sum((obs_data - np.mean(obs_data)) * (sim_data - np.mean(sim_data)))) /
                   (np.sum(np.square(obs_data - np.mean(obs_data))) * np.sum(np.square(sim_data - np.mean(sim_data)))))
        else:
            print('Inconsistent shape between observation and simulation data!')
            sys.exit()


    def PBIAS(self, obs_data, sim_data):
        if obs_data.shape == sim_data.shape:
            return 100 * np.sum(np.abs(obs_data - sim_data)) / np.sum(obs_data)
        else:
            print('Inconsistent shape between observation and simulation data!')
            sys.exit()


    def RMSE(self, obs_data, sim_data):
        if obs_data.shape == sim_data.shape:
            return np.sqrt(np.sum(np.square(sim_data - obs_data)) / obs_data.shape[0])
        else:
            print('Inconsistent shape between observation and simulation data!')
            sys.exit()


    def RMSE_Moving_Window(self, obs_data, sim_data, half_winsize):
        if obs_data.shape == sim_data.shape:
            return np.sqrt(np.sum(np.square(sim_data - obs_data)) / (2 * half_winsize + 1))
        else:
            print('Inconsistent shape between observation and simulation data!')
            sys.exit()


    def Veg_sim_area_weighted_avg(self, Veg_pop_idx, HRU_ID, HRU_area_dict):
        pop_idx = Veg_pop_idx[0]
        # print('pop_idx:', pop_idx)
        Veg_sim_data_dict = {k: v for k, v in filter(lambda x: x[0] in HRU_ID, Veg_pop_idx[1].items())}
        # print('Veg_sim_data_dict:', len(Veg_sim_data_dict))
        Veg_sim_data_dict_area_w = np.zeros(shape=len(list(Veg_sim_data_dict.values())[0]))
        for Veg_HRU, Veg_val in Veg_sim_data_dict.items():
            Veg_sim_data_dict_area_w += Veg_val * (HRU_area_dict[Veg_HRU] / sum(list(HRU_area_dict.values())))
        return (pop_idx, Veg_sim_data_dict_area_w)


    def RS_sim_area_weighted_avg(self, RS_pop_idx):
        pop_idx = RS_pop_idx[0]
        # print('pop_idx:', pop_idx)
        RS_sim_data_dict = RS_pop_idx[1]
        # print('RS_sim_data_dict:', len(RS_sim_data_dict))
        RS_sim_data_dict_area_w = np.zeros(shape=len(list(RS_sim_data_dict.values())[0]))
        for RS_Sp_ID, RS_val in RS_sim_data_dict.items():
            RS_sim_data_dict_area_w += RS_val * (self.Area_dict[RS_Sp_ID] / sum(list(self.Area_dict.values())))
        return (pop_idx, RS_sim_data_dict_area_w)


    def LAI_ET_sim_area_weighted_avg(self, RS_pop_idx, RS_ID):
        pop_idx = RS_pop_idx[0]
        # print('pop_idx:', pop_idx)
        RS_sim_data_dict = RS_pop_idx[RS_ID]
        # print('RS_sim_data_dict:', len(RS_sim_data_dict))
        RS_sim_data_dict_area_w = np.zeros(shape=len(list(RS_sim_data_dict.values())[0]))
        for RS_Sp_ID, RS_val in RS_sim_data_dict.items():
            RS_sim_data_dict_area_w += RS_val * (self.Area_dict[RS_Sp_ID] / sum(list(self.Area_dict.values())))
        return (pop_idx, RS_sim_data_dict_area_w)


    def RS_LAI_ET_sim_area_weighted_avg(self, RS_pop_idx):
        pop_idx = RS_pop_idx[0]
        # print('pop_idx:', pop_idx)
        RS_sim_data_dict = {k: v for k, v in filter(lambda x: x[0] in self.HRU_ID_Veg, RS_pop_idx[2].items())}
        # print('RS_sim_data_dict:', len(RS_sim_data_dict))
        RS_sim_data_dict_area_w = np.zeros(shape=len(list(RS_sim_data_dict.values())[0]))
        for RS_Sp_ID, RS_val in RS_sim_data_dict.items():
            RS_sim_data_dict_area_w += RS_val * (self.HRU_veg_area_dict[RS_Sp_ID] / sum(list(self.HRU_veg_area_dict.values())))
        return (pop_idx, RS_sim_data_dict_area_w)


    def Water_budget_sim_area_weighted_avg(self, WB_pop_idx, WB_id, HRU_ID, HRU_area_dict):
        pop_idx = WB_pop_idx[0]
        # print('pop_idx:', pop_idx)
        WB_sim_data_dict = {k: v for k, v in filter(lambda x: x[0] in HRU_ID, WB_pop_idx[WB_id].items())}
        # print('WB_sim_data_dict:', len(WB_sim_data_dict))
        WB_sim_data_dict_area_w = np.zeros(shape=len(list(WB_sim_data_dict.values())[0]))
        for WB_HRU, WB_val in WB_sim_data_dict.items():
            WB_sim_data_dict_area_w += WB_val * (HRU_area_dict[WB_HRU] / sum(list(HRU_area_dict.values())))
        return (pop_idx, WB_sim_data_dict_area_w)


    def mean_seasonal_cycle(self, date, data):
        mean_sea_df = pd.DataFrame({'date': date, 'value': data})
        mean_sea_df = mean_sea_df[~(mean_sea_df['date'].dt.month == 2) | ~(mean_sea_df['date'].dt.day == 29)]
        mean_sea_df['month_day'] = mean_sea_df['date'].dt.strftime('%m-%d')
        mean_sea_avg = mean_sea_df.groupby('month_day')['value'].mean().reset_index()
        return mean_sea_avg['value'].to_numpy()
