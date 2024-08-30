# -*- coding: utf-8 -*-
"""
Created on 2023.10.29
@author: Mao Huihui
"""
import sys
import numpy as np
from pymoo.core.problem import Problem


class SWATProblem(Problem):  # Object-oriented definition Problem which implements a method evaluating a set of solutions
    def __init__(self, swat_execution_run, **kwargs):
        self.swat_execution_run = swat_execution_run
        self.swat_para          = swat_execution_run.swat_parameter
        self.cali_scheme        = swat_execution_run.cal_scheme
        self.objectf            = swat_execution_run.objective_funs
        self.objectf_num        = swat_execution_run.obj_func_num
        self.constr_num         = swat_execution_run.constraint_num
        self.cal_vars           = swat_execution_run.cal_vars_list
        self.hydro_stas         = swat_execution_run.hydro_stas

        xl = [para_idx[1][0] for para_idx in self.swat_para]
        xu = [para_idx[1][1] for para_idx in self.swat_para]

        super().__init__(n_var=len(self.swat_para),      # Integer value representing the number of design variables
                         n_obj=self.objectf_num,         # Integer value representing the number of objectives
                         n_ieq_constr=self.constr_num,   # Integer value representing the number of inequality constraints
                         xl=np.array(xl),                # Float or np.ndarray of length n_var representing the lower bounds of the design variables
                         xu=np.array(xu),                # Float or np.ndarray of length n_var representing the upper bounds of the design variables
                         **kwargs)


    def _evaluate(self, x, out, *args, **kwargs):
        # One Variables
        if self.cal_vars == ['Streamflow']:
            if self.cali_scheme == 'Multi-site':
                if len(self.hydro_stas) == 2:
                    # SWAT Model Simulation
                    rch_pop_obj_func_list = self.swat_execution_run.SWAT_model_execution(x)
                    NingDu_obj_func  = np.array(rch_pop_obj_func_list[0])
                    FenKeng_obj_func = np.array(rch_pop_obj_func_list[1])
                    if self.objectf[0] in ['NSE', 'KGE', 'R2']:
                        # Calculate Objective Functions and Constraints
                        NingDu_f_Q  = 1 - NingDu_obj_func   # f = 1 - KGE (0~+∞, minimized)
                        FenKeng_f_Q = 1 - FenKeng_obj_func  # f = 1 - KGE (0~+∞, minimized)

                        NingDu_g_Q  = -NingDu_obj_func   # -KGE <= 0
                        FenKeng_g_Q = -FenKeng_obj_func  # -KGE <= 0

                        out['F'] = np.column_stack([NingDu_f_Q, FenKeng_f_Q])  # Objective functions
                        out['G'] = np.column_stack([NingDu_g_Q, FenKeng_g_Q])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf[0] in ['RMSE', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        NingDu_f_Q  = NingDu_obj_func   # f = RMSE (0~+∞, minimized)
                        FenKeng_f_Q = FenKeng_obj_func  # f = RMSE (0~+∞, minimized)

                        out['F'] = np.column_stack([NingDu_f_Q, FenKeng_f_Q])  # Objective functions

                        print('F:', out['F'].shape)
                elif len(self.hydro_stas) == 3:
                    # SWAT Model Simulation
                    rch_pop_obj_func_list = self.swat_execution_run.SWAT_model_execution(x)
                    NingDu_obj_func   = np.array(rch_pop_obj_func_list[0])
                    ShiCheng_obj_func = np.array(rch_pop_obj_func_list[1])
                    FenKeng_obj_func  = np.array(rch_pop_obj_func_list[2])

                    if self.objectf[0] in ['NSE', 'KGE', 'R2']:
                        # Calculate Objective Functions and Constraints
                        NingDu_f_Q   = 1 - NingDu_obj_func    # f = 1 - KGE (0~+∞, minimized)
                        ShiCheng_f_Q = 1 - ShiCheng_obj_func  # f = 1 - KGE (0~+∞, minimized)
                        FenKeng_f_Q  = 1 - FenKeng_obj_func   # f = 1 - KGE (0~+∞, minimized)

                        # NingDu_g_Q   = -NingDu_obj_func       # -KGE <= 0
                        # ShiCheng_g_Q = -ShiCheng_obj_func     # -KGE <= 0
                        # FenKeng_g_Q  = -FenKeng_obj_func      # -KGE <= 0

                        out['F'] = np.column_stack([NingDu_f_Q, ShiCheng_f_Q, FenKeng_f_Q])  # Objective functions
                        # out['G'] = np.column_stack([NingDu_g_Q, ShiCheng_g_Q, FenKeng_g_Q])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf[0] in ['RMSE', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        NingDu_f_Q   = NingDu_obj_func        # f = RMSE (0~+∞, minimized)
                        ShiCheng_f_Q = ShiCheng_obj_func      # f = RMSE (0~+∞, minimized)
                        FenKeng_f_Q  = FenKeng_obj_func       # f = RMSE (0~+∞, minimized)

                        out['F'] = np.column_stack([NingDu_f_Q, ShiCheng_f_Q, FenKeng_f_Q])  # Objective functions

                        print('F:', out['F'].shape)

            elif self.cali_scheme == 'Multi-objective':
                if len(self.hydro_stas) == 1:
                    if len(self.objectf) == 2:
                        # SWAT Model Simulation
                        rch_pop_obj_func1_list, rch_pop_obj_func2_list = self.swat_execution_run.SWAT_model_execution(x)
                        FenKeng_obj_func1 = np.array(rch_pop_obj_func1_list[0])
                        FenKeng_obj_func2 = np.array(rch_pop_obj_func2_list[0])

                        if self.objectf == ['NSE', 'KGE']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_NSE_Q = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_KGE_Q = 1 - FenKeng_obj_func2  # f = 1 - KGE (0~+∞, minimized)

                            FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0
                            FenKeng_g_KGE_Q = -FenKeng_obj_func2  # -KGE <= 0

                            out['F'] = np.column_stack([FenKeng_f_NSE_Q, FenKeng_f_KGE_Q])  # Objective functions
                            out['G'] = np.column_stack([FenKeng_g_NSE_Q, FenKeng_g_KGE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'R2']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_NSE_Q = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_R2_Q  = 1 - FenKeng_obj_func2  # f = 1 - R2 (0~+∞, minimized)

                            FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0
                            FenKeng_g_R2_Q  = -FenKeng_obj_func2  # -R2 <= 0

                            out['F'] = np.column_stack([FenKeng_f_NSE_Q, FenKeng_f_R2_Q])  # Objective functions
                            out['G'] = np.column_stack([FenKeng_g_NSE_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'PBIAS']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_NSE_Q   = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func2  # f = PBIAS (0~+∞, minimized)

                            # FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0

                            out['F'] = np.column_stack([FenKeng_f_NSE_Q, FenKeng_f_PBIAS_Q])  # Objective functions
                            # out['G'] = np.column_stack([FenKeng_g_NSE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_NSE_Q  = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q = FenKeng_obj_func2  # f = RMSE (0~+∞, minimized)

                            FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0

                            out['F'] = np.column_stack([FenKeng_f_NSE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([FenKeng_g_NSE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['KGE', 'R2']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_KGE_Q = 1 - FenKeng_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_R2_Q  = 1 - FenKeng_obj_func2  # f = 1 - R2 (0~+∞, minimized)

                            FenKeng_g_KGE_Q = -FenKeng_obj_func1  # -KGE <= 0
                            FenKeng_g_R2_Q  = -FenKeng_obj_func2  # -R2 <= 0

                            out['F'] = np.column_stack([FenKeng_f_KGE_Q, FenKeng_f_R2_Q])  # Objective functions
                            out['G'] = np.column_stack([FenKeng_g_KGE_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['KGE', 'PBIAS']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_KGE_Q   = 1 - FenKeng_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func2  # f = PBIAS (0~+∞, minimized)

                            FenKeng_g_KGE_Q = -FenKeng_obj_func1  # -KGE <= 0

                            out['F'] = np.column_stack([FenKeng_f_KGE_Q, FenKeng_f_PBIAS_Q])  # Objective functions
                            out['G'] = np.column_stack([FenKeng_g_KGE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['KGE', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_KGE_Q  = 1 - FenKeng_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q = FenKeng_obj_func2  # f = RMSE (0~+∞, minimized)

                            FenKeng_g_KGE_Q = -FenKeng_obj_func1  # -KGE <= 0

                            out['F'] = np.column_stack([FenKeng_f_KGE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([FenKeng_g_KGE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['R2', 'PBIAS']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_R2_Q    = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func2  # f = PBIAS (0~+∞, minimized)

                            # FenKeng_g_R2_Q = -FenKeng_obj_func1  # -R2 <= 0

                            out['F'] = np.column_stack([FenKeng_f_R2_Q, FenKeng_f_PBIAS_Q])  # Objective functions
                            # out['G'] = np.column_stack([FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['R2', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_R2_Q   = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q = FenKeng_obj_func2  # f = RMSE (0~+∞, minimized)

                            FenKeng_g_R2_Q = -FenKeng_obj_func1  # -R2 <= 0

                            out['F'] = np.column_stack([FenKeng_f_R2_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['PBIAS', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func1  # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_RMSE_Q  = FenKeng_obj_func2  # f = RMSE (0~+∞, minimized)

                            out['F'] = np.column_stack([FenKeng_f_PBIAS_Q, FenKeng_f_RMSE_Q])  # Objective functions

                            print('F:', out['F'].shape)
                    elif len(self.objectf) == 3:
                        # SWAT Model Simulation
                        rch_pop_obj_func1_list, rch_pop_obj_func2_list, rch_pop_obj_func3_list = self.swat_execution_run.SWAT_model_execution(x)
                        FenKeng_obj_func1 = np.array(rch_pop_obj_func1_list[0])
                        FenKeng_obj_func2 = np.array(rch_pop_obj_func2_list[0])
                        FenKeng_obj_func3 = np.array(rch_pop_obj_func3_list[0])

                        if self.objectf == ['NSE', 'KGE', 'R2']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_NSE_Q = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_KGE_Q = 1 - FenKeng_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_R2_Q  = 1 - FenKeng_obj_func3  # f = 1 - R2 (0~+∞, minimized)

                            FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0
                            FenKeng_g_KGE_Q = -FenKeng_obj_func2  # -KGE <= 0
                            FenKeng_g_R2_Q  = -FenKeng_obj_func3  # -R2 <= 0

                            out['F'] = np.column_stack([FenKeng_f_NSE_Q, FenKeng_f_KGE_Q, FenKeng_f_R2_Q])  # Objective functions
                            out['G'] = np.column_stack([FenKeng_g_NSE_Q, FenKeng_g_KGE_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'KGE', 'PBIAS']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_NSE_Q   = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_KGE_Q   = 1 - FenKeng_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func3  # f = PBIAS (0~+∞, minimized)

                            FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0
                            FenKeng_g_KGE_Q = -FenKeng_obj_func2  # -KGE <= 0

                            out['F'] = np.column_stack([FenKeng_f_NSE_Q, FenKeng_f_KGE_Q, FenKeng_f_PBIAS_Q])  # Objective functions
                            out['G'] = np.column_stack([FenKeng_g_NSE_Q, FenKeng_g_KGE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'KGE', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_NSE_Q  = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_KGE_Q  = 1 - FenKeng_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q = FenKeng_obj_func3  # f = RMSE (0~+∞, minimized)

                            FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0
                            FenKeng_g_KGE_Q = -FenKeng_obj_func2  # -KGE <= 0

                            out['F'] = np.column_stack([FenKeng_f_NSE_Q, FenKeng_f_KGE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([FenKeng_g_NSE_Q, FenKeng_g_KGE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'R2', 'PBIAS']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_NSE_Q   = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_R2_Q    = 1 - FenKeng_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func3  # f = PBIAS (0~+∞, minimized)

                            FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0
                            FenKeng_g_R2_Q  = -FenKeng_obj_func2  # -R2 <= 0

                            out['F'] = np.column_stack([FenKeng_f_NSE_Q, FenKeng_f_R2_Q, FenKeng_f_PBIAS_Q])  # Objective functions
                            out['G'] = np.column_stack([FenKeng_g_NSE_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'R2', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_NSE_Q  = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_R2_Q   = 1 - FenKeng_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_RMSE_Q = FenKeng_obj_func3  # f = RMSE (0~+∞, minimized)

                            FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0
                            FenKeng_g_R2_Q  = -FenKeng_obj_func2  # -R2 <= 0

                            out['F'] = np.column_stack([FenKeng_f_NSE_Q, FenKeng_f_R2_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([FenKeng_g_NSE_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'PBIAS', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_NSE_Q   = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func2  # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_RMSE_Q  = FenKeng_obj_func3  # f = RMSE (0~+∞, minimized)

                            FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0

                            out['F'] = np.column_stack([FenKeng_f_NSE_Q, FenKeng_f_PBIAS_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([FenKeng_g_NSE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['KGE', 'R2', 'PBIAS']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_KGE_Q   = 1 - FenKeng_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_R2_Q    = 1 - FenKeng_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func3  # f = PBIAS (0~+∞, minimized)

                            FenKeng_g_KGE_Q = -FenKeng_obj_func1  # -KGE <= 0
                            FenKeng_g_R2_Q  = -FenKeng_obj_func2  # -R2 <= 0

                            out['F'] = np.column_stack([FenKeng_f_KGE_Q, FenKeng_f_R2_Q, FenKeng_f_PBIAS_Q])  # Objective functions
                            out['G'] = np.column_stack([FenKeng_g_KGE_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['KGE', 'R2', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_KGE_Q  = 1 - FenKeng_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_R2_Q   = 1 - FenKeng_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_RMSE_Q = FenKeng_obj_func3  # f = RMSE (0~+∞, minimized)

                            FenKeng_g_KGE_Q = -FenKeng_obj_func1  # -KGE <= 0
                            FenKeng_g_R2_Q  = -FenKeng_obj_func2  # -KGE <= 0

                            out['F'] = np.column_stack([FenKeng_f_KGE_Q, FenKeng_f_R2_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([FenKeng_g_KGE_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['KGE', 'PBIAS', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_KGE_Q   = 1 - FenKeng_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func2  # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_RMSE_Q  = FenKeng_obj_func3  # f = RMSE (0~+∞, minimized)

                            FenKeng_g_KGE_Q = -FenKeng_obj_func1  # -KGE <= 0

                            out['F'] = np.column_stack([FenKeng_f_KGE_Q, FenKeng_f_PBIAS_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([FenKeng_g_KGE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['R2', 'PBIAS', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            FenKeng_f_R2_Q    = 1 - FenKeng_obj_func1  # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func2  # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_RMSE_Q  = FenKeng_obj_func3  # f = RMSE (0~+∞, minimized)

                            FenKeng_g_R2_Q = -FenKeng_obj_func1  # -R2 <= 0

                            out['F'] = np.column_stack([FenKeng_f_R2_Q, FenKeng_f_PBIAS_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)
                elif len(self.hydro_stas) == 2:
                    if len(self.objectf) == 2:
                        # SWAT Model Simulation
                        rch_pop_obj_func1_list, rch_pop_obj_func2_list = self.swat_execution_run.SWAT_model_execution(x)
                        NingDu_obj_func1  = np.array(rch_pop_obj_func1_list[0])
                        FenKeng_obj_func1 = np.array(rch_pop_obj_func1_list[1])

                        NingDu_obj_func2  = np.array(rch_pop_obj_func2_list[0])
                        FenKeng_obj_func2 = np.array(rch_pop_obj_func2_list[1])
                        if self.objectf == ['NSE', 'KGE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q  = 1 - NingDu_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_KGE_Q  = 1 - NingDu_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q = 1 - FenKeng_obj_func2  # f = 1 - KGE (0~+∞, minimized)

                            NingDu_g_NSE_Q  = -NingDu_obj_func1  # -NSE <= 0
                            FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0

                            NingDu_g_KGE_Q  = -NingDu_obj_func2  # -KGE <= 0
                            FenKeng_g_KGE_Q = -FenKeng_obj_func2  # -KGE <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, FenKeng_f_NSE_Q, NingDu_f_KGE_Q, FenKeng_f_KGE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, FenKeng_g_NSE_Q, NingDu_g_KGE_Q, FenKeng_g_KGE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'R2']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q  = 1 - NingDu_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_R2_Q  = 1 - NingDu_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_R2_Q = 1 - FenKeng_obj_func2  # f = 1 - R2 (0~+∞, minimized)

                            NingDu_g_NSE_Q  = -NingDu_obj_func1  # -NSE <= 0
                            FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0

                            NingDu_g_R2_Q  = -NingDu_obj_func2  # -R2 <= 0
                            FenKeng_g_R2_Q = -FenKeng_obj_func2  # -R2 <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, FenKeng_f_NSE_Q, NingDu_f_R2_Q, FenKeng_f_R2_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, FenKeng_g_NSE_Q, NingDu_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'PBIAS']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q  = 1 - NingDu_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_PBIAS_Q  = NingDu_obj_func2  # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func2  # f = PBIAS (0~+∞, minimized)

                            NingDu_g_NSE_Q  = -NingDu_obj_func1  # -NSE <= 0
                            FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, FenKeng_f_NSE_Q, NingDu_f_PBIAS_Q, FenKeng_f_PBIAS_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, FenKeng_g_NSE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q  = 1 - NingDu_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_RMSE_Q  = NingDu_obj_func2  # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q = FenKeng_obj_func2  # f = RMSE (0~+∞, minimized)

                            NingDu_g_NSE_Q  = -NingDu_obj_func1  # -NSE <= 0
                            FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, FenKeng_f_NSE_Q, NingDu_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, FenKeng_g_NSE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['KGE', 'R2']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_KGE_Q  = 1 - NingDu_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q = 1 - FenKeng_obj_func1  # f = 1 - KGE (0~+∞, minimized)

                            NingDu_f_R2_Q  = 1 - NingDu_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_R2_Q = 1 - FenKeng_obj_func2  # f = 1 - R2 (0~+∞, minimized)

                            NingDu_g_KGE_Q  = -NingDu_obj_func1  # -KGE <= 0
                            FenKeng_g_KGE_Q = -FenKeng_obj_func1  # -KGE <= 0

                            NingDu_g_R2_Q  = -NingDu_obj_func2  # -R2 <= 0
                            FenKeng_g_R2_Q = -FenKeng_obj_func2  # -R2 <= 0

                            out['F'] = np.column_stack([NingDu_f_KGE_Q, FenKeng_f_KGE_Q, NingDu_f_R2_Q, FenKeng_f_R2_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_KGE_Q, FenKeng_g_KGE_Q, NingDu_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['KGE', 'PBIAS']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_KGE_Q  = 1 - NingDu_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q = 1 - FenKeng_obj_func1  # f = 1 - KGE (0~+∞, minimized)

                            NingDu_f_PBIAS_Q  = NingDu_obj_func2  # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func2  # f = PBIAS (0~+∞, minimized)

                            NingDu_g_KGE_Q  = -NingDu_obj_func1  # -KGE <= 0
                            FenKeng_g_KGE_Q = -FenKeng_obj_func1  # -KGE <= 0

                            out['F'] = np.column_stack([NingDu_f_KGE_Q, FenKeng_f_KGE_Q, NingDu_f_PBIAS_Q, FenKeng_f_PBIAS_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_KGE_Q, FenKeng_g_KGE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['KGE', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_KGE_Q  = 1 - NingDu_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q = 1 - FenKeng_obj_func1  # f = 1 - KGE (0~+∞, minimized)

                            NingDu_f_RMSE_Q  = NingDu_obj_func2  # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q = FenKeng_obj_func2  # f = RMSE (0~+∞, minimized)

                            NingDu_g_KGE_Q  = -NingDu_obj_func1  # -KGE <= 0
                            FenKeng_g_KGE_Q = -FenKeng_obj_func1  # -KGE <= 0

                            out['F'] = np.column_stack([NingDu_f_KGE_Q, FenKeng_f_KGE_Q, NingDu_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_KGE_Q, FenKeng_g_KGE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['R2', 'PBIAS']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_R2_Q  = 1 - NingDu_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_R2_Q = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_PBIAS_Q  = NingDu_obj_func2  # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func2  # f = PBIAS (0~+∞, minimized)

                            NingDu_g_R2_Q  = -NingDu_obj_func1  # -R2 <= 0
                            FenKeng_g_R2_Q = -FenKeng_obj_func1  # -R2 <= 0

                            out['F'] = np.column_stack([NingDu_f_R2_Q, FenKeng_f_R2_Q, NingDu_f_PBIAS_Q, FenKeng_f_PBIAS_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['R2', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_R2_Q  = 1 - NingDu_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_R2_Q = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_RMSE_Q  = NingDu_obj_func2  # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q = FenKeng_obj_func2  # f = RMSE (0~+∞, minimized)

                            NingDu_g_R2_Q  = -NingDu_obj_func1  # -R2 <= 0
                            FenKeng_g_R2_Q = -FenKeng_obj_func1  # -R2 <= 0

                            out['F'] = np.column_stack([NingDu_f_R2_Q, FenKeng_f_R2_Q, NingDu_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['PBIAS', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_PBIAS_Q  = NingDu_obj_func1  # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func1  # f = PBIAS (0~+∞, minimized)

                            NingDu_f_RMSE_Q  = NingDu_obj_func2  # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q = FenKeng_obj_func2  # f = RMSE (0~+∞, minimized)

                            out['F'] = np.column_stack([NingDu_f_PBIAS_Q, FenKeng_f_PBIAS_Q, NingDu_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions

                            print('F:', out['F'].shape)
                    elif len(self.objectf) == 3:
                        # SWAT Model Simulation
                        rch_pop_obj_func1_list, rch_pop_obj_func2_list, rch_pop_obj_func3_list = self.swat_execution_run.SWAT_model_execution(x)
                        NingDu_obj_func1  = np.array(rch_pop_obj_func1_list[0])
                        FenKeng_obj_func1 = np.array(rch_pop_obj_func1_list[1])

                        NingDu_obj_func2  = np.array(rch_pop_obj_func2_list[0])
                        FenKeng_obj_func2 = np.array(rch_pop_obj_func2_list[1])

                        NingDu_obj_func3  = np.array(rch_pop_obj_func3_list[0])
                        FenKeng_obj_func3 = np.array(rch_pop_obj_func3_list[1])

                        if self.objectf == ['NSE', 'KGE', 'R2']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q  = 1 - NingDu_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_KGE_Q  = 1 - NingDu_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q = 1 - FenKeng_obj_func2  # f = 1 - KGE (0~+∞, minimized)

                            NingDu_f_R2_Q  = 1 - NingDu_obj_func3  # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_R2_Q = 1 - FenKeng_obj_func3  # f = 1 - R2 (0~+∞, minimized)

                            NingDu_g_NSE_Q  = -NingDu_obj_func1  # -NSE <= 0
                            FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0

                            NingDu_g_KGE_Q  = -NingDu_obj_func2  # -KGE <= 0
                            FenKeng_g_KGE_Q = -FenKeng_obj_func2  # -KGE <= 0

                            NingDu_g_R2_Q  = -NingDu_obj_func3  # -R2 <= 0
                            FenKeng_g_R2_Q = -FenKeng_obj_func3  # -R2 <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, FenKeng_f_NSE_Q,
                                                        NingDu_f_KGE_Q, FenKeng_f_KGE_Q,
                                                        NingDu_f_R2_Q, FenKeng_f_R2_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, FenKeng_g_NSE_Q,
                                                        NingDu_g_KGE_Q, FenKeng_g_KGE_Q,
                                                        NingDu_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'KGE', 'PBIAS']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q  = 1 - NingDu_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_KGE_Q  = 1 - NingDu_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q = 1 - FenKeng_obj_func2  # f = 1 - KGE (0~+∞, minimized)

                            NingDu_f_PBIAS_Q  = NingDu_obj_func3  # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func3  # f = PBIAS (0~+∞, minimized)

                            NingDu_g_NSE_Q  = -NingDu_obj_func1  # -NSE <= 0
                            FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0

                            NingDu_g_KGE_Q  = -NingDu_obj_func2  # -KGE <= 0
                            FenKeng_g_KGE_Q = -FenKeng_obj_func2  # -KGE <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, FenKeng_f_NSE_Q,
                                                        NingDu_f_KGE_Q, FenKeng_f_KGE_Q,
                                                        NingDu_f_PBIAS_Q, FenKeng_f_PBIAS_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, FenKeng_g_NSE_Q,
                                                        NingDu_g_KGE_Q, FenKeng_g_KGE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'KGE', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q  = 1 - NingDu_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_KGE_Q  = 1 - NingDu_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q = 1 - FenKeng_obj_func2  # f = 1 - KGE (0~+∞, minimized)

                            NingDu_f_RMSE_Q  = NingDu_obj_func3  # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q = FenKeng_obj_func3  # f = RMSE (0~+∞, minimized)

                            NingDu_g_NSE_Q  = -NingDu_obj_func1  # -NSE <= 0
                            FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0

                            NingDu_g_KGE_Q  = -NingDu_obj_func2  # -KGE <= 0
                            FenKeng_g_KGE_Q = -FenKeng_obj_func2  # -KGE <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, FenKeng_f_NSE_Q,
                                                        NingDu_f_KGE_Q, FenKeng_f_KGE_Q,
                                                        NingDu_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, FenKeng_g_NSE_Q,
                                                        NingDu_g_KGE_Q, FenKeng_g_KGE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'R2', 'PBIAS']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q  = 1 - NingDu_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_R2_Q  = 1 - NingDu_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_R2_Q = 1 - FenKeng_obj_func2  # f = 1 - R2 (0~+∞, minimized)

                            NingDu_f_PBIAS_Q  = NingDu_obj_func3  # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func3  # f = PBIAS (0~+∞, minimized)

                            NingDu_g_NSE_Q  = -NingDu_obj_func1  # -NSE <= 0
                            FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0

                            NingDu_g_R2_Q  = -NingDu_obj_func2  # -R2 <= 0
                            FenKeng_g_R2_Q = -FenKeng_obj_func2  # -R2 <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, FenKeng_f_NSE_Q,
                                                        NingDu_f_R2_Q, FenKeng_f_R2_Q,
                                                        NingDu_f_PBIAS_Q, FenKeng_f_PBIAS_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, FenKeng_g_NSE_Q,
                                                        NingDu_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'R2', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q  = 1 - NingDu_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_R2_Q  = 1 - NingDu_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_R2_Q = 1 - FenKeng_obj_func2  # f = 1 - R2 (0~+∞, minimized)

                            NingDu_f_RMSE_Q  = NingDu_obj_func3  # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q = FenKeng_obj_func3  # f = RMSE (0~+∞, minimized)

                            NingDu_g_NSE_Q  = -NingDu_obj_func1  # -NSE <= 0
                            FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0

                            NingDu_g_R2_Q  = -NingDu_obj_func2  # -R2 <= 0
                            FenKeng_g_R2_Q = -FenKeng_obj_func2  # -R2 <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, FenKeng_f_NSE_Q,
                                                        NingDu_f_R2_Q, FenKeng_f_R2_Q,
                                                        NingDu_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, FenKeng_g_NSE_Q,
                                                        NingDu_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'PBIAS', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q  = 1 - NingDu_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q = 1 - FenKeng_obj_func1  # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_PBIAS_Q  = NingDu_obj_func2  # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func2  # f = PBIAS (0~+∞, minimized)

                            NingDu_f_RMSE_Q  = NingDu_obj_func3  # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q = FenKeng_obj_func3  # f = RMSE (0~+∞, minimized)

                            NingDu_g_NSE_Q  = -NingDu_obj_func1  # -NSE <= 0
                            FenKeng_g_NSE_Q = -FenKeng_obj_func1  # -NSE <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, FenKeng_f_NSE_Q,
                                                        NingDu_f_PBIAS_Q, FenKeng_f_PBIAS_Q,
                                                        NingDu_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, FenKeng_g_NSE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['KGE', 'R2', 'PBIAS']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_KGE_Q  = 1 - NingDu_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q = 1 - FenKeng_obj_func1  # f = 1 - KGE (0~+∞, minimized)

                            NingDu_f_R2_Q  = 1 - NingDu_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_R2_Q = 1 - FenKeng_obj_func2  # f = 1 - R2 (0~+∞, minimized)

                            NingDu_f_PBIAS_Q  = NingDu_obj_func3  # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func3  # f = PBIAS (0~+∞, minimized)

                            NingDu_g_KGE_Q  = -NingDu_obj_func1  # -KGE <= 0
                            FenKeng_g_KGE_Q = -FenKeng_obj_func1  # -KGE <= 0

                            NingDu_g_R2_Q  = -NingDu_obj_func2  # -R2 <= 0
                            FenKeng_g_R2_Q = -FenKeng_obj_func2  # -R2 <= 0

                            out['F'] = np.column_stack([NingDu_f_KGE_Q, FenKeng_f_KGE_Q,
                                                        NingDu_f_R2_Q, FenKeng_f_R2_Q,
                                                        NingDu_f_PBIAS_Q, FenKeng_f_PBIAS_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_KGE_Q, FenKeng_g_KGE_Q,
                                                        NingDu_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['KGE', 'R2', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_KGE_Q  = 1 - NingDu_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q = 1 - FenKeng_obj_func1  # f = 1 - KGE (0~+∞, minimized)

                            NingDu_f_R2_Q  = 1 - NingDu_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_R2_Q = 1 - FenKeng_obj_func2  # f = 1 - R2 (0~+∞, minimized)

                            NingDu_f_RMSE_Q  = NingDu_obj_func3  # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q = FenKeng_obj_func3  # f = RMSE (0~+∞, minimized)

                            NingDu_g_KGE_Q  = -NingDu_obj_func1  # -KGE <= 0
                            FenKeng_g_KGE_Q = -FenKeng_obj_func1  # -KGE <= 0

                            NingDu_g_R2_Q  = -NingDu_obj_func2  # -KGE <= 0
                            FenKeng_g_R2_Q = -FenKeng_obj_func2  # -KGE <= 0

                            out['F'] = np.column_stack([NingDu_f_KGE_Q, FenKeng_f_KGE_Q,
                                                        NingDu_f_R2_Q, FenKeng_f_R2_Q,
                                                        NingDu_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_KGE_Q, FenKeng_g_KGE_Q,
                                                        NingDu_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['KGE', 'PBIAS', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_KGE_Q  = 1 - NingDu_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q = 1 - FenKeng_obj_func1  # f = 1 - KGE (0~+∞, minimized)

                            NingDu_f_PBIAS_Q  = NingDu_obj_func2  # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func2  # f = PBIAS (0~+∞, minimized)

                            NingDu_f_RMSE_Q  = NingDu_obj_func3  # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q = FenKeng_obj_func3  # f = RMSE (0~+∞, minimized)

                            NingDu_g_KGE_Q  = -NingDu_obj_func1  # -KGE <= 0
                            FenKeng_g_KGE_Q = -FenKeng_obj_func1  # -KGE <= 0

                            out['F'] = np.column_stack([NingDu_f_KGE_Q, FenKeng_f_KGE_Q,
                                                        NingDu_f_PBIAS_Q, FenKeng_f_PBIAS_Q,
                                                        NingDu_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_KGE_Q, FenKeng_g_KGE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['R2', 'PBIAS', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_R2_Q  = 1 - NingDu_obj_func1  # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_R2_Q = 1 - FenKeng_obj_func1  # f = 1 - R2 (0~+∞, minimized)

                            NingDu_f_PBIAS_Q  = NingDu_obj_func2  # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q = FenKeng_obj_func2  # f = PBIAS (0~+∞, minimized)

                            NingDu_f_RMSE_Q  = NingDu_obj_func3  # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q = FenKeng_obj_func3  # f = RMSE (0~+∞, minimized)

                            NingDu_g_R2_Q  = -NingDu_obj_func1  # -R2 <= 0
                            FenKeng_g_R2_Q = -FenKeng_obj_func1  # -R2 <= 0

                            out['F'] = np.column_stack([NingDu_f_R2_Q, FenKeng_f_R2_Q,
                                                        NingDu_f_PBIAS_Q, FenKeng_f_PBIAS_Q,
                                                        NingDu_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)
                elif len(self.hydro_stas) == 3:
                    if len(self.objectf) == 2:
                        # SWAT Model Simulation
                        rch_pop_obj_func1_list, rch_pop_obj_func2_list = self.swat_execution_run.SWAT_model_execution(x)

                        print('rch_pop_obj_func1_list:', len(rch_pop_obj_func1_list), rch_pop_obj_func1_list)
                        print('rch_pop_obj_func2_list:', len(rch_pop_obj_func2_list), rch_pop_obj_func2_list)

                        NingDu_obj_func1   = np.array(rch_pop_obj_func1_list[0])
                        ShiCheng_obj_func1 = np.array(rch_pop_obj_func1_list[1])
                        FenKeng_obj_func1  = np.array(rch_pop_obj_func1_list[2])

                        NingDu_obj_func2   = np.array(rch_pop_obj_func2_list[0])
                        ShiCheng_obj_func2 = np.array(rch_pop_obj_func2_list[1])
                        FenKeng_obj_func2  = np.array(rch_pop_obj_func2_list[2])

                        if self.objectf == ['NSE', 'KGE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q   = 1 - NingDu_obj_func1    # f = 1 - NSE (0~+∞, minimized)
                            ShiCheng_f_NSE_Q = 1 - ShiCheng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q  = 1 - FenKeng_obj_func1   # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_KGE_Q   = 1 - NingDu_obj_func2    # f = 1 - KGE (0~+∞, minimized)
                            ShiCheng_f_KGE_Q = 1 - ShiCheng_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q  = 1 - FenKeng_obj_func2   # f = 1 - KGE (0~+∞, minimized)

                            NingDu_g_NSE_Q   = -NingDu_obj_func1       # -NSE <= 0
                            ShiCheng_g_NSE_Q = -ShiCheng_obj_func1     # -NSE <= 0
                            FenKeng_g_NSE_Q  = -FenKeng_obj_func1      # -NSE <= 0

                            NingDu_g_KGE_Q   = -NingDu_obj_func2       # -KGE <= 0
                            ShiCheng_g_KGE_Q = -ShiCheng_obj_func2     # -KGE <= 0
                            FenKeng_g_KGE_Q  = -FenKeng_obj_func2      # -KGE <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, ShiCheng_f_NSE_Q, FenKeng_f_NSE_Q,
                                                        NingDu_f_KGE_Q, ShiCheng_f_KGE_Q, FenKeng_f_KGE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, ShiCheng_g_NSE_Q, FenKeng_g_NSE_Q,
                                                        NingDu_g_KGE_Q, ShiCheng_g_KGE_Q, FenKeng_g_KGE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'R2']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q   = 1 - NingDu_obj_func1    # f = 1 - NSE (0~+∞, minimized)
                            ShiCheng_f_NSE_Q = 1 - ShiCheng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q  = 1 - FenKeng_obj_func1   # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_R2_Q   = 1 - NingDu_obj_func2     # f = 1 - R2 (0~+∞, minimized)
                            ShiCheng_f_R2_Q = 1 - ShiCheng_obj_func2   # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_R2_Q  = 1 - FenKeng_obj_func2    # f = 1 - R2 (0~+∞, minimized)

                            NingDu_g_NSE_Q   = -NingDu_obj_func1       # -NSE <= 0
                            ShiCheng_g_NSE_Q = -ShiCheng_obj_func1     # -NSE <= 0
                            FenKeng_g_NSE_Q  = -FenKeng_obj_func1      # -NSE <= 0

                            NingDu_g_R2_Q   = -NingDu_obj_func2        # -R2 <= 0
                            ShiCheng_g_R2_Q = -ShiCheng_obj_func2      # -R2 <= 0
                            FenKeng_g_R2_Q  = -FenKeng_obj_func2       # -R2 <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, ShiCheng_f_NSE_Q, FenKeng_f_NSE_Q,
                                                        NingDu_f_R2_Q, ShiCheng_f_R2_Q, FenKeng_f_R2_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, ShiCheng_g_NSE_Q, FenKeng_g_NSE_Q,
                                                        NingDu_g_R2_Q, ShiCheng_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'PBIAS']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q   = 1 - NingDu_obj_func1    # f = 1 - NSE (0~+∞, minimized)
                            ShiCheng_f_NSE_Q = 1 - ShiCheng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q  = 1 - FenKeng_obj_func1   # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_PBIAS_Q   = NingDu_obj_func2      # f = PBIAS (0~+∞, minimized)
                            ShiCheng_f_PBIAS_Q = ShiCheng_obj_func2    # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q  = FenKeng_obj_func2     # f = PBIAS (0~+∞, minimized)

                            NingDu_g_NSE_Q   = -NingDu_obj_func1       # -NSE <= 0
                            ShiCheng_g_NSE_Q = -ShiCheng_obj_func1     # -NSE <= 0
                            FenKeng_g_NSE_Q  = -FenKeng_obj_func1      # -NSE <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, ShiCheng_f_NSE_Q, FenKeng_f_NSE_Q,
                                                        NingDu_f_PBIAS_Q, ShiCheng_f_PBIAS_Q, FenKeng_f_PBIAS_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, ShiCheng_g_NSE_Q, FenKeng_g_NSE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q   = 1 - NingDu_obj_func1    # f = 1 - NSE (0~+∞, minimized)
                            ShiCheng_f_NSE_Q = 1 - ShiCheng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q  = 1 - FenKeng_obj_func1   # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_RMSE_Q   = NingDu_obj_func2       # f = RMSE (0~+∞, minimized)
                            ShiCheng_f_RMSE_Q = ShiCheng_obj_func2     # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q  = FenKeng_obj_func2      # f = RMSE (0~+∞, minimized)

                            NingDu_g_NSE_Q   = -NingDu_obj_func1       # -NSE <= 0
                            ShiCheng_g_NSE_Q = -ShiCheng_obj_func1     # -NSE <= 0
                            FenKeng_g_NSE_Q  = -FenKeng_obj_func1      # -NSE <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, ShiCheng_f_NSE_Q, FenKeng_f_NSE_Q,
                                                        NingDu_f_RMSE_Q, ShiCheng_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, ShiCheng_g_NSE_Q, FenKeng_g_NSE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['KGE', 'R2']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_KGE_Q   = 1 - NingDu_obj_func1    # f = 1 - KGE (0~+∞, minimized)
                            ShiCheng_f_KGE_Q = 1 - ShiCheng_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q  = 1 - FenKeng_obj_func1   # f = 1 - KGE (0~+∞, minimized)

                            NingDu_f_R2_Q   = 1 - NingDu_obj_func2     # f = 1 - R2 (0~+∞, minimized)
                            ShiCheng_f_R2_Q = 1 - ShiCheng_obj_func2   # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_R2_Q  = 1 - FenKeng_obj_func2    # f = 1 - R2 (0~+∞, minimized)

                            NingDu_g_KGE_Q   = -NingDu_obj_func1       # -KGE <= 0
                            ShiCheng_g_KGE_Q = -ShiCheng_obj_func1     # -KGE <= 0
                            FenKeng_g_KGE_Q  = -FenKeng_obj_func1      # -KGE <= 0

                            NingDu_g_R2_Q   = -NingDu_obj_func2        # -R2 <= 0
                            ShiCheng_g_R2_Q = -ShiCheng_obj_func2      # -R2 <= 0
                            FenKeng_g_R2_Q  = -FenKeng_obj_func2       # -R2 <= 0

                            out['F'] = np.column_stack([NingDu_f_KGE_Q, ShiCheng_f_KGE_Q, FenKeng_f_KGE_Q,
                                                        NingDu_f_R2_Q, ShiCheng_f_R2_Q, FenKeng_f_R2_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_KGE_Q, ShiCheng_g_KGE_Q, FenKeng_g_KGE_Q,
                                                        NingDu_g_R2_Q, ShiCheng_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['KGE', 'PBIAS']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_KGE_Q   = 1 - NingDu_obj_func1    # f = 1 - KGE (0~+∞, minimized)
                            ShiCheng_f_KGE_Q = 1 - ShiCheng_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q  = 1 - FenKeng_obj_func1   # f = 1 - KGE (0~+∞, minimized)

                            NingDu_f_PBIAS_Q   = NingDu_obj_func2      # f = PBIAS (0~+∞, minimized)
                            ShiCheng_f_PBIAS_Q = ShiCheng_obj_func2    # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q  = FenKeng_obj_func2     # f = PBIAS (0~+∞, minimized)

                            NingDu_g_KGE_Q   = -NingDu_obj_func1       # -KGE <= 0
                            ShiCheng_g_KGE_Q = -ShiCheng_obj_func1     # -KGE <= 0
                            FenKeng_g_KGE_Q  = -FenKeng_obj_func1      # -KGE <= 0

                            out['F'] = np.column_stack([NingDu_f_KGE_Q, ShiCheng_f_KGE_Q, FenKeng_f_KGE_Q,
                                                        NingDu_f_PBIAS_Q, ShiCheng_f_PBIAS_Q, FenKeng_f_PBIAS_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_KGE_Q, ShiCheng_g_KGE_Q, FenKeng_g_KGE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['KGE', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_KGE_Q   = 1 - NingDu_obj_func1    # f = 1 - KGE (0~+∞, minimized)
                            ShiCheng_f_KGE_Q = 1 - ShiCheng_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q  = 1 - FenKeng_obj_func1   # f = 1 - KGE (0~+∞, minimized)

                            NingDu_f_RMSE_Q   = NingDu_obj_func2       # f = RMSE (0~+∞, minimized)
                            ShiCheng_f_RMSE_Q = ShiCheng_obj_func2     # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q  = FenKeng_obj_func2      # f = RMSE (0~+∞, minimized)

                            NingDu_g_KGE_Q   = -NingDu_obj_func1       # -KGE <= 0
                            ShiCheng_g_KGE_Q = -ShiCheng_obj_func1     # -KGE <= 0
                            FenKeng_g_KGE_Q  = -FenKeng_obj_func1      # -KGE <= 0

                            out['F'] = np.column_stack([NingDu_f_KGE_Q, ShiCheng_f_KGE_Q, FenKeng_f_KGE_Q,
                                                        NingDu_f_RMSE_Q, ShiCheng_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_KGE_Q, ShiCheng_g_KGE_Q, FenKeng_g_KGE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['R2', 'PBIAS']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_R2_Q   = 1 - NingDu_obj_func1    # f = 1 - NSE (0~+∞, minimized)
                            ShiCheng_f_R2_Q = 1 - ShiCheng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_R2_Q  = 1 - FenKeng_obj_func1   # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_PBIAS_Q   = NingDu_obj_func2     # f = PBIAS (0~+∞, minimized)
                            ShiCheng_f_PBIAS_Q = ShiCheng_obj_func2   # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q  = FenKeng_obj_func2    # f = PBIAS (0~+∞, minimized)

                            NingDu_g_R2_Q   = -NingDu_obj_func1       # -R2 <= 0
                            ShiCheng_g_R2_Q = -ShiCheng_obj_func1     # -R2 <= 0
                            FenKeng_g_R2_Q  = -FenKeng_obj_func1      # -R2 <= 0

                            out['F'] = np.column_stack([NingDu_f_R2_Q, ShiCheng_f_R2_Q, FenKeng_f_R2_Q,
                                                        NingDu_f_PBIAS_Q, ShiCheng_f_PBIAS_Q, FenKeng_f_PBIAS_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_R2_Q, ShiCheng_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['R2', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_R2_Q   = 1 - NingDu_obj_func1    # f = 1 - NSE (0~+∞, minimized)
                            ShiCheng_f_R2_Q = 1 - ShiCheng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_R2_Q  = 1 - FenKeng_obj_func1   # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_RMSE_Q   = NingDu_obj_func2      # f = RMSE (0~+∞, minimized)
                            ShiCheng_f_RMSE_Q = ShiCheng_obj_func2    # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q  = FenKeng_obj_func2     # f = RMSE (0~+∞, minimized)

                            NingDu_g_R2_Q   = -NingDu_obj_func1       # -R2 <= 0
                            ShiCheng_g_R2_Q = -ShiCheng_obj_func1     # -R2 <= 0
                            FenKeng_g_R2_Q  = -FenKeng_obj_func1      # -R2 <= 0

                            out['F'] = np.column_stack([NingDu_f_R2_Q, ShiCheng_f_R2_Q, FenKeng_f_R2_Q,
                                                        NingDu_f_RMSE_Q, ShiCheng_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_R2_Q, ShiCheng_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['PBIAS', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_PBIAS_Q   = NingDu_obj_func1    # f = PBIAS (0~+∞, minimized)
                            ShiCheng_f_PBIAS_Q = ShiCheng_obj_func1  # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q  = FenKeng_obj_func1   # f = PBIAS (0~+∞, minimized)

                            NingDu_f_RMSE_Q   = NingDu_obj_func2     # f = RMSE (0~+∞, minimized)
                            ShiCheng_f_RMSE_Q = ShiCheng_obj_func2   # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q  = FenKeng_obj_func2    # f = RMSE (0~+∞, minimized)

                            out['F'] = np.column_stack([NingDu_f_PBIAS_Q, ShiCheng_f_PBIAS_Q, FenKeng_f_PBIAS_Q,
                                                        NingDu_f_RMSE_Q, ShiCheng_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions

                            print('F:', out['F'].shape)
                    elif len(self.objectf) == 3:
                        # SWAT Model Simulation
                        rch_pop_obj_func1_list, rch_pop_obj_func2_list, rch_pop_obj_func3_list = self.swat_execution_run.SWAT_model_execution(x)
                        NingDu_obj_func1   = np.array(rch_pop_obj_func1_list[0])
                        ShiCheng_obj_func1 = np.array(rch_pop_obj_func1_list[1])
                        FenKeng_obj_func1  = np.array(rch_pop_obj_func1_list[2])

                        NingDu_obj_func2   = np.array(rch_pop_obj_func2_list[0])
                        ShiCheng_obj_func2 = np.array(rch_pop_obj_func2_list[1])
                        FenKeng_obj_func2  = np.array(rch_pop_obj_func2_list[2])

                        NingDu_obj_func3   = np.array(rch_pop_obj_func3_list[0])
                        ShiCheng_obj_func3 = np.array(rch_pop_obj_func3_list[1])
                        FenKeng_obj_func3  = np.array(rch_pop_obj_func3_list[2])

                        if self.objectf == ['NSE', 'KGE', 'R2']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q   = 1 - NingDu_obj_func1    # f = 1 - NSE (0~+∞, minimized)
                            ShiCheng_f_NSE_Q = 1 - ShiCheng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q  = 1 - FenKeng_obj_func1   # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_KGE_Q   = 1 - NingDu_obj_func2    # f = 1 - KGE (0~+∞, minimized)
                            ShiCheng_f_KGE_Q = 1 - ShiCheng_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q  = 1 - FenKeng_obj_func2   # f = 1 - KGE (0~+∞, minimized)

                            NingDu_f_R2_Q   = 1 - NingDu_obj_func3     # f = 1 - R2 (0~+∞, minimized)
                            ShiCheng_f_R2_Q = 1 - ShiCheng_obj_func3   # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_R2_Q  = 1 - FenKeng_obj_func3    # f = 1 - R2 (0~+∞, minimized)

                            NingDu_g_NSE_Q   = -NingDu_obj_func1       # -NSE <= 0
                            ShiCheng_g_NSE_Q = -ShiCheng_obj_func1     # -NSE <= 0
                            FenKeng_g_NSE_Q  = -FenKeng_obj_func1      # -NSE <= 0

                            NingDu_g_KGE_Q   = -NingDu_obj_func2       # -KGE <= 0
                            ShiCheng_g_KGE_Q = -ShiCheng_obj_func2     # -KGE <= 0
                            FenKeng_g_KGE_Q  = -FenKeng_obj_func2      # -KGE <= 0

                            NingDu_g_R2_Q   = -NingDu_obj_func3        # -R2 <= 0
                            ShiCheng_g_R2_Q = -ShiCheng_obj_func3      # -R2 <= 0
                            FenKeng_g_R2_Q  = -FenKeng_obj_func3       # -R2 <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, ShiCheng_f_NSE_Q, FenKeng_f_NSE_Q,
                                                        NingDu_f_KGE_Q, ShiCheng_f_KGE_Q, FenKeng_f_KGE_Q,
                                                        NingDu_f_R2_Q, ShiCheng_f_R2_Q, FenKeng_f_R2_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, ShiCheng_g_NSE_Q, FenKeng_g_NSE_Q,
                                                        NingDu_g_KGE_Q, ShiCheng_g_KGE_Q, FenKeng_g_KGE_Q,
                                                        NingDu_g_R2_Q, ShiCheng_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'KGE', 'PBIAS']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q   = 1 - NingDu_obj_func1    # f = 1 - NSE (0~+∞, minimized)
                            ShiCheng_f_NSE_Q = 1 - ShiCheng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q  = 1 - FenKeng_obj_func1   # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_KGE_Q   = 1 - NingDu_obj_func2    # f = 1 - KGE (0~+∞, minimized)
                            ShiCheng_f_KGE_Q = 1 - ShiCheng_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q  = 1 - FenKeng_obj_func2   # f = 1 - KGE (0~+∞, minimized)

                            NingDu_f_PBIAS_Q   = NingDu_obj_func3      # f = PBIAS (0~+∞, minimized)
                            ShiCheng_f_PBIAS_Q = ShiCheng_obj_func3    # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q  = FenKeng_obj_func3     # f = PBIAS (0~+∞, minimized)

                            NingDu_g_NSE_Q   = -NingDu_obj_func1       # -NSE <= 0
                            ShiCheng_g_NSE_Q = -ShiCheng_obj_func1     # -NSE <= 0
                            FenKeng_g_NSE_Q  = -FenKeng_obj_func1      # -NSE <= 0

                            NingDu_g_KGE_Q   = -NingDu_obj_func2       # -KGE <= 0
                            ShiCheng_g_KGE_Q = -ShiCheng_obj_func2     # -KGE <= 0
                            FenKeng_g_KGE_Q  = -FenKeng_obj_func2      # -KGE <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, ShiCheng_f_NSE_Q, FenKeng_f_NSE_Q,
                                                        NingDu_f_KGE_Q, ShiCheng_f_KGE_Q, FenKeng_f_KGE_Q,
                                                        NingDu_f_PBIAS_Q, ShiCheng_f_PBIAS_Q, FenKeng_f_PBIAS_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, ShiCheng_g_NSE_Q, FenKeng_g_NSE_Q,
                                                        NingDu_g_KGE_Q, ShiCheng_g_KGE_Q, FenKeng_g_KGE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'KGE', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q   = 1 - NingDu_obj_func1    # f = 1 - NSE (0~+∞, minimized)
                            ShiCheng_f_NSE_Q = 1 - ShiCheng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q  = 1 - FenKeng_obj_func1   # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_KGE_Q   = 1 - NingDu_obj_func2    # f = 1 - KGE (0~+∞, minimized)
                            ShiCheng_f_KGE_Q = 1 - ShiCheng_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q  = 1 - FenKeng_obj_func2   # f = 1 - KGE (0~+∞, minimized)

                            NingDu_f_RMSE_Q   = NingDu_obj_func3       # f = RMSE (0~+∞, minimized)
                            ShiCheng_f_RMSE_Q = ShiCheng_obj_func3     # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q  = FenKeng_obj_func3      # f = RMSE (0~+∞, minimized)

                            NingDu_g_NSE_Q   = -NingDu_obj_func1       # -NSE <= 0
                            ShiCheng_g_NSE_Q = -ShiCheng_obj_func1     # -NSE <= 0
                            FenKeng_g_NSE_Q  = -FenKeng_obj_func1      # -NSE <= 0

                            NingDu_g_KGE_Q   = -NingDu_obj_func2       # -KGE <= 0
                            ShiCheng_g_KGE_Q = -ShiCheng_obj_func2     # -KGE <= 0
                            FenKeng_g_KGE_Q  = -FenKeng_obj_func2      # -KGE <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, ShiCheng_f_NSE_Q, FenKeng_f_NSE_Q,
                                                        NingDu_f_KGE_Q, ShiCheng_f_KGE_Q, FenKeng_f_KGE_Q,
                                                        NingDu_f_RMSE_Q, ShiCheng_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, ShiCheng_g_NSE_Q, FenKeng_g_NSE_Q,
                                                        NingDu_g_KGE_Q, ShiCheng_g_KGE_Q, FenKeng_g_KGE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'R2', 'PBIAS']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q   = 1 - NingDu_obj_func1    # f = 1 - NSE (0~+∞, minimized)
                            ShiCheng_f_NSE_Q = 1 - ShiCheng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q  = 1 - FenKeng_obj_func1   # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_R2_Q   = 1 - NingDu_obj_func2     # f = 1 - R2 (0~+∞, minimized)
                            ShiCheng_f_R2_Q = 1 - ShiCheng_obj_func2   # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_R2_Q  = 1 - FenKeng_obj_func2    # f = 1 - R2 (0~+∞, minimized)

                            NingDu_f_PBIAS_Q   = NingDu_obj_func3      # f = PBIAS (0~+∞, minimized)
                            ShiCheng_f_PBIAS_Q = ShiCheng_obj_func3    # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q  = FenKeng_obj_func3     # f = PBIAS (0~+∞, minimized)

                            NingDu_g_NSE_Q   = -NingDu_obj_func1       # -NSE <= 0
                            ShiCheng_g_NSE_Q = -ShiCheng_obj_func1     # -NSE <= 0
                            FenKeng_g_NSE_Q  = -FenKeng_obj_func1      # -NSE <= 0

                            NingDu_g_R2_Q   = -NingDu_obj_func2        # -R2 <= 0
                            ShiCheng_g_R2_Q = -ShiCheng_obj_func2      # -R2 <= 0
                            FenKeng_g_R2_Q  = -FenKeng_obj_func2       # -R2 <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, ShiCheng_f_NSE_Q, FenKeng_f_NSE_Q,
                                                        NingDu_f_R2_Q, ShiCheng_f_R2_Q, FenKeng_f_R2_Q,
                                                        NingDu_f_PBIAS_Q, ShiCheng_f_PBIAS_Q, FenKeng_f_PBIAS_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, ShiCheng_g_NSE_Q, FenKeng_g_NSE_Q,
                                                        NingDu_g_R2_Q, ShiCheng_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'R2', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q   = 1 - NingDu_obj_func1    # f = 1 - NSE (0~+∞, minimized)
                            ShiCheng_f_NSE_Q = 1 - ShiCheng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q  = 1 - FenKeng_obj_func1   # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_R2_Q   = 1 - NingDu_obj_func2     # f = 1 - R2 (0~+∞, minimized)
                            ShiCheng_f_R2_Q = 1 - ShiCheng_obj_func2   # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_R2_Q  = 1 - FenKeng_obj_func2    # f = 1 - R2 (0~+∞, minimized)

                            NingDu_f_RMSE_Q   = NingDu_obj_func3       # f = RMSE (0~+∞, minimized)
                            ShiCheng_f_RMSE_Q = ShiCheng_obj_func3     # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q  = FenKeng_obj_func3      # f = RMSE (0~+∞, minimized)

                            NingDu_g_NSE_Q   = -NingDu_obj_func1       # -NSE <= 0
                            ShiCheng_g_NSE_Q = -ShiCheng_obj_func1     # -NSE <= 0
                            FenKeng_g_NSE_Q  = -FenKeng_obj_func1      # -NSE <= 0

                            NingDu_g_R2_Q   = -NingDu_obj_func2        # -R2 <= 0
                            ShiCheng_g_R2_Q = -ShiCheng_obj_func2      # -R2 <= 0
                            FenKeng_g_R2_Q  = -FenKeng_obj_func2       # -R2 <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, ShiCheng_f_NSE_Q, FenKeng_f_NSE_Q,
                                                        NingDu_f_R2_Q, ShiCheng_f_R2_Q, FenKeng_f_R2_Q,
                                                        NingDu_f_RMSE_Q, ShiCheng_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, ShiCheng_g_NSE_Q, FenKeng_g_NSE_Q,
                                                        NingDu_g_R2_Q, ShiCheng_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['NSE', 'PBIAS', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_NSE_Q   = 1 - NingDu_obj_func1    # f = 1 - NSE (0~+∞, minimized)
                            ShiCheng_f_NSE_Q = 1 - ShiCheng_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                            FenKeng_f_NSE_Q  = 1 - FenKeng_obj_func1   # f = 1 - NSE (0~+∞, minimized)

                            NingDu_f_PBIAS_Q   = NingDu_obj_func2      # f = PBIAS (0~+∞, minimized)
                            ShiCheng_f_PBIAS_Q = ShiCheng_obj_func2    # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q  = FenKeng_obj_func2     # f = PBIAS (0~+∞, minimized)

                            NingDu_f_RMSE_Q   = NingDu_obj_func3       # f = RMSE (0~+∞, minimized)
                            ShiCheng_f_RMSE_Q = ShiCheng_obj_func3     # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q  = FenKeng_obj_func3      # f = RMSE (0~+∞, minimized)

                            NingDu_g_NSE_Q   = -NingDu_obj_func1       # -NSE <= 0
                            ShiCheng_g_NSE_Q = -ShiCheng_obj_func1     # -NSE <= 0
                            FenKeng_g_NSE_Q  = -FenKeng_obj_func1      # -NSE <= 0

                            out['F'] = np.column_stack([NingDu_f_NSE_Q, ShiCheng_f_NSE_Q, FenKeng_f_NSE_Q,
                                                        NingDu_f_PBIAS_Q, ShiCheng_f_PBIAS_Q, FenKeng_f_PBIAS_Q,
                                                        NingDu_f_RMSE_Q, ShiCheng_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_NSE_Q, ShiCheng_g_NSE_Q, FenKeng_g_NSE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['KGE', 'R2', 'PBIAS']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_KGE_Q   = 1 - NingDu_obj_func1    # f = 1 - KGE (0~+∞, minimized)
                            ShiCheng_f_KGE_Q = 1 - ShiCheng_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q  = 1 - FenKeng_obj_func1   # f = 1 - KGE (0~+∞, minimized)

                            NingDu_f_R2_Q   = 1 - NingDu_obj_func2     # f = 1 - R2 (0~+∞, minimized)
                            ShiCheng_f_R2_Q = 1 - ShiCheng_obj_func2   # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_R2_Q  = 1 - FenKeng_obj_func2    # f = 1 - R2 (0~+∞, minimized)

                            NingDu_f_PBIAS_Q   = NingDu_obj_func3      # f = PBIAS (0~+∞, minimized)
                            ShiCheng_f_PBIAS_Q = ShiCheng_obj_func3    # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q  = FenKeng_obj_func3     # f = PBIAS (0~+∞, minimized)

                            NingDu_g_KGE_Q   = -NingDu_obj_func1       # -KGE <= 0
                            ShiCheng_g_KGE_Q = -ShiCheng_obj_func1     # -KGE <= 0
                            FenKeng_g_KGE_Q  = -FenKeng_obj_func1      # -KGE <= 0

                            NingDu_g_R2_Q   = -NingDu_obj_func2        # -R2 <= 0
                            ShiCheng_g_R2_Q = -ShiCheng_obj_func2      # -R2 <= 0
                            FenKeng_g_R2_Q  = -FenKeng_obj_func2       # -R2 <= 0

                            out['F'] = np.column_stack([NingDu_f_KGE_Q, ShiCheng_f_KGE_Q, FenKeng_f_KGE_Q,
                                                        NingDu_f_R2_Q, ShiCheng_f_R2_Q, FenKeng_f_R2_Q,
                                                        NingDu_f_PBIAS_Q, ShiCheng_f_PBIAS_Q, FenKeng_f_PBIAS_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_KGE_Q, ShiCheng_g_KGE_Q, FenKeng_g_KGE_Q,
                                                        NingDu_g_R2_Q, ShiCheng_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['KGE', 'R2', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_KGE_Q   = 1 - NingDu_obj_func1    # f = 1 - KGE (0~+∞, minimized)
                            ShiCheng_f_KGE_Q = 1 - ShiCheng_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q  = 1 - FenKeng_obj_func1   # f = 1 - KGE (0~+∞, minimized)

                            NingDu_f_R2_Q   = 1 - NingDu_obj_func2     # f = 1 - R2 (0~+∞, minimized)
                            ShiCheng_f_R2_Q = 1 - ShiCheng_obj_func2   # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_R2_Q  = 1 - FenKeng_obj_func2    # f = 1 - R2 (0~+∞, minimized)

                            NingDu_f_RMSE_Q   = NingDu_obj_func3       # f = RMSE (0~+∞, minimized)
                            ShiCheng_f_RMSE_Q = ShiCheng_obj_func3     # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q  = FenKeng_obj_func3      # f = RMSE (0~+∞, minimized)

                            NingDu_g_KGE_Q   = -NingDu_obj_func1       # -KGE <= 0
                            ShiCheng_g_KGE_Q = -ShiCheng_obj_func1     # -KGE <= 0
                            FenKeng_g_KGE_Q  = -FenKeng_obj_func1      # -KGE <= 0

                            NingDu_g_R2_Q   = -NingDu_obj_func2        # -KGE <= 0
                            ShiCheng_g_R2_Q = -ShiCheng_obj_func2      # -KGE <= 0
                            FenKeng_g_R2_Q  = -FenKeng_obj_func2       # -KGE <= 0

                            out['F'] = np.column_stack([NingDu_f_KGE_Q, ShiCheng_f_KGE_Q, FenKeng_f_KGE_Q,
                                                        NingDu_f_R2_Q, ShiCheng_f_R2_Q, FenKeng_f_R2_Q,
                                                        NingDu_f_RMSE_Q, ShiCheng_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_KGE_Q, ShiCheng_g_KGE_Q, FenKeng_g_KGE_Q,
                                                        NingDu_g_R2_Q, ShiCheng_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['KGE', 'PBIAS', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_KGE_Q   = 1 - NingDu_obj_func1    # f = 1 - KGE (0~+∞, minimized)
                            ShiCheng_f_KGE_Q = 1 - ShiCheng_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                            FenKeng_f_KGE_Q  = 1 - FenKeng_obj_func1   # f = 1 - KGE (0~+∞, minimized)

                            NingDu_f_PBIAS_Q   = NingDu_obj_func2      # f = PBIAS (0~+∞, minimized)
                            ShiCheng_f_PBIAS_Q = ShiCheng_obj_func2    # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q  = FenKeng_obj_func2     # f = PBIAS (0~+∞, minimized)

                            NingDu_f_RMSE_Q   = NingDu_obj_func3       # f = RMSE (0~+∞, minimized)
                            ShiCheng_f_RMSE_Q = ShiCheng_obj_func3     # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q  = FenKeng_obj_func3      # f = RMSE (0~+∞, minimized)

                            NingDu_g_KGE_Q   = -NingDu_obj_func1       # -KGE <= 0
                            ShiCheng_g_KGE_Q = -ShiCheng_obj_func1     # -KGE <= 0
                            FenKeng_g_KGE_Q  = -FenKeng_obj_func1      # -KGE <= 0

                            out['F'] = np.column_stack([NingDu_f_KGE_Q, ShiCheng_f_KGE_Q, FenKeng_f_KGE_Q,
                                                        NingDu_f_PBIAS_Q, ShiCheng_f_PBIAS_Q, FenKeng_f_PBIAS_Q,
                                                        NingDu_f_RMSE_Q, ShiCheng_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_KGE_Q, ShiCheng_g_KGE_Q, FenKeng_g_KGE_Q])  # Constraints

                            print('F:', out['F'].shape)

                        elif self.objectf == ['R2', 'PBIAS', 'RMSE']:
                            # Calculate Objective Functions and Constraints
                            NingDu_f_R2_Q   = 1 - NingDu_obj_func1    # f = 1 - R2 (0~+∞, minimized)
                            ShiCheng_f_R2_Q = 1 - ShiCheng_obj_func1  # f = 1 - R2 (0~+∞, minimized)
                            FenKeng_f_R2_Q  = 1 - FenKeng_obj_func1   # f = 1 - R2 (0~+∞, minimized)

                            NingDu_f_PBIAS_Q   = NingDu_obj_func2     # f = PBIAS (0~+∞, minimized)
                            ShiCheng_f_PBIAS_Q = ShiCheng_obj_func2   # f = PBIAS (0~+∞, minimized)
                            FenKeng_f_PBIAS_Q  = FenKeng_obj_func2    # f = PBIAS (0~+∞, minimized)

                            NingDu_f_RMSE_Q   = NingDu_obj_func3      # f = RMSE (0~+∞, minimized)
                            ShiCheng_f_RMSE_Q = ShiCheng_obj_func3    # f = RMSE (0~+∞, minimized)
                            FenKeng_f_RMSE_Q  = FenKeng_obj_func3     # f = RMSE (0~+∞, minimized)

                            NingDu_g_R2_Q   = -NingDu_obj_func1       # -R2 <= 0
                            ShiCheng_g_R2_Q = -ShiCheng_obj_func1     # -R2 <= 0
                            FenKeng_g_R2_Q  = -FenKeng_obj_func1      # -R2 <= 0

                            out['F'] = np.column_stack([NingDu_f_R2_Q, ShiCheng_f_R2_Q, FenKeng_f_R2_Q,
                                                        NingDu_f_PBIAS_Q, ShiCheng_f_PBIAS_Q, FenKeng_f_PBIAS_Q,
                                                        NingDu_f_RMSE_Q, ShiCheng_f_RMSE_Q, FenKeng_f_RMSE_Q])  # Objective functions
                            out['G'] = np.column_stack([NingDu_g_R2_Q, ShiCheng_g_R2_Q, FenKeng_g_R2_Q])  # Constraints

                            print('F:', out['F'].shape)

        elif self.cal_vars == ['LAI']:
            if self.cali_scheme == 'Single-objective':
                # SWAT Model Simulation
                LAI_pop_obj_func_list = self.swat_execution_run.SWAT_model_execution(x)
                LAI_obj_func = np.array(LAI_pop_obj_func_list)
                if self.objectf[0] in ['NSE', 'KGE', 'R2']:
                    # Calculate Objective Functions and Constraints
                    LAI_f = 1 - LAI_obj_func              # f = 1 - KGE (0~+∞, minimized)

                    LAI_g = -LAI_obj_func                 # -KGE <= 0

                    out['F'] = np.column_stack([LAI_f])   # Objective functions
                    out['G'] = np.column_stack([LAI_g])   # Constraints

                    print('F:', out['F'].shape)

                elif self.objectf[0] in ['RMSE', 'PBIAS']:
                    # Calculate Objective Functions and Constraints
                    LAI_f = LAI_obj_func                  # f = RMSE (0~+∞, minimized)

                    out['F'] = np.column_stack([LAI_f])   # Objective functions

                    print('F:', out['F'].shape)
            elif self.cali_scheme == 'Multi-objective':
                if len(self.objectf) == 2:
                    # SWAT Model Simulation
                    LAI_pop_obj_func1_list, LAI_pop_obj_func2_list = self.swat_execution_run.SWAT_model_execution(x)
                    LAI_obj_func1 = np.array(LAI_pop_obj_func1_list)
                    LAI_obj_func2 = np.array(LAI_pop_obj_func2_list)

                    if self.objectf == ['NSE', 'KGE']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_NSE = 1 - LAI_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        LAI_f_KGE = 1 - LAI_obj_func2  # f = 1 - KGE (0~+∞, minimized)

                        LAI_g_NSE = -LAI_obj_func1  # -NSE <= 0
                        LAI_g_KGE = -LAI_obj_func2  # -KGE <= 0

                        out['F'] = np.column_stack([LAI_f_NSE, LAI_f_KGE])  # Objective functions
                        out['G'] = np.column_stack([LAI_g_NSE, LAI_g_KGE])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'R2']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_NSE = 1 - LAI_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        LAI_f_R2  = 1 - LAI_obj_func2  # f = 1 - R2 (0~+∞, minimized)

                        LAI_g_NSE = -LAI_obj_func1  # -NSE <= 0
                        LAI_g_R2  = -LAI_obj_func2  # -R2 <= 0

                        out['F'] = np.column_stack([LAI_f_NSE, LAI_f_R2])  # Objective functions
                        out['G'] = np.column_stack([LAI_g_NSE, LAI_g_R2])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_NSE   = 1 - LAI_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        LAI_f_PBIAS = LAI_obj_func2      # f = PBIAS (0~+∞, minimized)

                        LAI_g_NSE = -LAI_obj_func1  # -NSE <= 0

                        out['F'] = np.column_stack([LAI_f_NSE, LAI_f_PBIAS])  # Objective functions
                        out['G'] = np.column_stack([LAI_g_NSE])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_NSE  = 1 - LAI_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        LAI_f_RMSE = LAI_obj_func2  # f = RMSE (0~+∞, minimized)

                        LAI_g_NSE = -LAI_obj_func1  # -NSE <= 0

                        out['F'] = np.column_stack([LAI_f_NSE, LAI_f_RMSE])  # Objective functions
                        out['G'] = np.column_stack([LAI_g_NSE])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['KGE', 'R2']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_KGE = 1 - LAI_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                        LAI_f_R2  = 1 - LAI_obj_func2  # f = 1 - R2 (0~+∞, minimized)

                        LAI_g_KGE = -LAI_obj_func1  # -KGE <= 0
                        LAI_g_R2  = -LAI_obj_func2  # -R2 <= 0

                        out['F'] = np.column_stack([LAI_f_KGE, LAI_f_R2])  # Objective functions
                        out['G'] = np.column_stack([LAI_g_KGE, LAI_g_R2])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['KGE', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_KGE   = 1 - LAI_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                        LAI_f_PBIAS = LAI_obj_func2  # f = PBIAS (0~+∞, minimized)

                        LAI_g_KGE = -LAI_obj_func1  # -KGE <= 0

                        out['F'] = np.column_stack([LAI_f_KGE, LAI_f_PBIAS])  # Objective functions
                        out['G'] = np.column_stack([LAI_g_KGE])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['KGE', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_KGE  = 1 - LAI_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                        LAI_f_RMSE = LAI_obj_func2  # f = RMSE (0~+∞, minimized)

                        LAI_g_KGE = -LAI_obj_func1  # -KGE <= 0

                        out['F'] = np.column_stack([LAI_f_KGE, LAI_f_RMSE])  # Objective functions
                        out['G'] = np.column_stack([LAI_g_KGE])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['R2', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_R2    = 1 - LAI_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        LAI_f_PBIAS = LAI_obj_func2  # f = PBIAS (0~+∞, minimized)

                        LAI_g_R2 = -LAI_obj_func1  # -R2 <= 0

                        out['F'] = np.column_stack([LAI_f_R2, LAI_f_PBIAS])  # Objective functions
                        out['G'] = np.column_stack([LAI_g_R2])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['R2', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_R2   = 1 - LAI_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        LAI_f_RMSE = LAI_obj_func2  # f = RMSE (0~+∞, minimized)

                        LAI_g_R2 = -LAI_obj_func1  # -R2 <= 0

                        out['F'] = np.column_stack([LAI_f_R2, LAI_f_RMSE])  # Objective functions
                        out['G'] = np.column_stack([LAI_g_R2])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['PBIAS', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_PBIAS = LAI_obj_func1  # f = PBIAS (0~+∞, minimized)
                        LAI_f_RMSE  = LAI_obj_func2  # f = RMSE (0~+∞, minimized)

                        out['F'] = np.column_stack([LAI_f_PBIAS, LAI_f_RMSE])  # Objective functions

                        print('F:', out['F'].shape)

                elif len(self.objectf) == 3:
                    # SWAT Model Simulation
                    LAI_pop_obj_func1_list, LAI_pop_obj_func2_list, LAI_pop_obj_func3_list = self.swat_execution_run.SWAT_model_execution(x)
                    LAI_obj_func1 = np.array(LAI_pop_obj_func1_list)
                    LAI_obj_func2 = np.array(LAI_pop_obj_func2_list)
                    LAI_obj_func3 = np.array(LAI_pop_obj_func3_list)

                    if self.objectf == ['NSE', 'KGE', 'R2']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_NSE = 1 - LAI_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        LAI_f_KGE = 1 - LAI_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                        LAI_f_R2  = 1 - LAI_obj_func3  # f = 1 - R2 (0~+∞, minimized)

                        LAI_g_NSE = -LAI_obj_func1  # -NSE <= 0
                        LAI_g_KGE = -LAI_obj_func2  # -KGE <= 0
                        LAI_g_R2  = -LAI_obj_func3  # -R2 <= 0

                        out['F'] = np.column_stack([LAI_f_NSE, LAI_f_KGE, LAI_f_R2])  # Objective functions
                        out['G'] = np.column_stack([LAI_g_NSE, LAI_g_KGE, LAI_g_R2])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'KGE', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_NSE   = 1 - LAI_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        LAI_f_KGE   = 1 - LAI_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                        LAI_f_PBIAS = LAI_obj_func3      # f = PBIAS (0~+∞, minimized)

                        LAI_g_NSE = -LAI_obj_func1  # -NSE <= 0
                        LAI_g_KGE = -LAI_obj_func2  # -KGE <= 0

                        out['F'] = np.column_stack([LAI_f_NSE, LAI_f_KGE, LAI_f_PBIAS])  # Objective functions
                        out['G'] = np.column_stack([LAI_g_NSE, LAI_g_KGE])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'KGE', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_NSE  = 1 - LAI_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        LAI_f_KGE  = 1 - LAI_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                        LAI_f_RMSE = LAI_obj_func3      # f = RMSE (0~+∞, minimized)

                        LAI_g_NSE = -LAI_obj_func1  # -NSE <= 0
                        LAI_g_KGE = -LAI_obj_func2  # -KGE <= 0

                        out['F'] = np.column_stack([LAI_f_NSE, LAI_f_KGE, LAI_f_RMSE])  # Objective functions
                        out['G'] = np.column_stack([LAI_g_NSE, LAI_g_KGE])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'R2', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_NSE   = 1 - LAI_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        LAI_f_R2    = 1 - LAI_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                        LAI_f_PBIAS = LAI_obj_func3      # f = PBIAS (0~+∞, minimized)

                        LAI_g_NSE = -LAI_obj_func1  # -NSE <= 0
                        LAI_g_R2  = -LAI_obj_func2  # -R2 <= 0

                        out['F'] = np.column_stack([LAI_f_NSE, LAI_f_R2, LAI_f_PBIAS])  # Objective functions
                        out['G'] = np.column_stack([LAI_g_NSE, LAI_g_R2])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'R2', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_NSE  = 1 - LAI_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        LAI_f_R2   = 1 - LAI_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                        LAI_f_RMSE = LAI_obj_func3      # f = RMSE (0~+∞, minimized)

                        LAI_g_NSE = -LAI_obj_func1  # -NSE <= 0
                        LAI_g_R2  = -LAI_obj_func2  # -R2 <= 0

                        out['F'] = np.column_stack([LAI_f_NSE, LAI_f_R2, LAI_f_RMSE])  # Objective functions
                        out['G'] = np.column_stack([LAI_g_NSE, LAI_g_R2])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'PBIAS', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_NSE   = 1 - LAI_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        LAI_f_PBIAS = LAI_obj_func2      # f = PBIAS (0~+∞, minimized)
                        LAI_f_RMSE  = LAI_obj_func3      # f = RMSE (0~+∞, minimized)

                        LAI_g_NSE = -LAI_obj_func1  # -NSE <= 0

                        out['F'] = np.column_stack([LAI_f_NSE, LAI_f_PBIAS, LAI_f_RMSE])  # Objective functions
                        out['G'] = np.column_stack([LAI_g_NSE])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['KGE', 'R2', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_KGE   = 1 - LAI_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                        LAI_f_R2    = 1 - LAI_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                        LAI_f_PBIAS = LAI_obj_func3      # f = PBIAS (0~+∞, minimized)

                        LAI_g_KGE = -LAI_obj_func1  # -KGE <= 0
                        LAI_g_R2  = -LAI_obj_func2  # -R2 <= 0

                        out['F'] = np.column_stack([LAI_f_KGE, LAI_f_R2, LAI_f_PBIAS])  # Objective functions
                        out['G'] = np.column_stack([LAI_g_KGE, LAI_g_R2])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['KGE', 'R2', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_KGE  = 1 - LAI_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                        LAI_f_R2   = 1 - LAI_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                        LAI_f_RMSE = LAI_obj_func3      # f = RMSE (0~+∞, minimized)

                        LAI_g_KGE = -LAI_obj_func1  # -KGE <= 0
                        LAI_g_R2  = -LAI_obj_func2  # -KGE <= 0

                        out['F'] = np.column_stack([LAI_f_KGE, LAI_f_R2, LAI_f_RMSE])  # Objective functions
                        out['G'] = np.column_stack([LAI_g_KGE, LAI_g_R2])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['KGE', 'PBIAS', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_KGE   = 1 - LAI_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                        LAI_f_PBIAS = LAI_obj_func2      # f = PBIAS (0~+∞, minimized)
                        LAI_f_RMSE  = LAI_obj_func3      # f = RMSE (0~+∞, minimized)

                        LAI_g_KGE = -LAI_obj_func1  # -KGE <= 0

                        out['F'] = np.column_stack([LAI_f_KGE, LAI_f_PBIAS, LAI_f_RMSE])  # Objective functions
                        out['G'] = np.column_stack([LAI_g_KGE])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['R2', 'PBIAS', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        LAI_f_R2    = 1 - LAI_obj_func1  # f = 1 - R2 (0~+∞, minimized)
                        LAI_f_PBIAS = LAI_obj_func2      # f = PBIAS (0~+∞, minimized)
                        LAI_f_RMSE  = LAI_obj_func3      # f = RMSE (0~+∞, minimized)

                        LAI_g_R2 = -LAI_obj_func1  # -R2 <= 0

                        out['F'] = np.column_stack([LAI_f_R2, LAI_f_PBIAS, LAI_f_RMSE])  # Objective functions
                        out['G'] = np.column_stack([LAI_g_R2])  # Constraints

                        print('F:', out['F'].shape)

        elif self.cal_vars == ['BIOM']:
            if self.cali_scheme == 'Single-objective':
                # SWAT Model Simulation
                BIOM_pop_obj_func_list = self.swat_execution_run.SWAT_model_execution(x)
                BIOM_obj_func = np.array(BIOM_pop_obj_func_list)
                if self.objectf[0] in ['NSE', 'KGE', 'R2']:
                    # Calculate Objective Functions and Constraints
                    BIOM_f = 1 - BIOM_obj_func  # f = 1 - KGE (0~+∞, minimized)

                    out['F'] = np.column_stack([BIOM_f])  # Objective functions

                    print('F:', out['F'].shape)

                elif self.objectf[0] in ['RMSE', 'PBIAS']:
                    # Calculate Objective Functions and Constraints
                    BIOM_f = BIOM_obj_func  # f = RMSE (0~+∞, minimized)

                    out['F'] = np.column_stack([BIOM_f])  # Objective functions

                    print('F:', out['F'].shape)
            elif self.cali_scheme == 'Multi-objective':
                if len(self.objectf) == 2:
                    # SWAT Model Simulation
                    BIOM_pop_obj_func1_list, BIOM_pop_obj_func2_list = self.swat_execution_run.SWAT_model_execution(x)
                    BIOM_obj_func1 = np.array(BIOM_pop_obj_func1_list)
                    BIOM_obj_func2 = np.array(BIOM_pop_obj_func2_list)

                    if self.objectf == ['NSE', 'KGE']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_NSE = 1 - BIOM_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        BIOM_f_KGE = 1 - BIOM_obj_func2  # f = 1 - KGE (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_NSE, BIOM_f_KGE])  # Objective functions

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'R2']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_NSE = 1 - BIOM_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        BIOM_f_R2  = 1 - BIOM_obj_func2  # f = 1 - R2 (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_NSE, BIOM_f_R2])  # Objective functions

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_NSE   = 1 - BIOM_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        BIOM_f_PBIAS = BIOM_obj_func2      # f = PBIAS (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_NSE, BIOM_f_PBIAS])  # Objective functions

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_NSE  = 1 - BIOM_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        BIOM_f_RMSE = BIOM_obj_func2  # f = RMSE (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_NSE, BIOM_f_RMSE])  # Objective functions

                        print('F:', out['F'].shape)

                    elif self.objectf == ['KGE', 'R2']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_KGE = 1 - BIOM_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                        BIOM_f_R2  = 1 - BIOM_obj_func2  # f = 1 - R2 (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_KGE, BIOM_f_R2])  # Objective functions

                        print('F:', out['F'].shape)

                    elif self.objectf == ['KGE', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_KGE   = 1 - BIOM_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                        BIOM_f_PBIAS = BIOM_obj_func2  # f = PBIAS (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_KGE, BIOM_f_PBIAS])  # Objective functions

                        print('F:', out['F'].shape)

                    elif self.objectf == ['KGE', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_KGE  = 1 - BIOM_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                        BIOM_f_RMSE = BIOM_obj_func2  # f = RMSE (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_KGE, BIOM_f_RMSE])  # Objective functions

                        print('F:', out['F'].shape)

                    elif self.objectf == ['R2', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_R2    = 1 - BIOM_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        BIOM_f_PBIAS = BIOM_obj_func2  # f = PBIAS (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_R2, BIOM_f_PBIAS])  # Objective functions

                        print('F:', out['F'].shape)

                    elif self.objectf == ['R2', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_R2   = 1 - BIOM_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        BIOM_f_RMSE = BIOM_obj_func2  # f = RMSE (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_R2, BIOM_f_RMSE])  # Objective functions

                        print('F:', out['F'].shape)

                    elif self.objectf == ['PBIAS', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_PBIAS = BIOM_obj_func1  # f = PBIAS (0~+∞, minimized)
                        BIOM_f_RMSE  = BIOM_obj_func2  # f = RMSE (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_PBIAS, BIOM_f_RMSE])  # Objective functions

                        print('F:', out['F'].shape)

                elif len(self.objectf) == 3:
                    # SWAT Model Simulation
                    BIOM_pop_obj_func1_list, BIOM_pop_obj_func2_list, BIOM_pop_obj_func3_list = self.swat_execution_run.SWAT_model_execution(x)
                    BIOM_obj_func1 = np.array(BIOM_pop_obj_func1_list)
                    BIOM_obj_func2 = np.array(BIOM_pop_obj_func2_list)
                    BIOM_obj_func3 = np.array(BIOM_pop_obj_func3_list)

                    if self.objectf == ['NSE', 'KGE', 'R2']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_NSE = 1 - BIOM_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        BIOM_f_KGE = 1 - BIOM_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                        BIOM_f_R2  = 1 - BIOM_obj_func3  # f = 1 - R2 (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_NSE, BIOM_f_KGE, BIOM_f_R2])  # Objective functions

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'KGE', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_NSE   = 1 - BIOM_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        BIOM_f_KGE   = 1 - BIOM_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                        BIOM_f_PBIAS = BIOM_obj_func3      # f = PBIAS (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_NSE, BIOM_f_KGE, BIOM_f_PBIAS])  # Objective functions

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'KGE', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_NSE  = 1 - BIOM_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        BIOM_f_KGE  = 1 - BIOM_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                        BIOM_f_RMSE = BIOM_obj_func3      # f = RMSE (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_NSE, BIOM_f_KGE, BIOM_f_RMSE])  # Objective functions

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'R2', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_NSE   = 1 - BIOM_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        BIOM_f_R2    = 1 - BIOM_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                        BIOM_f_PBIAS = BIOM_obj_func3      # f = PBIAS (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_NSE, BIOM_f_R2, BIOM_f_PBIAS])  # Objective functions

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'R2', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_NSE  = 1 - BIOM_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        BIOM_f_R2   = 1 - BIOM_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                        BIOM_f_RMSE = BIOM_obj_func3      # f = RMSE (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_NSE, BIOM_f_R2, BIOM_f_RMSE])  # Objective functions

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'PBIAS', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_NSE   = 1 - BIOM_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        BIOM_f_PBIAS = BIOM_obj_func2      # f = PBIAS (0~+∞, minimized)
                        BIOM_f_RMSE  = BIOM_obj_func3      # f = RMSE (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_NSE, BIOM_f_PBIAS, BIOM_f_RMSE])  # Objective functions

                        print('F:', out['F'].shape)

                    elif self.objectf == ['KGE', 'R2', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_KGE   = 1 - BIOM_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                        BIOM_f_R2    = 1 - BIOM_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                        BIOM_f_PBIAS = BIOM_obj_func3      # f = PBIAS (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_KGE, BIOM_f_R2, BIOM_f_PBIAS])  # Objective functions

                        print('F:', out['F'].shape)

                    elif self.objectf == ['KGE', 'R2', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_KGE  = 1 - BIOM_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                        BIOM_f_R2   = 1 - BIOM_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                        BIOM_f_RMSE = BIOM_obj_func3      # f = RMSE (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_KGE, BIOM_f_R2, BIOM_f_RMSE])  # Objective functions

                        print('F:', out['F'].shape)

                    elif self.objectf == ['KGE', 'PBIAS', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_KGE   = 1 - BIOM_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                        BIOM_f_PBIAS = BIOM_obj_func2      # f = PBIAS (0~+∞, minimized)
                        BIOM_f_RMSE  = BIOM_obj_func3      # f = RMSE (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_KGE, BIOM_f_PBIAS, BIOM_f_RMSE])  # Objective functions

                        print('F:', out['F'].shape)

                    elif self.objectf == ['R2', 'PBIAS', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        BIOM_f_R2    = 1 - BIOM_obj_func1  # f = 1 - R2 (0~+∞, minimized)
                        BIOM_f_PBIAS = BIOM_obj_func2      # f = PBIAS (0~+∞, minimized)
                        BIOM_f_RMSE  = BIOM_obj_func3      # f = RMSE (0~+∞, minimized)

                        out['F'] = np.column_stack([BIOM_f_R2, BIOM_f_PBIAS, BIOM_f_RMSE])  # Objective functions

                        print('F:', out['F'].shape)

        elif self.cal_vars == ['ET']:
            if self.cali_scheme == 'Single-objective':
                # SWAT Model Simulation
                ET_pop_obj_func_list = self.swat_execution_run.SWAT_model_execution(x)
                ET_obj_func = np.array(ET_pop_obj_func_list)
                if self.objectf[0] in ['NSE', 'KGE', 'R2']:
                    # Calculate Objective Functions and Constraints
                    ET_f = 1 - ET_obj_func              # f = 1 - KGE (0~+∞, minimized)

                    ET_g = -ET_obj_func                 # -KGE <= 0

                    out['F'] = np.column_stack([ET_f])   # Objective functions
                    out['G'] = np.column_stack([ET_g])   # Constraints

                    print('F:', out['F'].shape)

                elif self.objectf[0] in ['RMSE', 'PBIAS']:
                    # Calculate Objective Functions and Constraints
                    ET_f = ET_obj_func                  # f = RMSE (0~+∞, minimized)

                    out['F'] = np.column_stack([ET_f])   # Objective functions

                    print('F:', out['F'].shape)
            elif self.cali_scheme == 'Multi-objective':
                if len(self.objectf) == 2:
                    # SWAT Model Simulation
                    ET_pop_obj_func1_list, ET_pop_obj_func2_list = self.swat_execution_run.SWAT_model_execution(x)
                    ET_obj_func1 = np.array(ET_pop_obj_func1_list)
                    ET_obj_func2 = np.array(ET_pop_obj_func2_list)

                    if self.objectf == ['NSE', 'KGE']:
                        # Calculate Objective Functions and Constraints
                        ET_f_NSE = 1 - ET_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        ET_f_KGE = 1 - ET_obj_func2  # f = 1 - KGE (0~+∞, minimized)

                        ET_g_NSE = -ET_obj_func1  # -NSE <= 0
                        ET_g_KGE = -ET_obj_func2  # -KGE <= 0

                        out['F'] = np.column_stack([ET_f_NSE, ET_f_KGE])  # Objective functions
                        out['G'] = np.column_stack([ET_g_NSE, ET_g_KGE])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'R2']:
                        # Calculate Objective Functions and Constraints
                        ET_f_NSE = 1 - ET_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        ET_f_R2  = 1 - ET_obj_func2  # f = 1 - R2 (0~+∞, minimized)

                        ET_g_NSE = -ET_obj_func1  # -NSE <= 0
                        ET_g_R2  = -ET_obj_func2  # -R2 <= 0

                        out['F'] = np.column_stack([ET_f_NSE, ET_f_R2])  # Objective functions
                        out['G'] = np.column_stack([ET_g_NSE, ET_g_R2])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        ET_f_NSE   = 1 - ET_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        ET_f_PBIAS = ET_obj_func2      # f = PBIAS (0~+∞, minimized)

                        ET_g_NSE = -ET_obj_func1  # -NSE <= 0

                        out['F'] = np.column_stack([ET_f_NSE, ET_f_PBIAS])  # Objective functions
                        out['G'] = np.column_stack([ET_g_NSE])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        ET_f_NSE  = 1 - ET_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        ET_f_RMSE = ET_obj_func2  # f = RMSE (0~+∞, minimized)

                        ET_g_NSE = -ET_obj_func1  # -NSE <= 0

                        out['F'] = np.column_stack([ET_f_NSE, ET_f_RMSE])  # Objective functions
                        out['G'] = np.column_stack([ET_g_NSE])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['KGE', 'R2']:
                        # Calculate Objective Functions and Constraints
                        ET_f_KGE = 1 - ET_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                        ET_f_R2  = 1 - ET_obj_func2  # f = 1 - R2 (0~+∞, minimized)

                        ET_g_KGE = -ET_obj_func1  # -KGE <= 0
                        ET_g_R2  = -ET_obj_func2  # -R2 <= 0

                        out['F'] = np.column_stack([ET_f_KGE, ET_f_R2])  # Objective functions
                        out['G'] = np.column_stack([ET_g_KGE, ET_g_R2])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['KGE', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        ET_f_KGE   = 1 - ET_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                        ET_f_PBIAS = ET_obj_func2  # f = PBIAS (0~+∞, minimized)

                        ET_g_KGE = -ET_obj_func1  # -KGE <= 0

                        out['F'] = np.column_stack([ET_f_KGE, ET_f_PBIAS])  # Objective functions
                        out['G'] = np.column_stack([ET_g_KGE])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['KGE', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        ET_f_KGE  = 1 - ET_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                        ET_f_RMSE = ET_obj_func2  # f = RMSE (0~+∞, minimized)

                        ET_g_KGE = -ET_obj_func1  # -KGE <= 0

                        out['F'] = np.column_stack([ET_f_KGE, ET_f_RMSE])  # Objective functions
                        out['G'] = np.column_stack([ET_g_KGE])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['R2', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        ET_f_R2    = 1 - ET_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        ET_f_PBIAS = ET_obj_func2  # f = PBIAS (0~+∞, minimized)

                        ET_g_R2 = -ET_obj_func1  # -R2 <= 0

                        out['F'] = np.column_stack([ET_f_R2, ET_f_PBIAS])  # Objective functions
                        out['G'] = np.column_stack([ET_g_R2])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['R2', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        ET_f_R2   = 1 - ET_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        ET_f_RMSE = ET_obj_func2  # f = RMSE (0~+∞, minimized)

                        ET_g_R2 = -ET_obj_func1  # -R2 <= 0

                        out['F'] = np.column_stack([ET_f_R2, ET_f_RMSE])  # Objective functions
                        out['G'] = np.column_stack([ET_g_R2])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['PBIAS', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        ET_f_PBIAS = ET_obj_func1  # f = PBIAS (0~+∞, minimized)
                        ET_f_RMSE  = ET_obj_func2  # f = RMSE (0~+∞, minimized)

                        out['F'] = np.column_stack([ET_f_PBIAS, ET_f_RMSE])  # Objective functions

                        print('F:', out['F'].shape)

                elif len(self.objectf) == 3:
                    # SWAT Model Simulation
                    ET_pop_obj_func1_list, ET_pop_obj_func2_list, ET_pop_obj_func3_list = self.swat_execution_run.SWAT_model_execution(x)
                    ET_obj_func1 = np.array(ET_pop_obj_func1_list)
                    ET_obj_func2 = np.array(ET_pop_obj_func2_list)
                    ET_obj_func3 = np.array(ET_pop_obj_func3_list)

                    if self.objectf == ['NSE', 'KGE', 'R2']:
                        # Calculate Objective Functions and Constraints
                        ET_f_NSE = 1 - ET_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        ET_f_KGE = 1 - ET_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                        ET_f_R2  = 1 - ET_obj_func3  # f = 1 - R2 (0~+∞, minimized)

                        ET_g_NSE = -ET_obj_func1  # -NSE <= 0
                        ET_g_KGE = -ET_obj_func2  # -KGE <= 0
                        ET_g_R2  = -ET_obj_func3  # -R2 <= 0

                        out['F'] = np.column_stack([ET_f_NSE, ET_f_KGE, ET_f_R2])  # Objective functions
                        out['G'] = np.column_stack([ET_g_NSE, ET_g_KGE, ET_g_R2])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'KGE', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        ET_f_NSE   = 1 - ET_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        ET_f_KGE   = 1 - ET_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                        ET_f_PBIAS = ET_obj_func3      # f = PBIAS (0~+∞, minimized)

                        ET_g_NSE = -ET_obj_func1  # -NSE <= 0
                        ET_g_KGE = -ET_obj_func2  # -KGE <= 0

                        out['F'] = np.column_stack([ET_f_NSE, ET_f_KGE, ET_f_PBIAS])  # Objective functions
                        out['G'] = np.column_stack([ET_g_NSE, ET_g_KGE])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'KGE', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        ET_f_NSE  = 1 - ET_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        ET_f_KGE  = 1 - ET_obj_func2  # f = 1 - KGE (0~+∞, minimized)
                        ET_f_RMSE = ET_obj_func3      # f = RMSE (0~+∞, minimized)

                        ET_g_NSE = -ET_obj_func1  # -NSE <= 0
                        ET_g_KGE = -ET_obj_func2  # -KGE <= 0

                        out['F'] = np.column_stack([ET_f_NSE, ET_f_KGE, ET_f_RMSE])  # Objective functions
                        out['G'] = np.column_stack([ET_g_NSE, ET_g_KGE])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'R2', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        ET_f_NSE   = 1 - ET_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        ET_f_R2    = 1 - ET_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                        ET_f_PBIAS = ET_obj_func3      # f = PBIAS (0~+∞, minimized)

                        ET_g_NSE = -ET_obj_func1  # -NSE <= 0
                        ET_g_R2  = -ET_obj_func2  # -R2 <= 0

                        out['F'] = np.column_stack([ET_f_NSE, ET_f_R2, ET_f_PBIAS])  # Objective functions
                        out['G'] = np.column_stack([ET_g_NSE, ET_g_R2])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'R2', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        ET_f_NSE  = 1 - ET_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        ET_f_R2   = 1 - ET_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                        ET_f_RMSE = ET_obj_func3      # f = RMSE (0~+∞, minimized)

                        ET_g_NSE = -ET_obj_func1  # -NSE <= 0
                        ET_g_R2  = -ET_obj_func2  # -R2 <= 0

                        out['F'] = np.column_stack([ET_f_NSE, ET_f_R2, ET_f_RMSE])  # Objective functions
                        out['G'] = np.column_stack([ET_g_NSE, ET_g_R2])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['NSE', 'PBIAS', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        ET_f_NSE   = 1 - ET_obj_func1  # f = 1 - NSE (0~+∞, minimized)
                        ET_f_PBIAS = ET_obj_func2      # f = PBIAS (0~+∞, minimized)
                        ET_f_RMSE  = ET_obj_func3      # f = RMSE (0~+∞, minimized)

                        ET_g_NSE = -ET_obj_func1  # -NSE <= 0

                        out['F'] = np.column_stack([ET_f_NSE, ET_f_PBIAS, ET_f_RMSE])  # Objective functions
                        out['G'] = np.column_stack([ET_g_NSE])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['KGE', 'R2', 'PBIAS']:
                        # Calculate Objective Functions and Constraints
                        ET_f_KGE   = 1 - ET_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                        ET_f_R2    = 1 - ET_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                        ET_f_PBIAS = ET_obj_func3      # f = PBIAS (0~+∞, minimized)

                        ET_g_KGE = -ET_obj_func1  # -KGE <= 0
                        ET_g_R2  = -ET_obj_func2  # -R2 <= 0

                        out['F'] = np.column_stack([ET_f_KGE, ET_f_R2, ET_f_PBIAS])  # Objective functions
                        out['G'] = np.column_stack([ET_g_KGE, ET_g_R2])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['KGE', 'R2', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        ET_f_KGE  = 1 - ET_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                        ET_f_R2   = 1 - ET_obj_func2  # f = 1 - R2 (0~+∞, minimized)
                        ET_f_RMSE = ET_obj_func3      # f = RMSE (0~+∞, minimized)

                        ET_g_KGE = -ET_obj_func1  # -KGE <= 0
                        ET_g_R2  = -ET_obj_func2  # -KGE <= 0

                        out['F'] = np.column_stack([ET_f_KGE, ET_f_R2, ET_f_RMSE])  # Objective functions
                        out['G'] = np.column_stack([ET_g_KGE, ET_g_R2])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['KGE', 'PBIAS', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        ET_f_KGE   = 1 - ET_obj_func1  # f = 1 - KGE (0~+∞, minimized)
                        ET_f_PBIAS = ET_obj_func2      # f = PBIAS (0~+∞, minimized)
                        ET_f_RMSE  = ET_obj_func3      # f = RMSE (0~+∞, minimized)

                        ET_g_KGE = -ET_obj_func1  # -KGE <= 0

                        out['F'] = np.column_stack([ET_f_KGE, ET_f_PBIAS, ET_f_RMSE])  # Objective functions
                        out['G'] = np.column_stack([ET_g_KGE])  # Constraints

                        print('F:', out['F'].shape)

                    elif self.objectf == ['R2', 'PBIAS', 'RMSE']:
                        # Calculate Objective Functions and Constraints
                        ET_f_R2    = 1 - ET_obj_func1  # f = 1 - R2 (0~+∞, minimized)
                        ET_f_PBIAS = ET_obj_func2      # f = PBIAS (0~+∞, minimized)
                        ET_f_RMSE  = ET_obj_func3      # f = RMSE (0~+∞, minimized)

                        ET_g_R2 = -ET_obj_func1  # -R2 <= 0

                        out['F'] = np.column_stack([ET_f_R2, ET_f_PBIAS, ET_f_RMSE])  # Objective functions
                        out['G'] = np.column_stack([ET_g_R2])  # Constraints

                        print('F:', out['F'].shape)

        # Two Variables
        elif self.cal_vars == ['Streamflow', 'ET']:
            if len(self.hydro_stas) == 1:
                # SWAT Model Simulation
                rch_pop_obj_func_list, ET_pop_obj_func_list = self.swat_execution_run.SWAT_model_execution(x)
                FenKeng_obj_func = np.array(rch_pop_obj_func_list[0])
                ET_obj_func      = np.array(ET_pop_obj_func_list)

                if self.objectf[0] in ['NSE', 'KGE', 'R2']:
                    # Calculate Objective Functions and Constraints
                    FenKeng_f_Q = 1 - FenKeng_obj_func  # f = 1 - KGE (0~+∞, minimized)
                    ET_f        = 1 - ET_obj_func       # f = 1 - KGE (0~+∞, minimized)

                    FenKeng_g_Q = -FenKeng_obj_func  # -KGE <= 0
                    ET_g        = -ET_obj_func       # -KGE <= 0

                    out['F'] = np.column_stack([FenKeng_f_Q, ET_f])  # Objective functions
                    out['G'] = np.column_stack([FenKeng_g_Q, ET_g])  # Constraints

                    print('F:', out['F'].shape)
                elif self.objectf[0] in ['RMSE', 'PBIAS']:
                    # Calculate Objective Functions and Constraints
                    FenKeng_f_Q = FenKeng_obj_func  # f = RMSE (0~+∞, minimized)
                    ET_f        = ET_obj_func       # f = RMSE (0~+∞, minimized)

                    out['F'] = np.column_stack([FenKeng_f_Q, ET_f])  # Objective functions

                    print('F:', out['F'].shape)
            elif len(self.hydro_stas) == 2:
                # SWAT Model Simulation
                rch_pop_obj_func_list, ET_pop_obj_func_list = self.swat_execution_run.SWAT_model_execution(x)
                NingDu_obj_func  = np.array(rch_pop_obj_func_list[0])
                FenKeng_obj_func = np.array(rch_pop_obj_func_list[1])
                ET_obj_func      = np.array(ET_pop_obj_func_list)
                if self.objectf[0] in ['NSE', 'KGE', 'R2']:
                    # Calculate Objective Functions and Constraints
                    NingDu_f_Q  = 1 - NingDu_obj_func   # f = 1 - KGE (0~+∞, minimized)
                    FenKeng_f_Q = 1 - FenKeng_obj_func  # f = 1 - KGE (0~+∞, minimized)
                    ET_f        = 1 - ET_obj_func       # f = 1 - KGE (0~+∞, minimized)

                    NingDu_g_Q  = -NingDu_obj_func   # -KGE <= 0
                    FenKeng_g_Q = -FenKeng_obj_func  # -KGE <= 0
                    ET_g        = -ET_obj_func       # -KGE <= 0

                    out['F'] = np.column_stack([NingDu_f_Q, FenKeng_f_Q, ET_f])  # Objective functions
                    out['G'] = np.column_stack([NingDu_g_Q, FenKeng_g_Q, ET_g])  # Constraints

                    print('F:', out['F'].shape)

                elif self.objectf[0] in ['RMSE', 'PBIAS']:
                    # Calculate Objective Functions and Constraints
                    NingDu_f_Q  = NingDu_obj_func   # f = RMSE (0~+∞, minimized)
                    FenKeng_f_Q = FenKeng_obj_func  # f = RMSE (0~+∞, minimized)
                    ET_f        = ET_obj_func       # f = RMSE (0~+∞, minimized)

                    out['F'] = np.column_stack([NingDu_f_Q, FenKeng_f_Q, ET_f])  # Objective functions

                    print('F:', out['F'].shape)
            elif len(self.hydro_stas) == 3:
                # SWAT Model Simulation
                rch_pop_obj_func_list, ET_pop_obj_func_list = self.swat_execution_run.SWAT_model_execution(x)
                NingDu_obj_func   = np.array(rch_pop_obj_func_list[0])
                ShiCheng_obj_func = np.array(rch_pop_obj_func_list[1])
                FenKeng_obj_func  = np.array(rch_pop_obj_func_list[2])
                ET_obj_func       = np.array(ET_pop_obj_func_list)

                if self.objectf[0] in ['NSE', 'KGE', 'R2']:
                    # Calculate Objective Functions and Constraints
                    NingDu_f_Q   = 1 - NingDu_obj_func    # f = 1 - KGE (0~+∞, minimized)
                    ShiCheng_f_Q = 1 - ShiCheng_obj_func  # f = 1 - KGE (0~+∞, minimized)
                    FenKeng_f_Q  = 1 - FenKeng_obj_func   # f = 1 - KGE (0~+∞, minimized)
                    ET_f         = 1 - ET_obj_func        # f = 1 - KGE (0~+∞, minimized)

                    # NingDu_g_Q   = -NingDu_obj_func       # -KGE <= 0
                    # ShiCheng_g_Q = -ShiCheng_obj_func     # -KGE <= 0
                    # FenKeng_g_Q  = -FenKeng_obj_func      # -KGE <= 0
                    # ET_g         = -ET_obj_func           # -KGE <= 0

                    out['F'] = np.column_stack([NingDu_f_Q, ShiCheng_f_Q, FenKeng_f_Q, ET_f])  # Objective functions
                    # out['G'] = np.column_stack([NingDu_g_Q, ShiCheng_g_Q, FenKeng_g_Q, ET_g])  # Constraints

                    print('F:', out['F'].shape)

                elif self.objectf[0] in ['RMSE', 'PBIAS']:
                    # Calculate Objective Functions and Constraints
                    NingDu_f_Q   = NingDu_obj_func        # f = RMSE (0~+∞, minimized)
                    ShiCheng_f_Q = ShiCheng_obj_func      # f = RMSE (0~+∞, minimized)
                    FenKeng_f_Q  = FenKeng_obj_func       # f = RMSE (0~+∞, minimized)
                    ET_f         = ET_obj_func            # f = RMSE (0~+∞, minimized)

                    out['F'] = np.column_stack([NingDu_f_Q, ShiCheng_f_Q, FenKeng_f_Q, ET_f])  # Objective functions

                    print('F:', out['F'].shape)

        elif self.cal_vars == ['Streamflow', 'RZSW']:
            # SWAT Model Simulation
            rch_pop_obj_func_dict = self.swat_execution_run.SWAT_model_execution(x)

        # Three Variables
        elif self.cal_vars == ['Streamflow', 'ET', 'RZSW']:
            # SWAT Model Simulation
            rch_pop_obj_func_dict = self.swat_execution_run.SWAT_model_execution(x)
