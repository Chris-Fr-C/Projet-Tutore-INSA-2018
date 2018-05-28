#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:40:02 2018

Basé sur le problème posé par Marc Budinger ainsi que son code partiel en script.

@author: christophe
"""
import scipy
import scipy.optimize
from math import pi
import numpy as np
import pandas as pd
import preprocess
import pyDOE
import sys
sys.flush()

preprocess.run()

# TWEAKING PARAMETERS
METHOD  = "SLSQP"
SOLVER_OPTIONS = {'maxiter':5000, 'disp':False}
TOL = 1e-3



# Optimization

class InitializationVariables():
    def __init__(self):
        self.static_force=40e3   # [N]
        self.inertial_load=800.  # [kg]
        self.displacement_magnitude=10e-3 # [m]
        self.frequency=2   # [Hz]
        # Speed and load acceleration
        self.max_speed=2*pi*self.frequency*self.displacement_magnitude     # [m/s]
        self.max_acceleration=2*pi*self.frequency*self.max_speed   # [m/s²]
        self.max_load=self.static_force+self.max_acceleration*self.inertial_load    # [N]
        # Imposed values
        self.pitch=10e-3/2/pi  # [m/rad]
        # Optimisation variables
        #Variables d'optimisation
        self.n_reduc=2
        self.k_oversizing=1
        
        self.last_results = dict()
        
        self.parameters = np.array([self.n_reduc, self.k_oversizing]).reshape(-1,1) # Column vector
        
        # The constraints are the same so they can be defined here.
        return
    
    def get_dataframe(self,x):
        key = ( float(x[0]), float(x[1]) )
        
        values = self.last_results[key]
        df = pd.DataFrame(data={"Parameter":tuple(values.keys()),"Value":tuple(values.values())})
        return df

class ScalingObject(InitializationVariables):
    def __init__(self):
        super().__init__()
        
        # Constraints preparation
        # Computing initial parameters if the solver requires to check
        # the constraints before an iteration
        self.sizing_code(self.parameters)
        
        self.constraints = (
                {'type':'ineq','fun':self.constraint_1},
                {'type':'ineq','fun':self.constraint_2},
                {'type':'ineq','fun':self.constraint_3},
                {'type':'ineq','fun':self.constraint_4},
                )

        return
 
    def constraint_1(self,x:np.ndarray):
        n_reduc = float(x[0])
        const = self.speed-n_reduc*self.max_speed/self.pitch
        return const
    def constraint_2(self,x:np.ndarray):
        const = self.t_em_est-self.t_em_max
        return const
    
    def constraint_3(self,x:np.ndarray):
        """
        k_oversizing >1
        equals to
        k_oversizing -1.0 > 0.0
        """
        const = float(x[1]) -1.0
        return const
    def constraint_4(self,x:np.ndarray):
        """mass>0"""
        const= self.mass
        return const
    
    def sizing_code(self,param):    
        #  Variables
        n_reduc=param[0]
        k_oversizing=param[1]
        # extracting self variables for easier reading
        max_load = self.max_load
        pitch = self.pitch
        max_acceleration = self.max_acceleration
        max_speed = self.max_speed
        # Torque motor estimation
        tem_est= k_oversizing*max_load*pitch/n_reduc
    
        
        # Parameter estimation with scaling laws
        Tmax_ref=13.4    # [N.m]
        Wmax_ref=754    # [rad/s]
        Jref=2.9e-4     # [kg.m²]
        Mref=3.8        # [kg]
    
        mass=Mref*(tem_est/Tmax_ref)**(3/3.5)
        inertia=Jref*(tem_est/Tmax_ref)**(5/3.5)
        speed=Wmax_ref*(tem_est/Tmax_ref)**(-1/3.5)
    
        
        # Saving variables
        self.n_reduc = n_reduc
        self.k_oversizing = k_oversizing
        self.mass = mass
        self.inertia= inertia
        self.speed = speed
        self.t_em_est = tem_est
    
        # -------------------------------------------------
    
        # Exact torque calculation with motor inertia
        tem_max=max_load*pitch/n_reduc+inertia*max_acceleration*n_reduc/pitch
        
        self.t_em_max = tem_max
    
        # Saving the current state values in the dict we prepared (that might be used for the projection)
        key = (float(param[0]),float(param[1]))
        self.last_results[key] = {'mass':mass,'speed':speed,'inertia':inertia,'t_em_max':tem_max}
    
        # Objectifs et contraintes

        return mass
    
    def sizing_code_cobyla_wrapper(self, param, arg):
        mass = self.sizing_code(param)
        if arg=='Obj':
            return mass
        else:
            #print('constraints sizing')
            #dfp = pd.DataFrame(data = {"Data":["v-nred*vmax/pitch","tem_est-tem_max"],"Value":[speed-n_reduc*max_speed/pitch, tem_est-tem_max]})
            #dfp = dfp.set_index("Data")
            #print(dfp)
            return [
                    self.speed-self.n_reduc*self.max_speed/self.pitch,
                    self.t_em_est-self.t_em_max,
                    float(self.k_oversizing -1.0)
                    ]

    

class CombinedObject(InitializationVariables):
    def __init__(self):
        super().__init__()
        self.sizing_code(self.parameters)
        
        self.constraints = (
                {'type':'ineq','fun':self.constraint_1},
                {'type':'ineq','fun':self.constraint_2},
                {'type':'ineq','fun':self.constraint_3},
                {'type':'ineq','fun':self.constraint_4},
                )
        return
    
    def constraint_1(self,x:np.ndarray):
        n_reduc = float(x[0])
        const = self.speed-n_reduc*self.max_speed/self.pitch
        return const
    def constraint_2(self,x:np.ndarray):
        const = self.t_em_est-self.t_em_max
        return const
    
    def constraint_3(self,x:np.ndarray):
        """
        k_oversizing >1
        equals to
        k_oversizing -1.0 > 0.0
        """
        const = float(x[1]) -1.0
        return const
    
    def constraint_4(self,x:np.ndarray):
        """mass > 0"""
        const = self.mass
        return const
    
    def sizing_code(self,params):
        # Ensuring the proper format of the variable
        # Extracting variables
        n_reduc = params[0]
        k_oversizing = params[1]
        
        # extracting variables from "self" for easier reading
        max_load = self.max_load
        pitch = self.pitch
        max_acceleration = self.max_acceleration
        
        # Torque motor estimation
        t_em_est= k_oversizing*max_load*pitch/n_reduc
        # =======================================
        # Computation part
        # =======================================
        # Estimation part:
        # ===================
    
        # Building the prediction models
        t_arr = np.array([t_em_est]).reshape(-1,1)
        mass = preprocess.combined_models['m'].predict(t_arr)[0]
        speed = preprocess.combined_models['speed'].predict(t_arr)[0] # *60/(2*np.pi)#should it be in rad/sec ?
        inertia = float(preprocess.combined_models['inertia'].predict(t_arr)[0])
    
    
        # Saving parameters
        self.mass = mass
        self.speed = speed
        self.inertia = inertia
        self.n_reduc = n_reduc
        self.k_oversizing = k_oversizing
        self.t_em_est = t_em_est
        # Parameter estimation with scaling laws
        """
        t_max_ref=13.4    # [N.m]
        w_max_ref=754    # [rad/s]
        j_ref=2.9e-4     # [kg.m²]
        m_ref=3.8        # [kg]
    
        mass=m_ref*(t_em_est/t_max_ref)**(3/3.5)
        inertia=j_ref*(t_em_est/t_max_ref)**(5/3.5)
        speed=w_max_ref*(t_em_est/t_max_ref)**(-1/3.5)
        """
        # ===================
        #  # Exact torque calculation with motor inertia
        t_em_max= max_load*pitch/n_reduc+inertia*max_acceleration*n_reduc/pitch
        
        self.t_em_max = t_em_max
        
        # Saving the current state values in the dict we prepared (that might be used for the projection)
        key = (float(params[0]),float(params[1]))
        self.last_results[key] = {'mass':float(mass),'speed':float(speed),'inertia':float(inertia),'t_em_max':float(t_em_max)}
        return mass

    
    def sizing_code_cobyla_wrapper(self,params:np.ndarray, options:str):
        """Receiving the parameters and computing the output.
        
        Args:
            params (np.ndarray): Vector composed as the following : 
                                 [  
                                    [n_reduc],
                                    [k_oversizing] 
                                    ]
            options (str): Option of the function.
        """
        mass = self.sizing_code(params)
        # Objectifs et contraintes
        if options=='Obj':
            return mass
        else:
            #dfp = pd.DataFrame(data = {"Data":["v-nred*vmax/pitch","tem_est-tem_max","k_oversizing-1.0"],"Value":[float(speed-n_reduc*max_speed/pitch), float(t_em_est-t_em_max), float(k_oversizing-1.0)]})
            #dfp = dfp.set_index("Data")
            #print("\n\n")
            #print(dfp)
            return [float(self.speed-self.n_reduc*self.max_speed/self.pitch),
                    float(self.t_em_est-self.t_em_max),
                    float(self.k_oversizing-1.0)]


class TreeObject(CombinedObject):
    def __init__(self):
        super().__init__()
        return
    
    def sizing_code(self,params):
        """
        We project mass inertia before using it for the computation
        """
        # Ensuring the proper format of the variable
        # Extracting variables
        n_reduc = params[0]
        k_oversizing = params[1]
        
        # extracting variables from "self" for easier reading
        max_load = self.max_load
        pitch = self.pitch
        max_acceleration = self.max_acceleration
        
        # Torque motor estimation
        t_em_est= k_oversizing*max_load*pitch/n_reduc
        # =======================================
        # Computation part
        # =======================================
        # Estimation part:
        # ===================

        # Building the prediction models
        t_arr = np.array([t_em_est]).reshape(-1,1)
        mass = preprocess.tree_models['m'].predict(t_arr)[0]
        speed = preprocess.tree_models['speed'].predict(t_arr)[0] # *60/(2*np.pi)#should it be in rad/sec ?
        inertia = float(preprocess.tree_models['inertia'].predict(t_arr)[0])
    
    
        # Saving parameters
        self.mass = mass
        self.speed = speed
        self.inertia = inertia
        self.n_reduc = n_reduc
        self.k_oversizing = k_oversizing
        self.t_em_est = t_em_est

        # ===================
        #  # Exact torque calculation with motor inertia
        t_em_max= max_load*pitch/n_reduc+inertia*max_acceleration*n_reduc/pitch
        
        self.t_em_max = t_em_max
        
        # Saving the current state values in the dict we prepared (that might be used for the projection)
        key = (float(params[0]),float(params[1]))
        self.last_results[key] = {'mass':float(mass),'speed':float(speed),'inertia':float(inertia),'t_em_max':float(t_em_max)}
        return mass


# Building the initialization variables for the two optimizations.
        
v_scal = ScalingObject()
v_comb = CombinedObject()
v_scal_2 = ScalingObject()
v_comb_2 = CombinedObject()
v_tree = TreeObject()
v_tree_2 = TreeObject()


print("="*50)
print("\t\tVEGA CASE")
print("="*50)
print("\t\tScaling laws optimization")
# INITIAL VALUES
print("Initial values")
print("\tn_reduc = ", v_scal.n_reduc)
print("\tk_oversizing  = ", v_scal.k_oversizing)
# COBYLA OPTIMIZATION
print("\nRunning the optimization")
result_scaling_laws = scipy.optimize.fmin_cobyla(func=v_scal.sizing_code_cobyla_wrapper, x0=v_scal.parameters, cons=v_scal.sizing_code_cobyla_wrapper, args=('Obj',), consargs=('Cst',) )

# Extraction of the variables

print("Cobyla results are : ")
print("\tn_reduc : {res}".format(res=result_scaling_laws[0]))
print("\tk_surdim : {res}".format(res=result_scaling_laws[1]))
print("\tVariables related are")
print(v_scal.get_dataframe(result_scaling_laws))

print("- "*25)
# MINIMIZATION

result_scaling_laws_2 = scipy.optimize.minimize(v_scal_2.sizing_code,
                                                v_scal_2.parameters,
                                                method = METHOD,
                                                constraints = v_scal_2.constraints,
                                                tol=TOL,
                                                options=SOLVER_OPTIONS)
print("{meth} results are : ".format(meth=METHOD))
print("\tSucess : ",result_scaling_laws_2.success)
print("\tOutput message : ",result_scaling_laws_2.message)
print("\tn_reduc : {res}".format(res=result_scaling_laws_2.x[0]))
print("\tk_surdim : {res}".format(res=result_scaling_laws_2.x[1]))
print("\tVariables related are")
print(v_scal_2.get_dataframe(result_scaling_laws_2.x))


print("="*50)
print("\t\tCombined model optimization")
# INITIAL VALUES
print("preprocess module is")
print("Initial values")
print("\tn_reduc = ", v_comb.n_reduc)
print("\tk_oversizing  = ", v_comb.k_oversizing)
print("\nRunning the optimization")
result_comb = scipy.optimize.fmin_cobyla(func=v_comb.sizing_code_cobyla_wrapper, x0=v_comb.parameters, cons=v_comb.sizing_code_cobyla_wrapper, args=('Obj',), consargs=('Cst',) )
print("Cobyla results are : ")
print("\tn_reduc : {res}".format(res=result_comb[0]))
print("\tk_surdim : {res}".format(res=result_comb[1]))
print("\tVariables related are")
print(v_comb.get_dataframe(result_comb))

# MINIMIZATIOn
result_combined_2 = scipy.optimize.minimize(v_comb_2.sizing_code,
                                                v_comb_2.parameters,
                                                method = METHOD,
                                                constraints = v_comb_2.constraints,
                                                tol=TOL,
                                                options=SOLVER_OPTIONS)
print("- "*25)
print("{meth} results are : ".format(meth=METHOD))
print("\tSucess : ",result_combined_2.success)
print("\tOutput message : ",result_combined_2.message)
print("\tn_reduc : {res}".format(res=result_combined_2.x[0]))
print("\tk_surdim : {res}".format(res=result_combined_2.x[1]))

print("\tVariables related are")
print(v_comb_2.get_dataframe(result_combined_2.x))


print("="*50)
print("\t\Tree model optimization")
# INITIAL VALUES
print("preprocess module is")
print("Initial values")
print("\tn_reduc = ", v_tree.n_reduc)
print("\tk_oversizing  = ", v_tree.k_oversizing)
print("\nRunning the optimization")
result_tree = scipy.optimize.fmin_cobyla(func=v_tree.sizing_code_cobyla_wrapper, x0=v_comb.parameters, cons=v_tree.sizing_code_cobyla_wrapper, args=('Obj',), consargs=('Cst',) )
print("Cobyla results are : ")
print("\tn_reduc : {res}".format(res=result_tree[0]))
print("\tk_surdim : {res}".format(res=result_tree[1]))
print("\tVariables related are")
print(v_tree.get_dataframe(result_tree))

# MINIMIZATION
# Setting initial point of the optimization to be next to the old x_opt found 
# with RBF
result_tree_2 = scipy.optimize.minimize(v_tree_2.sizing_code,
                                                v_tree_2.parameters,
                                                method = METHOD,
                                                constraints = v_tree_2.constraints,
                                                tol=TOL,
                                                options=SOLVER_OPTIONS)
print("- "*25)
print("{meth} results are : ".format(meth=METHOD))
print("\tSucess : ",result_tree_2.success)
print("\tOutput message : ",result_tree_2.message)
print("\tn_reduc : {res}".format(res=result_tree_2.x[0]))
print("\tk_surdim : {res}".format(res=result_tree_2.x[1]))

print("\tVariables related are")
print(v_tree_2.get_dataframe(result_tree_2.x))

# DOE on the results of the RBF optimization
print("\n"+"="*25+"DOE"+"="*25)
doe_model = v_tree_2
SENSIBILITY_NRED = 2 #+-2
SENSIBILITY_KSURD = .2  # +-0.2
SHALL_RESPECT_CONSTRAINTS = True
central_point = result_combined_2.x
doe_model.parameters = central_point
print("Central point of the doe (result from the combined model optimization) :")
print("\t",central_point[0],central_point[1])
print("With an objective value (on the comined) model of : ",v_comb_2.sizing_code(central_point))
# Starting objective function value
obj_val_starting = doe_model.sizing_code(central_point)

doe=pyDOE.lhs(2,samples=10)
# setting the doe to be positive and negative
doe = (doe - 0.5)*2
# Rescaling to the values we want
doe = doe * (SENSIBILITY_NRED*2, SENSIBILITY_KSURD*2)
# Adding the offset
doe = doe + central_point.reshape(-1,2)

# Doing the evaluation of each the objective function
minimum_found = obj_val_starting
found_at = central_point
new_point_respects_constraints = True
for point in doe:
    # It is really important to call the sizing code BEFORE the constraints
    # evaluation. Because it saves some values.
    new_val = doe_model.sizing_code(point)
    constraints_respected = [check['fun'](point)>0.0 for check in doe_model.constraints]
    constraints_respected = np.array(constraints_respected)
    respect_all = constraints_respected.all()
    if (new_val < minimum_found) and ((not SHALL_RESPECT_CONSTRAINTS) or respect_all):
            minimum_found = new_val
            found_at = point
            new_point_respects_constraints = respect_all
            
print("DOE fully evaluated with a sensibility of +-",SENSIBILITY_NRED, " and +-", SENSIBILITY_KSURD)
print("Initial minimum (tree model) {m} found at {p}".format(m=obj_val_starting,p=central_point))
print("New minimum has a value of {v} found at  {p}".format(v = minimum_found, p = found_at))
print("Checking constraints : ",SHALL_RESPECT_CONSTRAINTS)
print("This point respects the constraints after projection: ",new_point_respects_constraints)
print("="*25+"END"+"="*25)