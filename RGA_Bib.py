
from tkinter import N
import numpy as np 
import RGA_Operators
import External_Functions

class RGA:
    
    def __init__(self, Population_size, Variables_number, Number_max_generations, Upper_Variables_Bound, Lower_Variable_Bound, probability_of_crossover, crossover_variable, mutation_variable, Noise_parameter):
        

        #########################################################################################################################################################################
        # DECLARAÇÃO DE VARIAVEIS
        #########################################################################################################################################################################    

        self.N = Population_size # Population size 
        self.n = Variables_number # Number of real variables
        self.T = Number_max_generations # N of generations 
        self.Noise_Parameter = Noise_parameter # Constant used for disturbance
        self.Constant_Noise_Parameter = Noise_parameter
        
        self.fx = np.zeros((self.N), dtype=float)  
        self.fc = np.zeros((self.N), dtype=float)
        self.fm = np.zeros((self.N), dtype=float)
        self.fo = np.zeros((self.N), dtype=float)

        self.Upper_Limit = Upper_Variables_Bound
        self.Lower_Limit = Lower_Variable_Bound


        self.X_j = np.zeros((self.N,self.n),dtype=float)
        self.o_j = np.copy(self.X_j)
        
        # Operators used in GA 
        self.Pc = probability_of_crossover # probability of crossover
        self.Pm = 1/self.n # Probability of mutation
        self.nc = crossover_variable # SBX crossover operator 
        self.nm = mutation_variable  # polynomial mutation operator 

        # Initializng random population 

        for i in range(self.N):
            for j in range(self.n):
                if (type(self.Lower_Limit) is int) or (type(self.Lower_Limit) is float): 
                    self.X_j[i][j] = np.random.uniform(self.Lower_Limit,self.Upper_Limit)   
                else:
                    self.X_j[i][j] = np.random.uniform(self.Lower_Limit[i][j],self.Upper_Limit[i][j])

        ##########################################################################################################################################################################
        # PROGRAMA PRINCIPAL
        ##########################################################################################################################################################################


        self.Winner_Xj = np.zeros((self.N,self.n),dtype=float)
        self.Winner_fx = np.zeros((self.N),dtype=float)
        self.Memory_fx = []
        self.Memory_t = []

    def RGA_Optimizer (self, Objective_Function, target):

        t = 0
        E = 0.0001
        M = 0
        standard_deviation = 0

        while t < self.T:
            
            x = np.copy(self.X_j) 
            self.fx = Objective_Function(xj=x)
            #print(self.fx)

            #Selection operator - Binary tournament
            
            self.Winner_Xj, self.Winner_fx = RGA_Operators.binary_tournament(x, self.fx, self.Winner_Xj, self.Winner_fx, target)
            
            '''if t == 0:
                self.Winner_Xj, self.Winner_fx = RGA_Operators.binary_tournament(x, self.fx, self.Winner_Xj, self.Winner_fx, target)
                
            else: 
                self.Winner_Xj, self.Winner_fx = np.copy(self.X_j), np.copy(self.fx)
            '''
            #################### crossover operator ###########################
            aux_o = np.copy(self.o_j)
            o_j_aux, beta_memory = RGA_Operators.crossover_operator(self.Pc,self.Winner_Xj, aux_o, self.Upper_Limit, self.Lower_Limit,self.nc) 
            
            self.fc = Objective_Function(xj=o_j_aux)
            #print(self.fc)
            ################### mutation operator ##############################

            self.o_j = RGA_Operators.mutation_operator(self.Pm,o_j_aux,self.Upper_Limit,self.Lower_Limit,self.nm)
            #self.o_j = RGA_Operators.mutation_operator(self.Pm,self.Winner_Xj,self.Upper_Limit,self.Lower_Limit,self.nm)

            ################## Elimination and Survivors ########################
            
            aux_fo = np.copy(self.o_j)
            self.fo = Objective_Function(xj=aux_fo)
            
            #print(self.fo)
            Ct = np.hstack((self.Winner_fx,self.fo))

            if target == 'min':
                Ct_sorted = sorted(Ct)
            elif target == 'max':
                Ct_sorted = sorted((Ct),reverse=True)

            Ct_matrix = np.vstack((self.Winner_Xj,self.o_j))
            
            pointer_ct = []
            
            for i in range(len(Ct)):
                for j in range(len(Ct_sorted)):
                    if Ct_sorted[i] == np.copy(Ct[j]):
                        pointer_ct.append(j)
                            
            Pt_matrix = np.copy(self.Winner_Xj)
            fo_ptMatrix = np.copy(self.fo)
            for i in range(len(self.X_j)):
                Pt_matrix[i] = np.copy(Ct_matrix[pointer_ct[i]])
            
            Alpha_Wolf = Pt_matrix[0]
            aux_fo_ptMatrix = np.copy(Pt_matrix)
            fo_ptMatrix = Objective_Function(xj=aux_fo_ptMatrix)

            #self.X_j[:,:] = Alpha_Wolf
            self.X_j = np.copy(Pt_matrix)
            self.fx = np.copy(fo_ptMatrix)
            
            ###################### Disturbance ###############################
            
            self.X_j = RGA_Operators.KinhoDEV(self.X_j, self.Noise_Parameter)
            
                    
            self.Noise_Parameter = self.Constant_Noise_Parameter * (1 - (t/self.T))
            
            self.Memory_fx.append(min(self.fx))
            self.Memory_t.append(t)
            t = t + 1 

#print(X_j)
#print(beta_memory)
#print (fo_ptMatrix)

    

