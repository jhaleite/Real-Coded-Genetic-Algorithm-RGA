import numpy as np 

def crossover_operator(pc,Winner_Xj,o_j,Upper_Limit,Lower_Limit,nc):

    N = len(Winner_Xj)
    
    ofssprings = np.copy(o_j)

    Idex_Xj = np.arange(N)
    
    r = np.zeros((int(N/2)),dtype=float)

    for i in range(len(r)):
        r[i] = np.random.uniform(0,1)
        
#### cria os apontadores onde se vao guardar as posiçoes dos pais 
    Pointer1 = np.zeros((int(N/2)),dtype=int)
    Pointer2 = np.zeros((int(N/2)),dtype=int)
    Pointer1[0:int(N/2)] = Idex_Xj[0:int(N/2)]
    Pointer2[0:int(N/2)] = Idex_Xj[int(N/2):N]
    np.random.shuffle(Pointer1)
    np.random.shuffle(Pointer2)

    p1 = 0 
    p2 = 0 
    beta = 0
    beta_memory = []
    ui = np.zeros((len(Winner_Xj[0])),dtype=float)



    for k in range(len(r)): #linhas seguintes sao a condição que garante que o p1  é sempre < que o p2
        for i in range (len(Winner_Xj[0])):
            if r[k] <= pc:
                if Winner_Xj[Pointer1[k]][i] < Winner_Xj[Pointer2[k]][i]:
                    p1 = Winner_Xj[Pointer1[k]][i] #x1 pai 1 
                    p2 = Winner_Xj[Pointer2[k]][i] #x1 pai 2 

                elif Winner_Xj[Pointer1[k]][i] > Winner_Xj[Pointer2[k]][i]:
                    p1 = Winner_Xj[Pointer2[k]][i] #x1 pai 1 
                    p2 = Winner_Xj[Pointer1[k]][i] #x1 pai 2

                elif Winner_Xj[Pointer1[k]][i] == Winner_Xj[Pointer2[k]][i]:
                    p1 = Winner_Xj[Pointer2[k]][i] #x1 pai 1 
                    p2 = Winner_Xj[Pointer1[k]][i] #x1 pai 2

                ########### A testar ainda: calculo dos Betas
                #B_L = p1+p2-2*Lower_Limit/abs(p2-p1)
                #B_U = 2*Upper_Limit-p1-p2/abs(p2-p1)
                # B1 = np.random.uniform(B_L,B_U)
                # B2 = np.random.uniform(B_L,B_U)

                ui[i] = np.random.uniform(0,1)
                if ui[i] <= 0.5: 
                    beta = (2*ui[i])**(1/(nc+1))
                    beta_memory.append(beta)

                else:
                    beta = (1/(2*(1-ui[i])))**(1/(nc+1))
                    beta_memory.append(beta)


                ofssprings[Pointer1[k]][i] = 0.5*((1+beta)*p1 + (1-beta)*p2)
                ofssprings[Pointer2[k]][i] = 0.5*((1-beta)*p1 + (1+beta)*p2)

                #ofssprings[Pointer1[k]][i] = 0.5*((p1 + p2) - abs((beta*(p2 - p1))))
                #ofssprings[Pointer2[k]][i] = 0.5*((p1 + p2) + abs((beta*(p2 - p1))))
    
                if ofssprings[Pointer1[k]][i] > Upper_Limit:
                    ofssprings[Pointer1[k]][i] = Upper_Limit
                elif ofssprings[Pointer1[k]][i] < Lower_Limit:
                    ofssprings[Pointer1[k]][i] = Lower_Limit
                elif ofssprings[Pointer2[k]][i] > Upper_Limit:
                    ofssprings[Pointer2[k]][i] = Upper_Limit
                elif ofssprings[Pointer2[k]][i] < Lower_Limit:
                    ofssprings[Pointer2[k]][i] = Lower_Limit
            
            else:
            
                ofssprings[Pointer1[k]][i] = Winner_Xj[Pointer1[k]][i]
                ofssprings[Pointer2[k]][i] = Winner_Xj[Pointer2[k]][i] 

                
    return ofssprings, beta_memory

def mutation_operator(pm,o_j_aux,Upper_Limit,Lower_Limit,nm):
    
    N = len(o_j_aux)
    n = len(o_j_aux[0])
    o_j_aux_1 = np.copy(o_j_aux)
    delta = 0

    for k in range(N):
        random_no = np.random.uniform(0,1) 
        for i in range (n):
            if random_no <= pm:
                rj = np.random.uniform(0,1)
                if rj < 0.5:
                    delta = ((2*rj)**(1/(nm+1)))-1
                elif rj >= 0.5:
                    delta = 1-((2*(1-rj)))**(1/(nm+1))
                    
                if k == 0:
                    break
                elif k >= 1:
                    o_j_aux_1[k][i] = o_j_aux_1[k][i] + ((Upper_Limit-Lower_Limit)*delta) 

                if o_j_aux_1[k][i] > Upper_Limit:
                    o_j_aux_1[k][i] = Upper_Limit
                elif o_j_aux_1[k][i] < Lower_Limit:
                    o_j_aux_1[k][i] = Lower_Limit
                elif o_j_aux_1[k][i] > Upper_Limit:
                    o_j_aux_1[k][i] = Upper_Limit
                elif o_j_aux_1[k][i] < Lower_Limit:
                    o_j_aux_1[k][i] = Lower_Limit
            
    return o_j_aux_1        

def binary_tournament(X_j, fx, Winner_Xj, Winner_fx, target):
    
    N = len(X_j)
    n = len(X_j[0])
    xj = np.copy(X_j)
    
    
    Fixed_Pointer = np.arange(N) # Pointer vector for the tournment 
    Variable_Pointer = np.copy(Fixed_Pointer) 
    Pointer_Winner_Vector = np.zeros((N),dtype=int)

 
    #Selection operator - Binary tournament
    np.random.shuffle(Variable_Pointer)
    np.random.shuffle(Fixed_Pointer)
    
    for i in range(N):
        if Fixed_Pointer[i] == Variable_Pointer[i]:
            while Fixed_Pointer[i] == Variable_Pointer[i]:
                Variable_Pointer[i] = np.random.randint(0,N)
                        
    if target == 'min':
        for i in range(N):
            if fx[Fixed_Pointer[i]] < fx[Variable_Pointer[i]]:
                Pointer_Winner_Vector[i] = Fixed_Pointer[i]
                Winner_Xj[i] = np.copy(X_j[Fixed_Pointer[i]])
                Winner_fx[i] = np.copy(fx[Fixed_Pointer[i]])
                
            elif fx[Fixed_Pointer[i]] > fx[Variable_Pointer[i]]:
                Pointer_Winner_Vector[i] = Variable_Pointer[i]
                Winner_Xj[i] = np.copy(X_j[Variable_Pointer[i]])
                Winner_fx[i] = np.copy(fx[Variable_Pointer[i]])
                
    if target == 'max':
        for i in range(N):
            if fx[Fixed_Pointer[i]] > fx[Variable_Pointer[i]]:
                Pointer_Winner_Vector[i] = Fixed_Pointer[i]
                Winner_Xj[i] = np.copy(X_j[Fixed_Pointer[i]])
                Winner_fx[i] = np.copy(fx[Fixed_Pointer[i]])

            elif fx[Fixed_Pointer[i]] < fx[Variable_Pointer[i]]:
                Pointer_Winner_Vector[i] = Variable_Pointer[i]
                Winner_Xj[i] = np.copy(X_j[Variable_Pointer[i]])
                Winner_fx[i] = np.copy(fx[Variable_Pointer[i]])  
    
    return Winner_Xj , Winner_fx     

def KinhoDEV(X_j, Noise_Parameter):
    
    x_j = np.copy(X_j)
    n = len(x_j[0])
    N = len(x_j)
    aux = np.copy(x_j)
    
    for i in range(N):
        if i > 0:
            for j in range(n):
                x_j[i][j] = aux[i][j] + (Noise_Parameter * np.random.normal())    
                
    return x_j    