import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import RegAlgs as ra
import ExpSetups as es
import AlgoPrediction as ap
import Affichage as tool
import Control as ctrl

d=1
n=1000
a=.1


def slide3():
    n = 1000
    d = 1
    
    X,Y = es.setting_A(n,d)

    plt.scatter(X,Y, s=10, alpha=0.5)
    plt.xlabel(r'Ecart à l\textquoteright occupation standard', size=15)
    plt.ylabel(r'Ecart à la consommation standard', size=15)
    plt.title(r'Consommation en fonction de l\textquoteright occupation', size=25)
    plt.savefig("GraphSlide3.png")
    plt.show()


def slide4():
    n = 1000
    d = 1
    
    X,Y = es.setting_A(n,d)

    X_new = np.linspace(X.min(),X.max(),100)
    band = ap.conf_band_split(X,Y,a,ra.regalg_lasso,X_new)
    band_min = band[0]
    band_max = band[1]

    plt.scatter(X,Y, s=10, alpha=0.5)
    plt.plot(X_new,band_min,color='red')
    plt.plot(X_new,band_max,color='red')
    plt.fill_between(X_new,band_min,band_max,color='red',alpha=0.1,label='Couverture de prédiction C')

    plt.xlabel(r'Ecart à l\textquoteright occupation standard', size=15)
    plt.ylabel(r'Ecart à la consommation standard', size=15)
    plt.title(r'Consommation en fonction de l\textquoteright occupation', size=25)
    plt.legend()
    plt.savefig("GraphSlide4.png")
    plt.show()

def slide5_1():
    n = 50
    d = 1

    X,Y = es.setting_A(n,d)
    
    X_reg = np.array([X[i][0] for i in range(len(X))]) 
    X_reg.sort()   
    regalg = ra.regalg_rf(X_reg,Y)
    Y_reg = np.array([regalg(x) for x in X])

    plt.scatter(X,Y, s=10, alpha=0.5)
    plt.plot(X_reg,Y_reg, color='black', linestyle='--')

    plt.xlabel(r'X', size=15)
    plt.ylabel(r'Y', size=15)
    plt.title(r'Sur-entraînement', size=25)
    plt.savefig("GraphSlide5_1.png")
    plt.show()


def slide5_2():
    n = 50
    d = 1

    X,Y = es.setting_A(n,d)
    
    X_reg = np.linspace(X.min(),X.max(),100)
    Y_reg = np.array([0 for _ in range(100)])

    plt.scatter(X,Y, s=10, alpha=0.5)
    plt.plot(X_reg,Y_reg, color='black', linestyle='--')

    plt.xlabel(r'X', size=15)
    plt.ylabel(r'Y', size=15)
    plt.title(r'Sous-entraînement', size=25)
    plt.savefig("GraphSlide5_2.png")
    plt.show()


def slide6_1():
    n = 50
    d = 1

    X, Y = es.setting_A(n, d)

    X_n1 = -1
    Y_trial = np.linspace(Y.min(), Y.max(), 300)  # Plus fluide que 9 frames

    X_new = np.append(X, X_n1)


    fig, ax = plt.subplots()

    scatter_data = ax.scatter(X, Y, alpha=0.5, color='blue')
    point_test = ax.scatter([], [], color='red',label=r'Point de contrôle')
    reg_line, = ax.plot([], [], color='black')

    def init():
        return scatter_data, point_test, reg_line

    def update(frame):
        Y_i_j = Y_trial[frame]
        Y_new = np.append(Y, Y_i_j)

        regalg = ra.regalg_lasso(X_new, Y_new)
        X_reg = np.sort(X_new)
        Y_reg = np.array([regalg(x) for x in X_reg])

        point_test.set_offsets([[X_n1, Y_i_j]])
        reg_line.set_data(X_reg, Y_reg)

        return scatter_data, point_test, reg_line

    ani = animation.FuncAnimation(
        fig, update, frames=len(Y_trial),
        init_func=init, blit=True, interval=15
    )


    plt.xlabel(r'X')
    plt.ylabel(r'Y')
    plt.legend()

    ani.save("GraphSlide6_1.gif", writer="pillow", fps=20)
    plt.show()

def slide6_2():
    n = 50
    d = 1
    a=.1

    X_n1 = np.array([-1])
    X, Y = es.setting_A(n, d)
    Y_trial = np.linspace(Y.min(), Y.max(), 300)

    X_new = np.append(X, X_n1)

    regalg = ra.regalg_lasso(X,Y)

    band = ap.conf_band_full(X,Y,a,ra.regalg_lasso,X_n1,Y_trial)
    Y_min = band[0]
    Y_max = band[1]

    plt.scatter(X,Y,color='blue', alpha = .5)
    plt.scatter(X_n1,[Y_min], color='red')
    plt.scatter(X_n1,[Y_max], color='red')
    plt.fill_between(X_n1,Y_min,Y_max, color='red', alpha=0.5)
    
    plt.xlabel(r'X')
    plt.ylabel(r'Y')

    plt.savefig('GraphSlide6_2')
    plt.show()

def slide7():
    n = 50
    d = 1
    a=.1

    X, Y = es.setting_A(n, d)
    X_new = np.linspace(X.min(),X.max(),100)
    Y_trial = np.linspace(Y.min()-1, Y.max()+1, 100)


    regalg = ra.regalg_lasso(X,Y)    
    band = ap.conf_band_full(X,Y,a,ra.regalg_lasso,X_new,Y_trial)
    Y_min = band[0]
    Y_max = band[1]

    plt.scatter(X,Y,color='blue', alpha = .5)
    plt.plot(X_new,Y_min, color='red')
    plt.plot(X_new,Y_max, color='red')
    plt.fill_between(X_new,Y_min,Y_max, color='red', alpha=0.1)
    
    plt.xlabel(r'X')
    plt.ylabel(r'Y')

    plt.title(r'Full conformal',size=25)
    plt.savefig('GraphSlide7')
    plt.show()  


def slide8():
    n = 50
    d = 1
    a=.1

    X, Y = es.setting_A(n, d)
    X_new = np.linspace(X.min(),X.max(),100)

    band = ap.conf_band_split(X,Y,a,ra.regalg_lasso,X_new)
    I_train = band[3]
    I_ctrl = band[4]

    plt.scatter(X[I_train],Y[I_train], color = 'blue', alpha =0.5)
    plt.scatter(X[I_ctrl],Y[I_ctrl], color = 'green', alpha =0.5)

    plt.xlabel(r'X')
    plt.ylabel(r'Y')
   

    plt.savefig('GraphSlide8')
    plt.show()

def slide9_1():
    n = 50
    d = 1
    a=.1

    X, Y = es.setting_A(n, d)
    X_new = np.linspace(X.min(),X.max(),100)
    regalg = ra.regalg_lasso(X,Y)

   
    Y_reg = np.array([regalg(x) for x in X_new])


    band = ap.conf_band_split(X,Y,a,ra.regalg_lasso,X_new)
    I_train = band[3]

    plt.scatter(X[I_train],Y[I_train], color = 'blue', alpha =0.5)   
    plt.plot(X_new,Y_reg, color='black')
       
    plt.xlabel(r'X')
    plt.ylabel(r'Y')
   
    plt.title(r'Training points',size=25)
    plt.savefig('GraphSlide9_1')
    plt.show()

def slide9_2():
    n = 50
    d = 1
    a=.1

    X, Y = es.setting_A(n, d)
    X_new = np.linspace(X.min(),X.max(),100)
    regalg = ra.regalg_lasso(X,Y)
    

    band = ap.conf_band_split(X,Y,a,ra.regalg_lasso,X_new)
    I_ctrl = band[4]

    X_reg = np.array([X[i][0] for i in range(len(X))]) 
    Y_reg = np.array([regalg(x) for x in X_reg])

    lbl = True
    for i in I_ctrl:
        X_tmp = np.array([X[i] for _ in range(100)])
        Y_tmp = np.linspace(Y_reg[i],Y[i],100)
        
        if lbl:
            plt.plot(X_tmp,Y_tmp,color='red',linestyle='--',label=r'$R_i$')
            lbl = False
        else:
            plt.plot(X_tmp,Y_tmp,color='red',linestyle='--')

    plt.scatter(X[I_ctrl],Y[I_ctrl], color = 'green', alpha =0.5)   
    plt.plot(X[I_ctrl],Y_reg[I_ctrl], color='black')

       
    plt.xlabel(r'X')
    plt.ylabel(r'Y')

    plt.legend()
   
    plt.title(r'Control points',size=25)
    plt.savefig('GraphSlide9_2')
    plt.show()



def slide10():
    n = 50
    d = 1
    a=.1

    X, Y = es.setting_A(n, d)
    X_new = np.linspace(X.min(),X.max(),100)


    regalg = ra.regalg_lasso(X,Y)    
    band = ap.conf_band_split(X,Y,a,ra.regalg_lasso,X_new)
    Y_min = band[0]
    Y_max = band[1]
    I_train = band[3]
    I_ctrl = band[4]


    plt.scatter(X[I_train],Y[I_train],color='blue', alpha = 0.5)
    plt.scatter(X[I_ctrl],Y[I_ctrl],color='green',alpha=0.5)
    plt.plot(X_new,Y_min, color='red')
    plt.plot(X_new,Y_max, color='red')
    plt.fill_between(X_new,Y_min,Y_max, color='red', alpha=0.1)
    
    plt.xlabel(r'X')
    plt.ylabel(r'Y')

    plt.title(r'Split conformal',size=25)
    plt.savefig('GraphSlide10')
    plt.show()  



def slide12_1(): 
    n = 1000
    d = 1
    a=.1

    X, Y = es.setting_P5(n, d)
    X_new = np.linspace(X.min(),X.max(),100)


    regalg = ra.regalg_spline(X,Y)    
    band = ap.conf_band_split(X,Y,a,ra.regalg_spline,X_new)
    Y_min = band[0]
    Y_max = band[1]
    I_train = band[3]
    I_ctrl = band[4]


    plt.scatter(X[I_train],Y[I_train],color='blue', alpha = 0.5)
    plt.scatter(X[I_ctrl],Y[I_ctrl],color='green',alpha=0.5)
    plt.plot(X_new,Y_min, color='red')
    plt.plot(X_new,Y_max, color='red')
    plt.fill_between(X_new,Y_min,Y_max, color='red', alpha=0.1)
    
    plt.xlabel(r'X')
    plt.ylabel(r'Y')

    plt.title(r'Split conformal',size=25)
    plt.savefig('GraphSlide12_1')
    plt.show()  

def slide12_2(): 
    n = 1000
    d = 1
    a=.1

    X, Y = es.setting_P5(n, d)
    X_new = np.linspace(X.min(),X.max(),100)


    regalg = ra.regalg_spline(X,Y)    
    band = ap.conf_band_split_LW(X,Y,a,ra.regalg_spline,X_new)
    Y_min = band[0]
    Y_max = band[1]
    I_train = band[3]
    I_ctrl = band[4]


    plt.scatter(X[I_train],Y[I_train],color='blue', alpha = 0.5)
    plt.scatter(X[I_ctrl],Y[I_ctrl],color='green',alpha=0.5)
    plt.plot(X_new,Y_min, color='red')
    plt.plot(X_new,Y_max, color='red')
    plt.fill_between(X_new,Y_min,Y_max, color='red', alpha=0.1)
    
    plt.xlabel(r'X')
    plt.ylabel(r'Y')

    plt.title(r'Split conformal locally-weighted',size=25)
    plt.savefig('GraphSlide12_2')
    plt.show()  



def merci():
    width = .1
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    def trait(depart,arrivee):
        x_1,y_1 = depart
        x_2,y_2 = arrivee
        
        x_min = min(x_1,x_2)
        y_min = min(y_1,y_2)
        x_max = max(x_1,x_2)
        y_max = max(y_1,y_2)

        d_eucl = np.sqrt((x_1-x_2)**2+(y_1-y_2)**2)
        nb_point = int(50*d_eucl)

        if x_min == x_max: 
            X = np.random.normal(x_min,width,nb_point)
            Y = np.random.uniform(y_min-width,y_max+width,nb_point)
            return X,Y
        
        elif y_min == y_max:
            X = np.random.uniform(x_min-width,x_max+width,nb_point)
            Y = np.random.normal(y_min,width,nb_point)
            return X,Y            
        else: 
            delta = (y_1-y_2)/(x_1-x_2)
            eps = np.random.normal(0,width,nb_point)
            X = np.random.uniform(x_min-width,x_max+width,nb_point)
            Y = y_1 + delta*(X-x_min) + eps
            return X,Y
    
    M = [(1,1),(1,4),(2,3),(3,4),(3,1)]
    E_1 = [(6,4),(4,4),(4,1),(6,1)]
    E_2 = [(4,2.5),(5,2.5)]
    R = [(7,1),(7,4),(9,4),(9,2.5),(7,2.5),(9,1)]
    C = [(12,4),(10,4),(10,1),(12,1)]
    I = [(13,1),(13,4)]

    texte = [M,E_1,E_2,R,C,I]

    for lettre in texte:
        for i in range (len(lettre)-1):
            depart = lettre[i]
            arrivee = lettre[i+1]
            X,Y = trait(depart,arrivee)
            
            size_list = np.random.normal(50,25,len(X)) 
            size_list = np.array([max(float(s),20) for s in size_list])

            c_r = np.random.uniform(0, 1, len(X))
            c_g = np.random.uniform(0, 1, len(X))
            c_b = np.random.uniform(0, 1, len(X))    
            color_list = np.array([(c_r[i],c_g[i],c_b[i]) for i in range(len(X))])        

            plt.scatter(X,Y,alpha=.2,s=size_list,c=color_list)
    
    X_out = np.array([])
    Y_out = np.array([])
    for lettre in texte:
        for i in range(len(lettre)-1):
            depart = lettre[i]
            arrivee = lettre[i+1]
            X,Y = trait(depart,arrivee)
            X_out = np.concatenate((X_out,X))
            Y_out = np.concatenate((Y_out,Y))
    
    X_new = np.linspace(0,14,100)
    
    plt.axis((0,14,0,5))
    plt.tight_layout()

    #tool.plot_band(X_out,Y_out,.1,ra.regalg_lasso,ap.conf_band_split_LW,X_new)
    #plt.savefig('Merci_simple.png')
    plt.show()


    



if __name__ == '__main__':
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"  # facultatif
    })
    #seed = 1337
    np.random.seed(1337)

    merci()