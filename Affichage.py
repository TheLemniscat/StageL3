import numpy as np
import matplotlib.pyplot as plt

from RegAlgs import regalg_spline
from typing import Callable



def plot_band(X:np.ndarray,
              Y:np.ndarray,
              a:float,
              reg_fct:Callable[[np.ndarray,np.ndarray], Callable[[np.ndarray], float]],
              prd_alg:Callable[[np.ndarray,np.ndarray,float,Callable[[np.ndarray,np.ndarray], Callable[[np.ndarray], float]],np.ndarray|None], tuple],
              X_new:np.ndarray=np.array([])) -> None:
    """
    Affiche une bande de prédiction conforme pour les données X et Y.

    Paramètres
    ----------
    **np.ndarray** X : Tableau de forme (n, 1) contenant les é
    **np.ndarray** Y : Tableau de taille n contenant les valeurs de la variable expliquée.
    **float** a : Le niveau de signification pour la prédiction conforme.
    **Callable** reg_fct : Une fonction de régression qui prend X et Y et retourne
        une fonction entraînée sur les données.
    **Callable** prd_alg : Une fonction de prédiction conforme qui prend X, Y, a, reg_fct et X_new (optionnel) et retourne un tuple contenant les born
    **np.ndarray** X_new : Tableau de taille m contenant les nouvelles valeurs pour lesquelles on
        souhaite prédire les intervalles de confiance. Si None, il sera généré automatiquement.
    """

    
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"  # facultatif
    })

    if X_new.size == 0:
        X_new = np.linspace(X.min(), X.max(), 100)


    reg_fct_fct = reg_fct(X,Y)
    
    X_reg = np.linspace(min(X.min(), X_new.min()), max(X.max(), X_new.max()), 1000)
    Y_reg = np.array([reg_fct_fct(x) for x in X_reg])    

    band = prd_alg(X,Y,a,reg_fct,X_new)

    C_min = band[0]
    C_max = band[1]
    X_new = band[2]


    if (prd_alg.__name__ == 'conf_band_split') or (prd_alg.__name__ == 'conf_band_split_LW'):
        I_train = band[3]
        I_ctrl = band[4]
        
        X_train = np.array([X[i] for i in I_train])
        Y_train = np.array([Y[i] for i in I_train])

        X_ctrl = np.array([X[i] for i in I_ctrl])
        Y_ctrl = np.array([Y[i] for i in I_ctrl])
        
        plt.scatter(X_train,Y_train,s=10, alpha=0.5, color = 'blue', label = 'train data')
        plt.scatter(X_ctrl,Y_ctrl,s=10, alpha=0.5, color = 'green', label = 'control data')
    
    else:
        plt.scatter(X,Y, s=10, alpha=0.5, label = 'data')
    
    plt.plot(X_reg,Y_reg, color='black',linestyle = 'dotted', label = 'regression')

    plt.plot(X_new,C_min,color='red', label = 'prediction band')
    plt.plot(X_new,C_max,color='red')
    plt.fill_between(X_new, C_min, C_max, alpha=0.1, color='red')


    #plt.title(f'n={len(X)}, alpha={a}, reg=SplitConfLw, pred=Spline')
    plt.title(f'n={len(X)}, alpha={a}, reg={reg_fct.__name__}, pred={prd_alg.__name__}')

    plt.legend()
    plt.show()


def plot_band_mult(X:np.ndarray,
                   Y:np.ndarray,
                   a:float,
                   reg_fct:Callable[[np.ndarray,np.ndarray], Callable[[np.ndarray], float]],
                   prd_alg:Callable[[np.ndarray,np.ndarray,float,Callable[[np.ndarray,np.ndarray], Callable[[np.ndarray], float]],np.ndarray|None], tuple],
                   X_new:np.ndarray=np.array([])) -> None:
   
    """
    Affiche plusieurs bandes de prédiction conforme pour les données X et Y.

    Paramètres
    ----------
    **np.ndarray** X : Tableau de forme (n, 1) contenant les é
    **np.ndarray** Y : Tableau de taille n contenant les valeurs de la variable expliquée.
    **float** a : Le niveau de signification pour la prédiction conforme.
    **Callable** reg_fct : Une fonction de régression qui prend X et Y et retourne
        une fonction entraînée sur les données.
    **Callable** prd_alg : Une fonction de prédiction conforme qui prend X, Y, a, reg_fct et X_new (optionnel) et retourne un tuple contenant les born
    **np.ndarray** X_new : Tableau de taille m contenant les nouvelles valeurs pour lesquelles on
        souhaite prédire les intervalles de confiance. Si None, il sera généré automatiquement.
    """

    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"  # facultatif
    })

    if X_new.size == 0:
        X_new = np.linspace(X.min(), X.max(), 100)

    
    reg_fct_fct = reg_fct(X,Y)

    X_reg = np.linspace(min(X.min(),X_new.min()), max(X.max(),X_new.max()), 1000)
    Y_reg = np.array([reg_fct_fct(x) for x in X_reg])

    k,l = 3,3

    fig,axs = plt.subplots(k,l)

    for i in range(k):
        for j in range(l):
            

            band = prd_alg(X,Y,a,reg_fct,X_new)
            C_min = band[0]
            C_max = band[1]
            X_new = band[2]
            
            #Pour l'algo split, on sépare les points d'entrainements des points de contrôle 
            if (prd_alg.__name__ == 'conf_band_split') or (prd_alg.__name__ == 'conf_band_split_LW'):
                I_train = band[3]
                I_ctrl = band[4]
                
                X_train = np.array([X[i] for i in I_train])
                Y_train = np.array([Y[i] for i in I_train])

                X_ctrl = np.array([X[i] for i in I_ctrl])
                Y_ctrl = np.array([Y[i] for i in I_ctrl])
                
                axs[i][j].scatter(X_train,Y_train,s=10, alpha=0.5, color = 'blue',label = 'train data')
                axs[i][j].scatter(X_ctrl,Y_ctrl,s=10, alpha=0.5, color = 'green', label = 'control data')
            
            else:
                axs[i][j].scatter(X,Y, s=10, alpha=0.5, label = 'data')
            

            axs[i][j].plot(X_reg,Y_reg, color='black',linestyle = 'dotted', label = 'regression')

            axs[i][j].plot(X_new,C_min,color='red', label = 'prediction band')
            axs[i][j].plot(X_new,C_max,color='red')
            axs[i][j].fill_between(X_new, C_min, C_max, alpha=0.1, color='red')
            

            percent = int(100*(i*l + j + 1)/(k*l))
            if prd_alg.__name__ == 'conf_band_full':
                print(f'{percent}%')


    plt.suptitle(f'n={len(X)}, alpha={a}, reg={reg_fct.__name__}, pred={prd_alg.__name__}')
    plt.show()











def plot_band_latex(settings:list,
                   a:float,
                   reg_fct:Callable[[np.ndarray,np.ndarray], Callable[[np.ndarray], float]],
                   prd_alg:Callable[[np.ndarray,np.ndarray,float,Callable[[np.ndarray,np.ndarray], Callable[[np.ndarray], float]],np.ndarray|None], tuple]) -> None:
    """
    Affiche plusieurs bandes de prédiction conforme pour différentes configurations de données. Utilisé pour une figure dans le rapport de stage.
    
    Paramètres
    ----------
    **list** settings : Liste de fonctions de configuration qui génèrent les données X et Y
    **float** a : Le niveau de signification pour la prédiction conforme.
    **Callable** reg_fct : Une fonction de régression qui prend X et Y et
        retourne une fonction entraînée sur les données.
    **Callable** prd_alg : Une fonction de prédiction conforme qui prend X, Y, a, reg_fct et X_new (optionnel) et retourne un tuple contenant les born
    """

    
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"  # facultatif
    })
    
    
    n = 1000
    d = 1
    
    fig, axs = plt.subplots(2,2)
    
    for i in range(2):
        for j in range(2):
            if (i==1) and (j==1):
                reg_fct = regalg_spline
            
            setting = settings[2*i+j]
            X,Y = setting(n,d)
            
            X_new = np.linspace(X.min(), X.max(), 100)

            reg_fct_fct = reg_fct(X,Y)
            X_reg = np.linspace(min(X.min(),X_new.min()), max(X.max(),X_new.max()), 1000)
            Y_reg = np.array([reg_fct_fct(x) for x in X_reg])    

            band = prd_alg(X,Y,a,reg_fct,X_new)

            C_min = band[0]
            C_max = band[1]
            X_new = band[2]
            
            if (prd_alg.__name__ == 'conf_band_split') or (prd_alg.__name__ == 'conf_band_split_LW'):
                I_train = band[3]
                I_ctrl = band[4]
                
                X_train = np.array([X[i] for i in I_train])
                Y_train = np.array([Y[i] for i in I_train])

                X_ctrl = np.array([X[i] for i in I_ctrl])
                Y_ctrl = np.array([Y[i] for i in I_ctrl])
                
                axs[i][j].scatter(X_train,Y_train,s=10, alpha=0.5, color = 'blue', label = 'train data')
                axs[i][j].scatter(X_ctrl,Y_ctrl,s=10, alpha=0.5, color = 'green', label = 'control data')
            
            else:
                axs[i][j].scatter(X,Y, s=10, alpha=0.5, label = 'data')
            
                axs[i][j].plot(X_reg,Y_reg, color='black',linestyle = 'dotted', label = 'regression')

            axs[i][j].plot(X_new,C_min,color='red', label = 'prediction band')
            axs[i][j].plot(X_new,C_max,color='red')
            axs[i][j].fill_between(X_new, C_min, C_max, alpha=0.1, color='red')

            if (i==1) and (j==1):
                axs[i][j].set_title(rf'$n={len(X)}, alpha={a}, reg=Spline, pred=Full Conformal$')
            else:
                axs[i][j].set_title(rf'$n={len(X)}, alpha={a}, reg=Lasso, pred=Full Conformal$')

            axs[i][j].legend()
    
    plt.show()