import numpy as np

from typing import Callable


def full_conformal_prediction(X:np.ndarray,
                              Y:np.ndarray,
                              alpha:float,
                              regalg:Callable[[np.ndarray,np.ndarray], Callable[[np.ndarray], float]], 
                              X_new:np.ndarray|None=None, 
                              Y_trial:np.ndarray|None=None, 
                              avancement:bool=False) -> list:
    
    """
    Effectue une prédiction conforme complète sur les données X et Y.

    Paramètres
    ----------
    **np.ndarray** X : Tableau de forme (n, 1) contenant les echantillons de la variable explicative.
    **np.ndarray** Y : Tableau de taille n contenant les valeurs de la variable expliquée.
    **float** alpha : Le niveau de signification pour la prédiction conforme.
    **Callable** regalg : Une fonction de régression qui prend X et Y et retourne
        une fonction entraînée sur les données.
    **np.ndarray** X_new : Tableau de taille m contenant les nouvelles valeurs pour lesquelles on
        souhaite prédire les intervalles de confiance. Si None, il sera généré automatiquement.
    **np.ndarray** Y_trial : Tableau de taille p contenant les valeurs candidates pour les intervalles de confiance.
        Si None, il sera généré automatiquement.
    **bool** avancement : Si True, affiche l'avancement du calcul.

    Returns
    -------
    **list** : Une liste de listes contenant les intervalles de confiance pour chaque valeur de
        X_new. Chaque sous-liste contient les valeurs de Y_trial qui sont dans l'intervalle de confiance
        pour la valeur correspondante de X_new.
    """


    if Y_trial is None:
        Y_trial = np.linspace(Y.min(),Y.max(),5)

    if X_new is None:
        X_new = np.linspace(X.min(),X.max(),5)

    # Juste pour la verification des type 
    X_new = np.array(X_new)
    Y_trial = np.array(Y_trial)

    n = np.shape(X)[0]
    conf = []


    percentage = 0

    for x in X_new:
        conf_x = []
        for y in Y_trial:
            X_tmp = np.append(X,x)
            Y_tmp = np.append(Y,y)
            mu_y = regalg(X_tmp,Y_tmp)
            
            R_y = [abs(Y[i]-mu_y(X[i])) for i in range(n)]
            R_y.append(abs(y-mu_y(x)))

            pi_y = 1/(n+1)
            for i in range(n):
                if R_y[i] <= R_y[n]:
                    pi_y += 1/(n+1)
            
            if (n+1)*pi_y <= np.ceil((1-alpha)*(n+1)):
                conf_x.append(float(y))

            if avancement:
                percentage += 100/(len(X_new)*len(Y_trial))
                print(int(percentage))
        
        conf.append(conf_x.copy())
    
    return conf


def conf_band_full(X:np.ndarray,
                  Y:np.ndarray,
                  alpha:float,
                  regalg:Callable[[np.ndarray,np.ndarray], Callable[[np.ndarray], float]],
                  X_new:np.ndarray|None=None,
                  Y_trial:np.ndarray|None=None,
                  avancement:bool=False) -> tuple:
    """
    Effectue une bande de prédiction conforme complète sur les données X et Y.

    Paramètres
    ----------
    **np.ndarray** X : Tableau de forme (n, 1) contenant les échantillons de la variable explicative.
    **np.ndarray** Y : Tableau de taille n contenant les valeurs de la variable expliquée
    **float** alpha : Le niveau de signification pour la prédiction conforme.
    **Callable** regalg : Une fonction de régression qui prend X et Y et retourne
        une fonction entraînée sur les données.
    **np.ndarray** X_new : Tableau de taille m contenant les nouvelles valeurs pour lesquelles on
        souhaite prédire les intervalles de confiance. Si None, il sera généré automatiquement.
    **np.ndarray** Y_trial : Tableau de taille p contenant les valeurs candidates pour les intervalles de confiance.
        Si None, il sera généré automatiquement.
    **bool** avancement : Si True, affiche l'avancement du calcul.

    Returns
    -------
    **tuple** : Un tuple contenant trois éléments :
        - **np.ndarray** C_min: Tableau de taille m contenant les bornes inférieures des intervalles de confiance pour chaque valeur de X_new.
        - **np.ndarray** C_max: Tableau de taille m contenant les bornes supérieures des intervalles de confiance pour chaque valeur de X_new.
        - **np.ndarray** X_new: Tableau de taille m contenant les nouvelles valeurs pour lesquelles on a prédit les intervalles de confiance.
    """
    if Y_trial is None:
        Y_trial = np.linspace(Y.min(),Y.max(),25)

    C_pred = full_conformal_prediction(X, Y, alpha, regalg, X_new, Y_trial, avancement)
    C_min = np.array([min(l) if len(l)>0 else None for l in C_pred])
    C_max = np.array([max(l) if len(l)>0 else None for l in C_pred])
    return C_min,C_max,X_new   










def split_conformal_prediction(X:np.ndarray,
                              Y:np.ndarray,
                              alpha:float,
                              regalg:Callable[[np.ndarray,np.ndarray], Callable[[np.ndarray], float]]) -> tuple:
    """
    Effectue une prédiction conforme par la méthode de séparation sur les données X et Y.
    
    Paramètres
    ----------
    **np.ndarray** X : Tableau de forme (n, 1) contenant les échantillons de la variable explicative.
    **np.ndarray** Y : Tableau de taille n contenant les valeurs de la variable expliquée
    **float** alpha : Le niveau de signification pour la prédiction conforme.
    **Callable** regalg : Une fonction de régression qui prend X et Y et retourne
        une fonction entraînée sur les données.
    
    Returns
    -------
    **tuple** : Un tuple contenant trois éléments :
        - **function** C_split_fct : Une fonction qui prend un argument x et retourne
            les bornes inférieure et supérieure de la bande de prédiction pour cette valeur.
        - **np.ndarray** I1 : Un tableau d'indices représentant l'ensemble d'entraînement.
        - **np.ndarray** I2 : Un tableau d'indices représentant l'ensemble de contrôle
    """
    


    n = len(X)
    
    # Génère I1 et I2
    idx = np.random.permutation(n)
    n1 = n // 2
    I1 = idx[:n1]
    I2 = idx[n1:]

    X_train = np.array([X[i] for i in I1])
    Y_train = np.array([Y[i] for i in I1])

    mu_hat = regalg(X_train,Y_train)

    # Calcul les résidus sur I2
    R = [abs(Y[i] - mu_hat(X[i])) for i in I2]

    # Calcul la largeur de C
    k = int(np.ceil((n/2 + 1) * (1 - alpha)))
    R_sorted = np.sort(R)
    split_width = R_sorted[k-1] 

    # Retrun la bande de prédiction comme une fonction
    def C_split_fct(x):
        mu = mu_hat(x)
        return [mu - split_width, mu + split_width]
    return C_split_fct,I1,I2



def conf_band_split(X:np.ndarray,
                   Y:np.ndarray,
                   alpha:float,
                   regalg:Callable[[np.ndarray,np.ndarray], Callable[[np.ndarray], float]],
                   X_new:np.ndarray|None=None) -> tuple:
    """
    Effectue une bande de prédiction conforme par la méthode de séparation sur les données X et Y.

    Paramètres
    ----------
    **np.ndarray** X : Tableau de forme (n, 1) contenant les é
    **np.ndarray** Y : Tableau de taille n contenant les valeurs de la variable expliquée
    **float** alpha : Le niveau de signification pour la prédiction conforme.
    **Callable** regalg : Une fonction de régression qui prend X et Y et retourne
        une fonction entraînée sur les données.
    **np.ndarray** X_new : Tableau de taille m contenant les nouvelles valeurs pour lesquelles on
        souhaite prédire les intervalles de confiance. Si None, il sera généré automatiquement.
    
    Returns
    -------
    **tuple** : Un tuple contenant cinq éléments :
        - **np.ndarray** C_min: Tableau de taille m contenant les bornes inférieures des intervalles de confiance pour chaque valeur de X_new.
        - **np.ndarray** C_max: Tableau de taille m contenant les bornes supérieures des intervalles de confiance pour chaque valeur de X_new.
        - **np.ndarray** X_new: Tableau de taille m contenant les nouvelles valeurs pour lesquelles on a prédit les intervalles de confiance.
    """

    if X_new is None:
        X_new = np.linspace(X.min(), X.max(), 5)
    # Juste pour la verification des type
    X_new = np.array(X_new)

    tmp = split_conformal_prediction(X, Y, alpha, regalg)
    C_pred_fct = tmp[0]
    I_train = tmp[1]
    I_ctrl = tmp[2]

    C_min = [np.min(C_pred_fct(x)) for x in X_new]
    C_max = [np.max(C_pred_fct(x)) for x in X_new]
    return C_min, C_max, X_new, I_train, I_ctrl











def split_conformal_prediction_LW(X:np.ndarray,
                                  Y:np.ndarray,
                                  alpha:float,
                                  regalg:Callable[[np.ndarray,np.ndarray], Callable[[np.ndarray], float]]) -> tuple:
    """
    Effectue une prédiction conforme par la méthode de séparation avec pondération locale sur les données X et Y.

    Paramètres
    ----------
    **np.ndarray** X : Tableau de forme (n, 1) contenant les échantillons de la variable explicative.
    **np.ndarray** Y : Tableau de taille n contenant les valeurs de la variable expliquée
    **float** alpha : Le niveau de signification pour la prédiction conforme.
    **Callable** regalg : Une fonction de régression qui prend X et Y et retourne
        une fonction entraînée sur les données.
    
    Returns
    -------
    **tuple** : Un tuple contenant trois éléments :
        - **function** C_split_fct : Une fonction qui prend un argument x et retourne
            les bornes inférieure et supérieure de la bande de prédiction pour cette valeur.
        - **np.ndarray** I1 : Un tableau d'indices représentant l'ensemble d'entraînement.
        - **np.ndarray** I2 : Un tableau d'indices représentant l'ensemble de contrôle
    """

    

    n = len(X)
    idx = np.random.permutation(n)
    n1 = n // 2
    I1 = idx[:n1]
    I2 = idx[n1:]

    X_train = np.array([X[i] for i in I1])
    Y_train = np.array([Y[i] for i in I1])
    
    mu_hat = regalg(X_train, Y_train)

    # Estimation des résidus absolus pour le MAD local
    R = np.array([abs(Y[i] - mu_hat(X[i])) for i in I2])
    X_ctrl = np.array([X[i] for i in I2])

    # Estimation naïve de la fonction MAD locale par une régression linéaire sur les résidus absolus
    mad_model = regalg(X_ctrl, R)
    def rho_hat(x):
        return mad_model(x)

    # Résidus pondérés
    R_weighted = [R[i] / rho_hat(X_ctrl[i]) for i in range(n//2)]

    k = int(np.ceil((n1 + 1) * (1 - alpha)))
    R_sorted = np.sort(R_weighted)
    split_width = R_sorted[k - 1]

    def C_split_fct(x):
        mu = mu_hat(x)
        rho = rho_hat(x)
        return [mu - split_width * rho, mu + split_width * rho]

    return C_split_fct, I1, I2


def conf_band_split_LW(X:np.ndarray,
                       Y:np.ndarray,
                       alpha:float,
                       regalg:Callable[[np.ndarray,np.ndarray], Callable[[np.ndarray], float]],
                       X_new:np.ndarray|None=None) -> tuple:
    """
    Effectue une bande de prédiction conforme par la méthode de séparation avec pondération locale sur les données X et Y.
    
    Paramètres
    ----------
    **np.ndarray** X : Tableau de forme (n, 1) contenant les échantillons de la variable explicative.
    **np.ndarray** Y : Tableau de taille n contenant les valeurs de la variable expliquée
    **float** alpha : Le niveau de signification pour la prédiction conforme.
    **Callable** regalg : Une fonction de régression qui prend X et Y et retourne
        une fonction entraînée sur les données.
    **np.ndarray** X_new : Tableau de taille m contenant les nouvelles valeurs pour lesquelles on
        souhaite prédire les intervalles de confiance. Si None, il sera généré automatiquement.
    
    Returns
    -------
    **tuple** : Un tuple contenant cinq éléments :
        - **np.ndarray** C_min: Tableau de taille m contenant les bornes inférieures des intervalles de confiance pour chaque valeur de X_new.
        - **np.ndarray** C_max: Tableau de taille m contenant les bornes supérieures des intervalles de confiance pour chaque valeur de X_new.
        - **np.ndarray** X_new: Tableau de taille m contenant les nouvelles valeurs pour lesquelles on a prédit les intervalles de confiance.
    """

    if X_new is None:
        X_new = np.linspace(X.min(), X.max(), 100)
    # Juste pour la verification des type
    X_new = np.array(X_new)


    tmp = split_conformal_prediction_LW(X, Y, alpha, regalg)
    C_pred_fct = tmp[0]
    I_train = tmp[1]
    I_ctrl = tmp[2]

    C_min = [np.min(C_pred_fct(x)) for x in X_new]
    C_max = [np.max(C_pred_fct(x)) for x in X_new]
    return C_min, C_max, X_new, I_train, I_ctrl