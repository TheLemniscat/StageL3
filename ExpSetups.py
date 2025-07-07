import numpy as np
from scipy.stats import t, skewnorm, bernoulli
from patsy import dmatrix

import matplotlib.pyplot as plt

def X_setting_A(n:int,d:int) -> np.ndarray:
    """
    Simule un jeu de données X suivant le setting A.   
    
    Paramètres
    ----------
    **int** n : La taille de l'échantillon.
    **int** d : La dimension de chaque données.

    Returns
    -------
    np.ndarray : Un tableau de forme (n, d) contenant les données simulées.
    """

    return np.random.normal(0,1,size=(n, d))

def eps_setting_A(n:int) -> np.ndarray:
    """
    Simule un bruit aléatoire suivant le setting A.

    Paramètres
    ----------
    **int** n : La taille de l'échantillon.
    
    Returns
    -------
    np.ndarray : Un tableau de taille n contenant des valeurs aléatoires suivant une loi normale standard.
    """

    return np.random.normal(0,1,size= n)

def Y_setting_A(X:np.ndarray,eps:np.ndarray) -> np.ndarray:
    """
    Calcule la variable expliquée Y en fonction de X et du bruit eps suivant le Setting A.

    Paramètres
    ----------
    **np.ndarray** X : Un tableau de forme (n, d) contenant les données d'entrée.
    **np.ndarray** eps : Un tableau de taille n contenant le bruit aléatoire à ajouter à la somme des éléments
        de chaque ligne de X.

    Returns
    -------
    **np.ndarray** : Un tableau de taille n contenant les valeurs de Y, où chaque valeur est la somme des
        éléments de la ligne correspondante de X plus le bruit eps.
    """

    n = len(eps)
    return np.array([np.sum(X[i])+ eps[i] for i in range(n)])

def setting_A(n:int,d:int) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère un jeu de données X et Y suivant le Setting A.
    
    Paramètres
    ----------
    **int** n : La taille de l'échantillon.
    **int** d : La dimension de chaque donnée.

    Returns
    -------
    tuple : Un tuple contenant deux éléments :
        - **np.ndarray** X: tableau de forme (n, d) contenant les données d'entrée.
        - **np.ndarray** Y: tableau de taille n contenant les valeurs de la variable expliquée.
    """

    X = X_setting_A(n,d)
    eps = eps_setting_A(n)
    Y = Y_setting_A(X,eps)
    return X,Y











def X_setting_B(n:int,d:int) -> np.ndarray:
    """
    Simule un jeu de données X suivant le setting B.   
    
    Paramètres
    ----------
    **int** n : La taille de l'échantillon.
    **int** d : La dimension de chaque données.

    Returns
    -------
    np.ndarray : Un tableau de forme (n, d) contenant les données simulées.
    """

    return np.random.normal(0,1,size=(n, d))

def eps_setting_B(n):
    """
    Simule un bruit aléatoire suivant le setting B.

    Paramètres
    ----------
    **int** n : La taille de l'échantillon.
    
    Returns
    -------
    np.ndarray : Un tableau de taille n contenant des valeurs aléatoires suivant une loi normale standard.
    """

    return np.array(t.rvs(df=2, size = n))

def Y_setting_B(X:np.ndarray, eps:np.ndarray, coefs_list:list[float]=[]) -> np.ndarray:
    """
    Calcule la variable expliquée Y en fonction de X et du bruit eps suivant le Setting B.
    Les coefficients des splines sont générés aléatoirement si coefs_list n'est pas fourni.

    Paramètres
    ----------
    **np.ndarray** X : Un tableau de forme (n, d) contenant les données d'entrée.
    **np.ndarray** eps : Un tableau de taille n contenant le bruit aléatoire à ajouter à la somme des éléments
        de chaque ligne de X.
    **list** coefs_list : Une liste de coefficients pour les splines. Si la liste est vide, les coefficients sont générés aléatoirement.
        
    Returns
    -------
    **np.ndarray** : Un tableau de taille n contenant les valeurs de Y suivant le setting B.
    """
    
    n, d = np.shape(X)

    degree = 3
    n_splines = 5

    # Si coefs_list n'est pas fourni, on le génère
    if len(coefs_list) == 0:
        coefs_list = []
        for j in range(d):
            xj = X[:, j]
            design_matrix = dmatrix(
                f"bs(x, df={n_splines}, degree={degree}, include_intercept=False) - 1",
                {"x": xj},
                return_type='dataframe'
            ).values
            coefs = np.random.uniform(-2, 2, size=design_matrix.shape[1])
            coefs_list.append(coefs)
    # Calcul de mu(x) avec les coefs_list donnés
    mu = np.zeros(n)
    for j in range(d):
        xj = X[:, j]
        design_matrix = dmatrix(
            f"bs(x, df={n_splines}, degree={degree}, include_intercept=False) - 1",
            {"x": xj},
            return_type='dataframe'
        ).values
        mu += design_matrix @ coefs_list[j]

    return np.array(mu + eps)


def Coefs_setting_B(d:int, n_splines:int=5, degree:int=3, low:int=-2, high:int=2) -> list[np.ndarray]:
    """
    Génère des coefficients aléatoires pour les splines utilisées dans le Setting B.

    Paramètres
    ----------
    **int** d : La dimension de chaque donnée.
    **int** n_splines : Le nombre de splines à générer pour chaque dimension.
    **int** degree : Le degré des splines.
    **int** low : La borne inférieure pour les coefficients aléatoires.
    **int** high : La borne supérieure pour les coefficients aléatoires.
    
    Returns
    -------
    list : Une liste de tableaux numpy, où chaque tableau contient les coefficients pour une dimension.
    """
    return [np.random.uniform(low, high, size=n_splines) for _ in range(d)]


def setting_B(n:int,d:int,coefs:list[float]=[]) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère un jeu de données X et Y suivant le Setting B.
    
    Paramètres
    ----------
    **int** n : La taille de l'échantillon.
    **int** d : La dimension de chaque donnée.
    **list** coefs_list : Une liste de coefficients pour les splines. Si la liste est vide, les coefficients sont générés aléatoirement.

    Returns
    -------
    tuple : Un tuple contenant deux éléments :
        - **np.ndarray** X: tableau de forme (n, d) contenant les données d'entrée.
        - **np.ndarray** Y: tableau de taille n contenant les valeurs de la variable expliquée.
    """    
    X = X_setting_B(n,d)
    eps = eps_setting_B(n)
    Y = np.array(Y_setting_B(X,eps,coefs_list=coefs))
    return X,Y










def X_setting_C(n:int, d:int) -> np.ndarray:
    """
    Simule un jeu de données X suivant le setting A.   
    
    Paramètres
    ----------
    **int** n : La taille de l'échantillon.
    **int** d : La dimension de chaque données.

    Returns
    -------
    np.ndarray : Un tableau de forme (n, d) contenant les données simulées.
    """
    X = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            dist_choice = np.random.choice(["normal", "skew", "bernoulli"])
            if dist_choice == "normal":
                X[i, j] = np.random.normal(0, 1)
            elif dist_choice == "skew":
                X[i, j] = skewnorm.rvs(a=5, loc=0, scale=1)
            elif dist_choice == "bernoulli":
                X[i, j] = bernoulli.rvs(p=0.5)
    
    # Autocorrélation entre les composantes
    for i in range(n):
        for j in range(1, d):
            lookback = min(j, 3)
            weights = np.linspace(1, 0.5, lookback)
            past = X[i, j - lookback:j]
            if len(past) > 0:
                w_sum = np.sum(weights)
                X[i, j] = 0.5 * X[i, j] + 0.5 * np.sum(past * weights[::-1]) / w_sum
    return X

def eps_setting_C(n:int,X:np.ndarray) -> np.ndarray:
    """
    Simule un bruit aléatoire suivant le setting A.

    Paramètres
    ----------
    **int** n : La taille de l'échantillon.
    **np.ndarray** X : Un tableau de forme (n, d) contenant les données d'entrée.

    Returns
    -------
    np.ndarray : Un tableau de taille n contenant des valeurs aléatoires suivant une loi normale standard.
    """
        
    eps = np.array(t.rvs(df=2, size = n))
    tmp = []
    # Calcul de l'écart type sigma pour chaque ligne de X
    for i in range(n):
        mu_abs_cube = np.abs(np.sum(X[i])) ** 3
        expected = np.mean(np.abs(X)**3)
        sigma = 1+2*mu_abs_cube/expected
        tmp.append(eps[i]*sigma)
    
    return np.array(tmp)

def Y_setting_C(X:np.ndarray,eps:np.ndarray) -> np.ndarray:
    """
    Calcule la variable expliquée Y en fonction de X et du bruit eps suivant le Setting C.

    Paramètres
    ----------
    **np.ndarray** X : Un tableau de forme (n, d) contenant les données d'entrée.
    **np.ndarray** eps : Un tableau de taille n contenant le bruit aléatoire à ajouter à la somme des éléments
        de chaque ligne de X.

    Returns
    -------
    **np.ndarray** : Un tableau de taille n contenant les valeurs de Y suivant le setting C.
    """
   
    n = len(eps)
    Y = np.array([np.sum(X[i]) + eps[i] for i in range(n)])
    return Y

def setting_C(n:int, d:int) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère un jeu de données X et Y suivant le Setting C.
    
    Paramètres
    ----------
    **int** n : La taille de l'échantillon.
    **int** d : La dimension de chaque donnée.

    Returns
    -------
    tuple : Un tuple contenant deux éléments :
        - **np.ndarray** X: tableau de forme (n, d) contenant les données d'entrée.
        - **np.ndarray** Y: tableau de taille n contenant les valeurs de la variable expliquée.
        
    """
    
    X = X_setting_C(n, d)
    eps = eps_setting_C(n,X)
    Y = Y_setting_C(X,eps)
    
    return X,Y










def X_setting_P5(n:int,d:int) -> np.ndarray:
    """
    Simule un jeu de données X suivant le setting P5.   
    
    Paramètres
    ----------
    **int** n : La taille de l'échantillon.
    **int** d : La dimension de chaque données, il n'est pas utilisé dans ce setting.

    Returns
    -------
    np.ndarray : Un tableau de forme (n, d) contenant les données simulées.
    """
    return np.random.uniform(0,2*np.pi,size=(n))

def eps_setting_P5(n):
    """
    Simule un bruit aléatoire suivant le setting P5.

    Paramètres
    ----------
    **int** n : La taille de l'échantillon.
    **np.ndarray** X : Un tableau de forme (n, d) contenant les données d'entrée.

    Returns
    -------
    np.ndarray : Un tableau de taille n contenant des valeurs aléatoires suivant une loi normale standard.
    """
    return np.random.normal(0,1,size= n)

def Y_setting_P5(X:np.ndarray, eps:np.ndarray) -> np.ndarray:
    """
    Calcule la variable expliquée Y en fonction de X et du bruit eps suivant le Setting P5.

    Paramètres
    ----------
    **np.ndarray** X : Un tableau de forme (n, d) contenant les données d'entrée.
    **np.ndarray** eps : Un tableau de taille n contenant le bruit aléatoire à ajouter à la somme des éléments
        de chaque ligne de X.

    Returns
    -------
    **np.ndarray** : Un tableau de taille n contenant les valeurs de Y suivant le setting P5.
    """
    return np.sin(X) + np.pi * np.abs(X) * eps/20

def setting_P5(n:int, d:int) -> tuple[np.ndarray, np.ndarray]:
    X = X_setting_P5(n,d)
    eps = eps_setting_P5(n)
    Y = Y_setting_P5(X,eps)
    """
    Génère un jeu de données X et Y suivant le Setting P5.
    
    Paramètres
    ----------
    **int** n : La taille de l'échantillon.
    **int** d : La dimension de chaque donnée.

    Returns
    -------
    tuple : Un tuple contenant deux éléments :
        - **np.ndarray** X: tableau de forme (n, d) contenant les données d'entrée.
        - **np.ndarray** Y: tableau de taille n contenant les valeurs de la variable expliquée.
        
    """
    return X,Y










def setting_Trivial(n:int, d:int) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère un jeu de données X et Y où X est un linspace et Y est une fonction linéaire de X avec un bruit aléatoire.
    
    Paramètres
    ----------
    **int** n : La taille de l'échantillon.
    **int** d : La dimension de chaque donnée, il n'est pas utilisé dans ce setting.
    
    Returns
    -------
    tuple : Un tuple contenant deux éléments :
        - **np.ndarray** X: tableau de taille n contenant les données d'entrée.
        - **np.ndarray** Y: tableau de taille n contenant les valeurs de la variable expliquée.
    """
    X =  np.linspace(0,10,n)
    eps = np.random.normal(0,1,n)
    Y = 3*X+eps

    return X,Y




    
if __name__ == "__main__":

    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"  # facultatif
    })
    
    n = 1000
    d = 1
    X,Y = setting_B(n,d)
    plt.scatter(X,Y,alpha=.5)
    plt.title(r'Setting B')
    #plt.savefig('SettingB.png')
    plt.show()