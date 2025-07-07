import numpy as np

from patsy import dmatrix
from sklearn.linear_model import ElasticNet, LinearRegression, Lasso
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline

from typing import Callable


# Je n'ai pas implémenté stepwise et SPAM
# Stepwise n'est pas utilisable en une dimensionndarray
# SPAM semble plus compliqué



def regalg_lasso(X:np.ndarray,Y:np.ndarray,alpha:float=0.1) -> Callable[[np.ndarray], float]:
    """
    Effectue une régression Lasso sur les données X et Y.

    Paramètres
    ----------
    **np.ndarray** X : Tableau de forme (n, 1) contenant les données
    **np.ndarray** Y : Tableau de taille n contenant les valeurs de la variable expliquée.
    **float** alpha : Paramètre de régularisation pour la régression Lasso (par défaut 0.1).

    Returns
    -------
    **function** : Une fonction qui prend un argument x et retourne la prédiction de la régression Lasso pour cette valeur.
    """
    
    X = X.reshape(-1, 1)
    model = Lasso(alpha).fit(X, Y)
    def out(x):
        return model.predict(np.array([[float(x)]]))[0]
    return out





def regalg_elasticnet(X:np.ndarray,Y:np.ndarray,alpha:float=1.0,l1_ratio:float=0.5) -> Callable[[np.ndarray], float]:
    """
    Effectue une régression ElasticNet sur les données X et Y.

    Paramètres
    ----------
    **np.ndarray** X : Tableau de forme (n, 1) contenant les données
    **np.ndarray** Y : Tableau de taille n contenant les valeurs de la variable expliquée.
    **float** alpha : Le paramètre de régularisation pour la régression ElasticNet (par défaut 1.0).
    **float** l1_ratio : Le ratio de régularisation L1 pour la régression ElasticNet (par défaut 0.5).

    Returns
    -------
    **function** : Une fonction qui prend un argument x et retourne la prédiction de la régression Lasso pour cette valeur.
    """

    X = X.reshape(-1, 1)
    model = ElasticNet(alpha, l1_ratio=l1_ratio)
    model.fit(X, Y)
    def out(x):
        return model.predict(np.array([[float(x)]]))[0]
    return out






def regalg_rf(X:np.ndarray,Y:np.ndarray) -> Callable[[np.ndarray], float]:
    """
    Effectue une régression par forêt aléatoire sur les données X et Y.

    Paramètres
    ----------
    **np.ndarray** X : Tableau de forme (n, 1) contenant les données
    **np.ndarray** Y : Tableau de taille n contenant les valeurs de la variable expliquée.

    Returns
    -------
    **function** : Une fonction qui prend un argument x et retourne la prédiction de la régression par forêt aléatoire pour cette valeur.
    """
    X = X.reshape(-1, 1)
    Y = Y.ravel()
    model = RandomForestRegressor(n_estimators=100).fit(X, Y)
    def out(x):
        return model.predict(np.array([[float(x)]]))[0]
    return out






def regalg_spline(X:np.ndarray,Y:np.ndarray) -> Callable[[np.ndarray], float]:
    """
    Effectue une régression spline sur les données X et Y.

    Paramètres
    ----------
    **np.ndarray** X : Tableau de forme (n, 1) contenant les données
    **np.ndarray** Y : Tableau de taille n contenant les valeurs de la variable expliquée.
    
    Returns
    -------
    **function** : Une fonction qui prend un argument x et retourne la prédiction de la régression spline pour cette valeur.
    """


    X = X.reshape(-1,1)
    model = make_pipeline(SplineTransformer(degree=3, n_knots=6), LinearRegression())
    model.fit(X, Y)
    def spline_fct(x):
        return float(model.predict(np.array([[float(x)]]))[0])
    return spline_fct