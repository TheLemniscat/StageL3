import numpy as np

import RegAlgs as ra
import ExpSetups as es
import AlgoPrediction as ap







def coverage_test_split(C_fct, setting, n:int, d:int, coefs:list[np.ndarray]=[]):
    """
    Test de couverture pour une bande de prédiction conforme, générée par le split conformal, sur un ensemble de données de validation.

    Paramètres
    ----------
    **C_fct** : Fonction de bande de prédiction conforme.
    **setting** : Fonction de configuration qui génère les données d'entraînement et de validation
    **n** : Nombre d'échantillons dans l'ensemble de validation.
    **d** : Dimension des données.
    **coefs** : Coefficients pour la configuration des données (optionnel).

    Returns
    -------
    **float** : Pourcentage de couverture de la bande de prédiction conforme sur l'ensemble de validation.
    """
    
    
    if len(coefs) == 0:
        X_ctrl,Y_ctrl = setting(n,d)
    else:
        X_ctrl,Y_ctrl = setting(n,d,coefs)

    percent = 0
    for i in range(n):
        band = C_fct(X_ctrl[i])
        C_min = band[0]
        C_max = band[1]
        if C_min < Y_ctrl[i]:
            if C_max > Y_ctrl[i]:
                percent += 1/(n+1)
    
    return percent*100


def coverage_test_final_split():

    d=1
    n=1000
    a=.1

    out = {}

    reg_fct = ra.regalg_spline
    prd_alg = ap.split_conformal_prediction
    setting_list = [es.setting_A, es.setting_B, es.setting_C, es.setting_P5, es.setting_Trivial]

    nb_iteration = 10
    nb_parameter = len(setting_list)
    
    avancement = 0


    for s in range(nb_parameter):
        setting = setting_list[s]
        l_tmp = []
        for i in range(nb_iteration):
            X,Y = setting(n,d)
            C_fct = prd_alg(X,Y,a,reg_fct)[0]
            percent = coverage_test_split(C_fct,setting,n,d)
            l_tmp.append(percent)

            avancement += 100/(nb_parameter*nb_iteration)
            print(f'{int(avancement)}%')

        tmp = np.array(l_tmp)
        out[setting.__name__] = float(np.mean(tmp))
    
    print(out)



def coverage_test_setting(setting):
  
    d=1
    n=1000
    a=.1

    reg_fct = ra.regalg_spline
    prd_alg = ap.split_conformal_prediction

    nb_iteration = 100
    
    avancement = 0

    l_tmp = []
    for i in range(nb_iteration):
        coefs = es.Coefs_setting_B(d)
        X,Y = setting(n,d,coefs)
        C_fct = prd_alg(X,Y,a,reg_fct)[0]
        percent = coverage_test_split(C_fct,setting,n,d,coefs)
        l_tmp.append(percent)

        avancement += 100/nb_iteration
        print(f'{int(avancement)}%')

    out = np.array(l_tmp)
    
    print(float(np.mean(out)))  




def coverage_test_full(setting, n:int, d:int, a:float, coefs:np.ndarray=np.array([])):
    """
    Test de couverture pour une bande de prédiction conforme, générée par le full conformal, sur un ensemble de données de validation.

    Paramètres
    ----------
    **setting** : Fonction de configuration qui génère les données d'entraînement et de validation
    **n** : Nombre d'échantillons dans l'ensemble de validation.
    **d** : Dimension des données.
    **a** : Niveau de signification pour la prédiction conforme.
    **coefs** : Coefficients pour la configuration des données (optionnel).
    
    Returns
    -------
    **float** : Pourcentage de couverture de la bande de prédiction conforme sur l'ensemble de validation.
    """
    
    if coefs is None:
        X,Y = setting(n,d)
        X_ctrl,Y_ctrl = setting(n,d)
    else:
        X,Y = setting(n,d,coefs)
        X_ctrl,Y_ctrl = setting(n,d,coefs)

    Y_trial = np.linspace(Y.min()-1,Y.max()+1,5)

    band = ap.conf_band_full(X,Y,a,ra.regalg_spline,X_ctrl,Y_trial,avancement=True)
    C_min_list = band[0]
    C_max_list = band[1]

    percent = 0
    for i in range(n):
        C_min = C_min_list[i]
        C_max = C_max_list[i]
        if C_min is not None:
            if C_min < Y_ctrl[i]:
                if C_max > Y_ctrl[i]:
                    percent += 1/(n+1)
        
    print(percent*100) 



if __name__ == "__main__":
      
    d=1
    n=100
    a=.5

    coverage_test_full(es.setting_B,n,d,a)