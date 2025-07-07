import numpy as np
import matplotlib.pyplot as plt

import RegAlgs as ra
import ExpSetups as es
import AlgoPrediction as ap
import Affichage as tool
import Control as ctrl

#seed = 1337


d= 1
n=1000
a=.1


settings = [es.setting_A,es.setting_B,es.setting_C,es.setting_P5]
setting = es.setting_B
reg_fct = ra.regalg_spline
prd_alg = ap.conf_band_split

X,Y = setting(n,d)
X_new = np.linspace(X.min(),X.max(),100)
C_fct = ap.split_conformal_prediction(X,Y,a,reg_fct)[0]




p = ctrl.coverage_test_split(C_fct, setting, n, d)
print(p)

band = prd_alg(X,Y,a,reg_fct,X_new)
plt.scatter(X,Y,alpha=.3)
plt.plot(X_new,band[0],color='red')
plt.plot(X_new,band[1],color='red')
plt.show()


