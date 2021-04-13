import numpy as np
from scipy.stats import spearmanr, kendalltau

def logit(p):
    return np.log(p) - np.log(1 - p)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def spearman_kendall(player_coef, data, team_estimator):
    spearmans, kendalls = [], []
    for team_list in data:
        y_real = [team['position'] for team in team_list]
        #y_real = [team['questionsTotal'] for team in team_list]
        y_pred = -team_estimator(player_coef, team_list)
        
        spearman_coef, p = spearmanr(y_real, y_pred)
        kendall_coef, p = kendalltau(y_real, y_pred)

        spearmans.append(spearman_coef)
        kendalls.append(kendall_coef)
    s, k = np.array(spearmans), np.array(kendalls)
    return s[~np.isnan(s)].mean(), k[~np.isnan(k)].mean()