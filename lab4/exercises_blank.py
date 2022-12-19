# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow


def tar_imp_hists(all_scores, all_labels):
    # Function to compute target and impostor histogram
    
    tar_scores = []
    imp_scores = []

    ###########################################################
    # Here is your code
    for i in range(len(all_labels)):
        if all_labels[i] == 1:
            tar_scores.append(all_scores[i])
        else:
            imp_scores.append(all_scores[i])
    ###########################################################
    
    tar_scores = np.array(tar_scores)
    imp_scores = np.array(imp_scores)
    
    return tar_scores, imp_scores

def llr(all_scores, all_labels, tar_scores, imp_scores, gauss_pdf):
    # Function to compute log-likelihood ratio
    
    tar_scores_mean = np.mean(tar_scores)
    tar_scores_std  = np.std(tar_scores)
    imp_scores_mean = np.mean(imp_scores)
    imp_scores_std  = np.std(imp_scores)
    
    all_scores_sort   = np.zeros(len(all_scores))
    ground_truth_sort = np.zeros(len(all_scores), dtype='bool')
    
    ###########################################################
    # Here is your code
    all_scores = np.array(all_scores)
    all_labels = np.array([bool(l) for l in all_labels])
    inds = all_scores.argsort()
    all_scores_sort = all_scores[inds]
    ground_truth_sort = all_labels[inds]
    ###########################################################
    
    tar_gauss_pdf = np.zeros(len(all_scores))
    imp_gauss_pdf = np.zeros(len(all_scores))
    LLR           = np.zeros(len(all_scores))
    
    ###########################################################
    # Here is your code
    tar_gauss_pdf = np.array([gauss_pdf(score, tar_scores_mean, tar_scores_std) for score in all_scores_sort])
    imp_gauss_pdf = np.array([gauss_pdf(score, imp_scores_mean, imp_scores_std) for score in all_scores_sort])
    LLR = np.log(tar_gauss_pdf/imp_gauss_pdf)
    ###########################################################
    
    return ground_truth_sort, all_scores_sort, tar_gauss_pdf, imp_gauss_pdf, LLR

def map_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar):
    # Function to perform maximum a posteriori test
    
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    P_err   = np.zeros(len_thr)
    
    for idx in range(len_thr):
        solution = LLR > LLR[idx]                                      # decision
        
        err = (solution != ground_truth_sort)                          # error vector
        
        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores) # prob. of Type I  error P(Dimp|Htar), false negative rate (FNR)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores) # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)
        
        P_err[idx]   = fnr_thr[idx]*P_Htar + fpr_thr[idx]*(1 - P_Htar) # prob. of error
    
    # Plot error's prob.
    plot(LLR, P_err, color='blue')
    xlabel('$LLR$'); ylabel('$P_e$'); title('Probability of error'); grid(); show()
        
    P_err_idx = np.argmin(P_err) # argmin of error's prob.
    P_err_min = fnr_thr[P_err_idx]*P_Htar + fpr_thr[P_err_idx]*(1 - P_Htar)
    
    return LLR[P_err_idx], fnr_thr[P_err_idx], fpr_thr[P_err_idx], P_err_min

def neyman_pearson_test(ground_truth_sort, LLR, tar_scores, imp_scores, fnr):
    # Function to perform Neyman-Pearson test
    
    thr   = 0.0
    fpr   = 0.0
    
    ###########################################################
    # Here is your code
    len_thr = len(LLR)
    for idx in range(len_thr):
        solution = LLR > LLR[idx] 
        err = (solution != ground_truth_sort)  
        fpr_current = np.sum(err[~ground_truth_sort])/len(imp_scores)
        if fpr_current < fpr:
            thr = LLR[idx]
            fpr = fpr_current
    ###########################################################
    
    return thr, fpr

def bayes_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar, C00, C10, C01, C11):
    # Function to perform Bayes' test
    
    thr   = 0.0
    fnr   = 0.0
    fpr   = 0.0
    AC    = 0.0
    
    ###########################################################
    # Here is your code
    LLR_len = len(LLR)
    for idx in range(LLR_len):
        solution = LLR > LLR[idx] 
        t = (solution == ground_truth_sort)
        f = (solution != ground_truth_sort)

        p_d0h0 = np.sum(t[ground_truth_sort])/len(tar_scores)
        p_d1h0 = np.sum(f[ground_truth_sort])/len(tar_scores)
        p_d0h1 = np.sum(f[~ground_truth_sort])/len(imp_scores)
        p_d1h1 = np.sum(t[~ground_truth_sort])/len(imp_scores)

        p_h0 = P_Htar
        p_h1 = 1 - p_h0
        cost = C00*p_d0h0*p_h0 + C10*p_d1h0*p_h0 + C01*p_d0h1*p_h1 + C11*p_d1h1*p_h1
        if cost < AC:
            AC = cost
            thr = LLR[idx] 
            fnr = p_d0h1
            fpr = p_d1h0
    ###########################################################
    
    return thr, fnr, fpr, AC

def minmax_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar_thr, C00, C10, C01, C11):
    # Function to perform minimax test
    
    thr    = 0.0
    fnr    = 0.0
    fpr    = 0.0
    AC     = 0.0
    P_Htar = 0.0
    
    ###########################################################
    # Here is your code
    LLR_len = len(LLR)
    cost_matrix = np.zeros((LLR_len, len(P_Htar_thr)))
    thr_matrix = np.zeros((LLR_len, len(P_Htar_thr)))
    fnr_matrix = np.zeros((LLR_len, len(P_Htar_thr)))
    fpr_matrix = np.zeros((LLR_len, len(P_Htar_thr)))
    P_Htars = np.zeros((LLR_len, len(P_Htar_thr)))
    progress_max = LLR_len*len(P_Htar_thr)
    progress_step = int(progress_max/10)
    print(LLR_len, len(P_Htar_thr))
    progress_ind = 0
    for LLR_ind, P_Htar_ind in product(range(LLR_len), range(len(P_Htar_thr))):
        if progress_ind%progress_step == 0:
            print(f"{int(progress_ind*100/progress_max)}%: {progress_ind}/{progress_max}")
        solution = LLR > LLR[LLR_ind]
        
        t = (solution == ground_truth_sort)
        f = (solution != ground_truth_sort)

        p_d0h0 = np.sum(t[ground_truth_sort])/len(tar_scores)
        p_d1h0 = np.sum(f[ground_truth_sort])/len(tar_scores)
        p_d0h1 = np.sum(f[~ground_truth_sort])/len(imp_scores)
        p_d1h1 = np.sum(t[~ground_truth_sort])/len(imp_scores)

        p_h0 = P_Htar_thr[P_Htar_ind]
        p_h1 = 1 - p_h0
        
        thr_matrix[LLR_ind, P_Htar_ind] = LLR[LLR_ind]
        fnr_matrix[LLR_ind, P_Htar_ind] = p_d1h0
        fpr_matrix[LLR_ind, P_Htar_ind] = p_d0h1
        cost_matrix[LLR_ind, P_Htar_ind] = C00*p_d0h0*p_h0 + C10*p_d1h0*p_h0 + C01*p_d0h1*p_h1 + C11*p_d1h1*p_h1
        P_Htars[LLR_ind, P_Htar_ind] = p_h0
        progress_ind += 1
    # Min
    max_cost = np.amin(cost_matrix, axis=0)
    argmax_ind_list = np.argmin(cost_matrix, axis=0)
    # Max
    min_cost = np.amax(max_cost)
    print(min_cost)
    ind_1 = np.argmax(max_cost)
    ind_0 = argmax_ind_list[ind_1]
    thr = thr_matrix[ind_0, ind_1]
    fnr = fnr_matrix[ind_0, ind_1]
    fpr = fpr_matrix[ind_0, ind_1]
    AC = cost_matrix[ind_0, ind_1]
    P_Htar = P_Htars[ind_0, ind_1]
    ###########################################################
    
    return thr, fnr, fpr, AC, P_Htar