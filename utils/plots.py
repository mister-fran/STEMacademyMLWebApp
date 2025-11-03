import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from scipy.stats import gaussian_kde
import streamlit as st

def plotting(sande, forudsagte):

    r2 = sklearn.metrics.r2_score(sande, forudsagte) 
    MAE = sklearn.metrics.mean_absolute_error(sande, forudsagte)
    MSE = sklearn.metrics.mean_squared_error(sande, forudsagte)

    residuals = forudsagte - sande
    uncertainty = np.std(residuals)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ###---- VENSTRE PLOT -----###
    ax = axes[0]
    ax.hist((forudsagte - sande), bins = 100)
    ax.text(0.02, 0.98, f"Usikkerhed: {uncertainty:,.2f} mio.kr.",
    transform=ax.transAxes, va='top', ha='left',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
    ax.set_xlabel('Residual')
    ax.set_ylabel('counts')
    ax.set_title('Histogram residualer')

    ###---- HØJRE PLOT -----###
    ax = axes[1]
    ax.scatter(sande, forudsagte, alpha=0.3, s=10, label="Datapunkter")
    # Contour
    xy = np.vstack([sande, forudsagte])
    kde = gaussian_kde(xy)
    xmin, xmax = np.min(sande), np.max(sande)
    ymin, ymax = np.min(forudsagte), np.max(forudsagte)
    xgrid, ygrid = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xgrid.ravel(), ygrid.ravel()])
    density = np.reshape(kde(positions).T, xgrid.shape)
    ax.contour(xgrid, ygrid, density, colors='k', linewidths=1, alpha=0.7)

    ax.plot([xmin, xmax], [xmin, xmax], 'r--', label='Perfekt forudsigelse')
    ax.grid(True)
    ax.set_ylabel('Forudsagte pris (mio)')
    ax.set_xlabel('Sande pris (mio)')
    ax.text(0.02, 0.98, f"MAE: {MAE:,.2f}\nMSE: {MSE:,.2f}\nR²: {r2:,.2f}",
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
    ax.legend()
    ax.set_title('Sande vs forudsagte pris')

    fig.tight_layout()
    st.pyplot(fig)

def Plotting_class(label_test, Forudsigelse, y_pred_label, beslutningsgrænse):

    #ROC
    #Udregn raten af sande/falske forudsigelser
    fpr, tpr, _ = sklearn.metrics.roc_curve(label_test, Forudsigelse)    
    #AUC score
    auc_score = sklearn.metrics.auc(fpr,tpr)  

    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(10, 5))

    #Plotting
    axs[0].set_title('ROC-curve', size = 10)
    axs[0].plot(fpr, tpr, label=f'(AUC = {auc_score:5.3f})')
    axs[0].plot([0, 1], [0, 1], 'r--', label='Tilfældig')
    axs[0].legend(fontsize=10, loc ='lower right')
    axs[0].set_xlabel('False Positive Rate', size=10)
    axs[0].set_ylabel('True Positive Rate', size=10)
    axs[0].grid(True, alpha=0.3)

    #Beregning af falske klassificeringer
    sandsynlighed_falsk = Forudsigelse[label_test == 0]
    sandsynlighed_korrekt = Forudsigelse[label_test == 1]
    syg_but_pred_rask_mask = (label_test == 1) & (y_pred_label == 0)
    rask_but_pred_syg_mask = (label_test == 0) & (y_pred_label == 1)
    syg_but_pred_rask = syg_but_pred_rask_mask.sum()
    rask_but_pred_syg = rask_but_pred_syg_mask.sum()
    antal_syg = (label_test == 1).sum()
    antal_rask = (label_test == 0).sum()
    pct_syg_but_pred_rask = 100 * syg_but_pred_rask / antal_syg if antal_syg > 0 else 0
    pct_rask_but_pred_syg = 100 * rask_but_pred_syg / antal_rask if antal_rask > 0 else 0

    #Plotting
    bins = 10
    axs[1].hist(sandsynlighed_falsk, bins=bins, alpha=0.6, color='lightblue', 
                   label=f'Rask (n={len(sandsynlighed_falsk)})', edgecolor='black')
    axs[1].hist(sandsynlighed_korrekt, bins=bins, alpha=0.6, color='lightcoral', 
    label=f'Diabetes (n={len(sandsynlighed_korrekt)})', edgecolor='black')
    axs[1].set_xlabel('Forudsagt sandsynlighed diabetes')
    axs[1].set_ylabel('Antal')
    axs[1].set_title('Sandsynlighedsfordeling',size=10)
    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_label_position("right")
    axs[1].grid(True, alpha=0.3)
    axs[1].text(
    0.1, 0.5,
    (
        f"Raske klassificeret som syge: {rask_but_pred_syg} "
        f"({pct_rask_but_pred_syg:,.1f}%)\n"
        f"Syge klassificeret som raske: {syg_but_pred_rask} "
        f"({pct_syg_but_pred_rask:,.1f}%)"
    ),
    transform=axs[1].transAxes,
    va='top', ha='left',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8)
)
    axs[1].axvline(x=beslutningsgrænse, color='red', linestyle='--', linewidth=2, label=f'Beslutningsgrænse = {beslutningsgrænse}')
    axs[1].legend(loc='upper center')

    fig.tight_layout()  
    st.pyplot(fig)

#Funktion der visualiserer resultaterne
def plotting_partikel(label_test, Forudsigelse):

    #ROC
    #Udregn raten af sande/falske fordusigelser
    fpr, tpr, _ = sklearn.metrics.roc_curve(label_test, Forudsigelse)    
    #AUC score
    auc_score = sklearn.metrics.auc(fpr,tpr)  

    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(10, 6))

    #Plotting
    axs[0].set_title('ROC-curve', size = 10)
    axs[0].plot(fpr, tpr, label=f'(AUC = {auc_score:5.3f})')
    axs[0].legend(fontsize=10, loc ='lower right')
    axs[0].set_xlabel('False Positive Rate', size=10)
    axs[0].set_ylabel('True Positive Rate', size=10)
    axs[0].grid(True, alpha=0.3)

    #Histogram
    sandsynlighed_falsk = Forudsigelse[label_test == 0]
    sandsynlighed_korrekt = Forudsigelse[label_test == 1]

    #Plotting
    axs[1].hist(sandsynlighed_falsk, bins=30, alpha=0.6, color='lightblue', 
                   label=f'Ikke elektron (n={len(sandsynlighed_falsk)})', edgecolor='black')
    axs[1].hist(sandsynlighed_korrekt, bins=30, alpha=0.6, color='lightcoral', 
    label=f'Elektron (n={len(sandsynlighed_korrekt)})', edgecolor='black')
    axs[1].set_xlabel('Forudsagt sandsynlighed for at det er en elektron')
    axs[1].set_ylabel('Antal')
    axs[1].set_title('Fordeling af forudsagte sandsynligheder farvekodet efter sande værdier',size=10)
    axs[1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Beslutningsgrænse (0.5)')
    axs[1].legend(loc='upper center')
    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_label_position("right")
    axs[1].grid(True, alpha=0.3)


    fig.tight_layout()  
    st.pyplot(fig)