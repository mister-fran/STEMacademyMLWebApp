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
