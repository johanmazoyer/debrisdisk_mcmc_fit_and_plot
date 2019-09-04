#pylint: disable=C0103
""" author: J MAZOYER
A code that tahke the return of the MCMC and plot the SPFs for hr 4796
"""
import csv
import numpy as np

import matplotlib.pyplot as plot
import yaml

from scipy import optimize
from emcee import backends


def hg_2g(scatt_angles, g1, g2, alpha, Norm):
    """
    take a set of scatt angles and a set of HG parameter and return a
    2g HG SPF

    Args:
        scatt_angles: a list of angles in degrees of dimensions N_angles.
                        The list must contains 90 degree values
        g1: first HG parameter
        g2: second HG parameter
        alpha: relative weight
                 hg = alpha * hg1 * hg2 + (1 - alpha) * hg2
        Norm: Normalisation (value at 90 degree of the function)

    Returns:
        the 2g SPF, list of dimensions N_angles.

    """

    scattered_angles_rad = np.radians(scatt_angles)
    cos_phi = np.cos(scattered_angles_rad)

    g1_2 = g1 * g1  #First HG g squared
    g2_2 = g2 * g2  #Second HG g squared
    #Constant for HG function
    k = 1. / (4 * np.pi)

    #Henyey Greenstein function
    hg1 = k * alpha * (1. - g1_2) / (1. + g1_2 - (2 * g1 * cos_phi))**1.5
    hg2 = k * (1 - alpha) * (1. - g2_2) / (1. + g2_2 - (2 * g2 * cos_phi))**1.5
    hg = hg1 + hg2
    hg_norm = hg / hg[np.where(scatt_angles == 90)] * Norm
    return hg_norm


def hg_3g(scatt_angles, g1, g2, g3, alpha1, alpha2, Norm):
    """
    take a set of scatt angles and a set of HG parameter and return a
    3g HG SPF

    Args:
        scatt_angles: a list of angles in degrees of dimensions N_angles.
                        The list must contains 90 degree values
        g1: first HG parameter
        g2: second HG parameter
        g3: third HG parameter
        alpha1: first relative weight
        alpha2: second relative weight
                hg = alpha1 * hg1 + alpha2 * hg2 + (1 - alpha1 - alpha2) * hg3
        Norm: Normalisation (value at 90 degree of the function)

    Returns:
        the 3g SPF, list of dimensions N_angles.
    """

    scattered_angles_rad = np.radians(scatt_angles)
    cos_phi = np.cos(scattered_angles_rad)

    g1_2 = g1 * g1  #First HG g squared
    g2_2 = g2 * g2  #Second HG g squared
    g3_2 = g3 * g3  #Third HG g squared

    #Constant for HG function
    k = 1. / (4 * np.pi)

    #Henyey Greenstein function
    hg1 = k * (1. - g1_2) / (1. + g1_2 - (2 * g1 * cos_phi))**1.5
    hg2 = k * (1. - g2_2) / (1. + g2_2 - (2 * g2 * cos_phi))**1.5
    hg3 = k * (1. - g3_2) / (1. + g3_2 - (2 * g3 * cos_phi))**1.5
    hg = alpha1 * hg1 + alpha2 * hg2 + (1 - alpha1 - alpha2) * hg3

    hg_norm = hg / hg[np.where(scatt_angles == 90)] * Norm

    return hg_norm


def log_hg_2g(scatt_angles, g1, g2, alpha, Norm):
    """
    take a set of scatt angles and a set of HG parameter and return
    the log of a 2g HG SPF (usefull to fit from a set of points)

    Args:
        scatt_angles: a list of angles in degrees of dimensions N_angles.
                        The list must contains 90 degree values
        g1: first HG parameter
        g2: second HG parameter
        alpha: relative weight
                 hg = alpha * hg1 * hg2 + (1 - alpha) * hg2
        Norm: Normalisation (value at 90 degree of the function)

    Returns:
        the log of the 2g SPF, list of dimensions N_angles.

    """

    return np.log(hg_2g(scatt_angles, g1, g2, alpha, Norm))


def log_hg_3g(scatt_angles, g1, g2, g3, alpha1, alpha2, Norm):
    """
    take a set of scatt angles and a set of HG parameter and return the
    log of the 3g HG SPF (usefull to fit from a set of points)

    Args:
        scatt_angles: a list of angles in degrees of dimensions N_angles.
                        The list must contains 90 degree values
        g1: first HG parameter
        g2: second HG parameter
        g3: third HG parameter
        alpha1: first relative weight
        alpha2: second relative weight
                hg = alpha1 * hg1 + alpha2 * hg2 + (1 - alpha1 - alpha2) * hg3
        Norm: Normalisation (value at 90 degree of the function)

    Returns:
        the log of the 3g SPF, list of dimensions N_angles.
    """
    return np.log(hg_3g(scatt_angles, g1, g2, g3, alpha1, alpha2, Norm))


def measure_spf_errors(yaml_file_str, Number_rand_mcmc, Norm_90_inplot=1.):
    """
    take a set of scatt angles and a set of HG parameter and return
    the log of a 2g HG SPF (usefull to fit from a set of points)

    Args:
        yaml_file_str: name of the yaml_ parameter file
        Number_rand_mcmc: number of randomnly selected psf we use to
                        plot the error bars
        Norm_90_inplot: the value at which you want to normalize the spf
                        in the plot ay 90 degree (to re-measure the error bars properly)

    Returns:
        a dic that contains the 'best_spf', 'errorbar_sup', 'errorbar_sup', 'errorbar'
    """

    with open('initialization_files/' + yaml_file_str + '.yaml',
              'r') as yaml_file:
        params_mcmc_yaml = yaml.load(yaml_file, Loader=yaml.FullLoader)

    dico_return = dict()
    burnin = params_mcmc_yaml['BURNIN']
    thin = params_mcmc_yaml['THIN']
    n_dim_mcmc = params_mcmc_yaml['N_DIM_MCMC']

    band_dir = params_mcmc_yaml['BAND_DIR']

    datadir = basedir + band_dir

    mcmcresultdir = datadir + 'results_MCMC/'

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    name_h5 = file_prefix + "_backend_file_mcmc"

    chain_name = mcmcresultdir + name_h5 + ".h5"

    reader = backends.HDFBackend(chain_name)
    chain_flat = reader.get_chain(discard=burnin, flat=True, thin=thin)

    if n_dim_mcmc == 11:
        g1_chain = chain_flat[:, 3]
        g2_chain = chain_flat[:, 4]
        alph1_chain = chain_flat[:, 5]
        norm_chain = np.exp(chain_flat[:, 10])

    if n_dim_mcmc == 13:
        g1_chain = chain_flat[:, 3]
        g2_chain = chain_flat[:, 4]
        g3_chain = chain_flat[:, 5]
        alph1_chain = chain_flat[:, 6]
        alph2_chain = chain_flat[:, 7]
        norm_chain = np.exp(chain_flat[:, 12])

    bestmodel_g1 = np.percentile(g1_chain, 50)
    bestmodel_g2 = np.percentile(g2_chain, 50)
    bestmodel_alpha1 = np.percentile(alph1_chain, 50)
    bestmodel_Norm = np.percentile(norm_chain, 50)

    if n_dim_mcmc == 11:
        best_hg_mcmc = hg_2g(scattered_angles, bestmodel_g1, bestmodel_g2,
                             bestmodel_alpha1, Norm_90_inplot)

    if n_dim_mcmc == 13:
        bestmodel_g3 = np.percentile(g3_chain, [50])
        bestmodel_alpha2 = np.percentile(alph2_chain, [50])
        best_hg_mcmc = hg_3g(scattered_angles, bestmodel_g1, bestmodel_g2,
                             bestmodel_g3, bestmodel_alpha1, bestmodel_alpha2,
                             Norm_90_inplot)

    dico_return['best_spf'] = best_hg_mcmc

    random_param_number = np.random.randint(1,
                                            len(g1_chain) - 1,
                                            Number_rand_mcmc)

    g1_rand = g1_chain[random_param_number]
    g2_rand = g2_chain[random_param_number]
    alph1_rand = alph1_chain[random_param_number]
    norm_rand = norm_chain[random_param_number]

    if n_dim_mcmc == 13:
        g3_rand = g3_chain[random_param_number]
        alph2_rand = alph2_chain[random_param_number]

    hg_mcmc_rand = np.zeros((len(best_hg_mcmc), len(random_param_number)))

    errorbar_sup = scattered_angles * 0.
    errorbar_inf = scattered_angles * 0.
    errorbar = scattered_angles * 0.

    for num_model in range(Number_rand_mcmc):
        g1_here = g1_rand[num_model]
        g2_here = g2_rand[num_model]
        alph1_here = alph1_rand[num_model]
        norm_here = norm_rand[num_model]
        if n_dim_mcmc == 11:
            hg_mcmc_rand[:, num_model] = hg_2g(
                scattered_angles, g1_here, g2_here, alph1_here,
                norm_here / bestmodel_Norm * Norm_90_inplot)
        if n_dim_mcmc == 13:
            g3_here = g3_rand[num_model]
            alph2_here = alph2_rand[num_model]
            hg_mcmc_rand[:, num_model] = hg_3g(
                scattered_angles, g1_here, g2_here, g3_here, alph1_here,
                alph2_here, norm_here / bestmodel_Norm * Norm_90_inplot)

    for anglei in range(len(scattered_angles)):
        errorbar_sup[anglei] = np.max(hg_mcmc_rand[anglei, :])
        errorbar_inf[anglei] = np.min(hg_mcmc_rand[anglei, :])
        errorbar[anglei] = (np.max(hg_mcmc_rand[anglei, :]) -
                            np.min(hg_mcmc_rand[anglei, :])) / 2.

    dico_return['errorbar_sup'] = errorbar_sup
    dico_return['errorbar_inf'] = errorbar_inf
    dico_return['errorbar'] = errorbar

    return dico_return


basedir = '/Users/jmazoyer/Dropbox/ExchangeFolder/data_python/Aurora/'

min_scat = 13.3
max_scat = 166.7

folder_save_pdf = basedir + 'Spf_plots_produced/'

scattered_angles = np.arange(np.round(max_scat - min_scat)) + np.round(
    np.min(min_scat))

# ###########################################################################
# ### SPHERE H2 exctracted by Milli et al. 2017
# ###########################################################################
# sphere spf extracted by Milli et al. 2017
angles_sphere_extractJulien = np.zeros(49)
spf_shpere_extractJulien = np.zeros(49)
errors_sphere_extractJulien = np.zeros(49)

i = 0
with open(basedir + 'SPHERE_Hdata/' + 'SPHERE_extraction_Milli.csv',
          'rt') as f:
    readercsv = csv.reader(f)
    for row in readercsv:
        angles_sphere_extractJulien[i] = float(row[0])
        spf_shpere_extractJulien[i] = float(row[1])
        errors_sphere_extractJulien[i] = float(row[2])
        i += 1

# fit a 3g function to sphere spf extracted by Milli et al. 2017
log_spf_sphere = np.log(spf_shpere_extractJulien)
log_errors_sphere = np.log(spf_shpere_extractJulien +
                           errors_sphere_extractJulien) - np.log(
                               spf_shpere_extractJulien)

initial_guess = [
    0.99997858, 0.05095741, 0.04765255, 0.99946266, -0.03396517, 0.94720999
]
params = initial_guess
a = optimize.curve_fit(log_hg_3g,
                       angles_sphere_extractJulien,
                       log_spf_sphere,
                       initial_guess,
                       log_errors_sphere,
                       maxfev=10000,
                       bounds=([-1, -1., -1., -1., -1.,
                                -1.], [1, 1., 1., 1., 1., 4.]))
params_3g_fir_from_extraction = a[0]
hg3g_fitted_Milliextraction = hg_3g(
    scattered_angles, params_3g_fir_from_extraction[0],
    params_3g_fir_from_extraction[1], params_3g_fir_from_extraction[2],
    params_3g_fir_from_extraction[3], params_3g_fir_from_extraction[4],
    params_3g_fir_from_extraction[5])

# ##############################
# # GPI K1 extracted by Pauline from Pol
# ##############################

dir_exctract_Pauline = basedir + '150403_K1_Spec/Pauline_SPF_from_Pol/'

data_exctract_Pauline = dir_exctract_Pauline + 'Pauline_SPF_ktot_NE_errs.txt'
text_array = np.loadtxt(data_exctract_Pauline)
angles_GPIPolK1_extractPauline = text_array[:, 0]
spf_GPIPolK1_extractPaulineNE = text_array[:, 1]
errors_GPIPolK1_extractPaulineNE = text_array[:, 2]

data_exctract_Pauline = dir_exctract_Pauline + 'Pauline_SPF_ktot_SW_errs.txt'

text_array = np.loadtxt(data_exctract_Pauline)
# angles_GPIPolK1_extractPaulineSW =  text_array[:, 0] ## identical to NE
spf_GPIPolK1_extractPaulineSW = text_array[:, 1]
errors_GPIPolK1_extractPaulineSW = text_array[:, 2]

spf_GPIPolK1_extractPauline = 0.5 * (spf_GPIPolK1_extractPaulineSW +
                                     spf_GPIPolK1_extractPaulineNE)

error_GPIPolK1_extractPauline = np.sqrt(errors_GPIPolK1_extractPaulineSW**2 +
                                        errors_GPIPolK1_extractPaulineNE**2)

error_GPIPolK1_extractPauline = error_GPIPolK1_extractPauline / spf_GPIPolK1_extractPauline[
    np.where(angles_GPIPolK1_extractPauline == 90)]
spf_GPIPolK1_extractPauline = spf_GPIPolK1_extractPauline / spf_GPIPolK1_extractPauline[
    np.where(angles_GPIPolK1_extractPauline == 90)]

index = (np.arange((error_GPIPolK1_extractPauline.shape[0]) / 2)) * 2 + 1

angles_GPIPolK1_extractPauline = np.delete(angles_GPIPolK1_extractPauline,
                                           index)
spf_GPIPolK1_extractPauline = np.delete(spf_GPIPolK1_extractPauline, index)
error_GPIPolK1_extractPauline = np.delete(error_GPIPolK1_extractPauline, index)

spf_sphere_h = measure_spf_errors('SPHERE_Hband_MCMC', 500)
spf_sphere_h_3g = measure_spf_errors('SPHERE_Hband_3g_MCMC', 500)
spf_gpi_not1at90 = measure_spf_errors('GPI_Hband_MCMC',
                                      500,
                                      Norm_90_inplot=0.93)
spf_gpi_h_1at90 = measure_spf_errors('GPI_Hband_MCMC', 500)
spf_gpi_j = measure_spf_errors('GPI_Jband_MCMC', 500)
spf_gpi_k1 = measure_spf_errors('GPI_K1band_MCMC', 500)
spf_gpi_k2 = measure_spf_errors('GPI_K2band_MCMC', 500)

color0 = 'black'
color1 = '#3B73FF'
color2 = '#ED0052'
color3 = '#00AF64'
color4 = '#FFCF0B'

####################################################################################
## plot h only
####################################################################################
plot.figure()
name_pdf = 'compare_SPHERE_GPI_H.pdf'

plot.plot(scattered_angles,
          spf_sphere_h['best_spf'],
          linewidth=2,
          color=color0,
          label="SPHERE H2 (extraction MCMC, this work)")
plot.fill_between(scattered_angles,
                  spf_sphere_h['errorbar_sup'],
                  spf_sphere_h['errorbar_inf'],
                  facecolor=color0,
                  alpha=0.2)

plot.plot(scattered_angles,
          spf_gpi_not1at90['best_spf'],
          linewidth=2,
          color=color1,
          label="GPI H")
plot.fill_between(scattered_angles,
                  spf_gpi_not1at90['errorbar_sup'],
                  spf_gpi_not1at90['errorbar_inf'],
                  facecolor=color1,
                  alpha=0.2)

plot.errorbar(angles_sphere_extractJulien,
              1.38 * spf_shpere_extractJulien,
              yerr=errors_sphere_extractJulien,
              fmt='o',
              label='SPHERE H2 (extraction Milli et al. 2017)',
              ms=3,
              capthick=1,
              capsize=2,
              elinewidth=1,
              markeredgewidth=1,
              color='grey')

# plot.errorbar(angles_GPIPolK1_extractPauline,
#               spf_GPIPolK1_extractPauline,
#               yerr=error_GPIPolK1_extractPauline,
#               fmt='x',
#               label='GPI Pol K1 (extraction P. Arriaga)',
#               ms=3,
#               capthick=1,
#               capsize=2,
#               elinewidth=1,
#               markeredgewidth=1,
#               color=color3)

plot.legend()
plot.yscale('log')

plot.ylim(bottom=0.3, top=30)
plot.xlim(left=0, right=180)
plot.xlabel('Scattering angles')
plot.ylabel('Normalized total intensity')

plot.tight_layout()

plot.savefig(folder_save_pdf + name_pdf)

plot.close()

####################################################################################
## plot all colors
####################################################################################
name_pdf = 'compare_GPI_color.pdf'
plot.figure()

plot.plot(scattered_angles,
          spf_gpi_j['best_spf'],
          linewidth=2,
          color=color4,
          label="GPI J")
plot.fill_between(scattered_angles,
                  spf_gpi_j['errorbar_sup'],
                  spf_gpi_j['errorbar_inf'],
                  facecolor=color4,
                  alpha=0.2)

plot.plot(scattered_angles,
          spf_sphere_h['best_spf'],
          linewidth=2,
          color=color0,
          label="SPHERE H2 (extraction MCMC, this work)")
plot.fill_between(scattered_angles,
                  spf_sphere_h['errorbar_sup'],
                  spf_sphere_h['errorbar_inf'],
                  facecolor=color0,
                  alpha=0.2)

plot.plot(scattered_angles,
          spf_gpi_h_1at90['best_spf'],
          linewidth=2,
          color=color1,
          label="GPI H")
plot.fill_between(scattered_angles,
                  spf_gpi_h_1at90['errorbar_sup'],
                  spf_gpi_h_1at90['errorbar_inf'],
                  facecolor=color1,
                  alpha=0.2)

plot.plot(scattered_angles,
          spf_gpi_k1['best_spf'],
          linewidth=2,
          color=color2,
          label="GPI K1")
plot.fill_between(scattered_angles,
                  spf_gpi_k1['errorbar_sup'],
                  spf_gpi_k1['errorbar_inf'],
                  facecolor=color2,
                  alpha=0.2)

plot.plot(scattered_angles,
          spf_gpi_k2['best_spf'],
          linewidth=2,
          color=color3,
          label="GPI k2")
plot.fill_between(scattered_angles,
                  spf_gpi_k2['errorbar_sup'],
                  spf_gpi_k2['errorbar_inf'],
                  facecolor=color3,
                  alpha=0.2)

plot.legend()
plot.yscale('log')

plot.ylim(bottom=0.3, top=30)
plot.xlim(left=0, right=180)
plot.xlabel('Scattering angles')
plot.ylabel('Normalized total intensity')

plot.tight_layout()

plot.savefig(folder_save_pdf + name_pdf)

plot.close()
####################################################################################
## 3g plot
####################################################################################
name_pdf = 'compare_3g_SPF.pdf'
plot.figure()

plot.plot(scattered_angles,
          spf_sphere_h['best_spf'],
          linewidth=2,
          color=color0,
          label="2 HG SPF extraction MCMC (this work)")
plot.fill_between(scattered_angles,
                  spf_sphere_h['errorbar_sup'],
                  spf_sphere_h['errorbar_inf'],
                  facecolor=color0,
                  alpha=0.2)

plot.plot(scattered_angles,
          spf_sphere_h_3g['best_spf'],
          linewidth=2,
          color=color1,
          label="3 HG SPF extraction MCMC (this work)")
plot.fill_between(scattered_angles,
                  spf_sphere_h_3g['errorbar_sup'],
                  spf_sphere_h_3g['errorbar_inf'],
                  facecolor=color1,
                  alpha=0.2)

plot.errorbar(angles_sphere_extractJulien,
              1.38 * spf_shpere_extractJulien,
              yerr=errors_sphere_extractJulien,
              fmt='o',
              label='SPHERE H2 (extraction Milli et al. 2017)',
              ms=3,
              capthick=1,
              capsize=2,
              elinewidth=1,
              markeredgewidth=1,
              color='grey')
plot.plot(scattered_angles,
          1.38 * hg3g_fitted_Milliextraction,
          linewidth=2,
          color=color3,
          label="3 HG SPF fit to the SPF extracted in Milli et al. 2017")

handles, labels = plot.gca().get_legend_handles_labels()
order = [3, 2, 1, 0]
plot.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

plot.yscale('log')

plot.ylim(bottom=0.3, top=30)
plot.xlim(left=0, right=180)
plot.xlabel('Scattering angles')
plot.ylabel('Normalized total intensity')

plot.tight_layout()

plot.savefig(folder_save_pdf + name_pdf)

plot.close()
