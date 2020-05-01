#pylint: disable=C0103
""" author: J MAZOYER
A code that tahke the chain return of the MCMCs and plot the SPFs for hr 4796
"""
import os
import csv
import numpy as np

import matplotlib.pyplot as plot
import yaml

from scipy import optimize
from emcee import backends

from disk_models import hg_1g, hg_2g, hg_3g, log_hg_2g, log_hg_3g


def measure_spf_errors(yaml_file_str, Number_rand_mcmc, Norm_90_inplot=1., save = False):
    """
    take a set of scatt angles and a set of HG parameter and return
    the log of a 2g HG SPF (usefull to fit from a set of points)

    Args:
        yaml_file_str: name of the yaml_ parameter file
        Number_rand_mcmc: number of randomnly selected psf we use to
                        plot the error bars
        Norm_90_inplot: the value at which you want to normalize the spf
                        in the plot ay 90 degree (to re-measure the error
                        bars properly)

    Returns:
        a dic that contains the 'best_spf', 'errorbar_sup',
                                'errorbar_sup', 'errorbar'
    """

    with open(os.path.join('initialization_files', yaml_file_str + '.yaml'),
              'r') as yaml_file:
        params_mcmc_yaml = yaml.load(yaml_file, Loader=yaml.FullLoader)

    dico_return = dict()

    nwalkers = params_mcmc_yaml['NWALKERS']
    n_dim_mcmc = params_mcmc_yaml['N_DIM_MCMC']

    datadir = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])

    mcmcresultdir = os.path.join(datadir, 'results_MCMC')

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    
    SPF_MODEL = params_mcmc_yaml['SPF_MODEL']  #Type of description for the SPF

    name_h5 = file_prefix + "_backend_file_mcmc"

    chain_name = os.path.join(mcmcresultdir, name_h5 + ".h5")

    reader = backends.HDFBackend(chain_name)


    #we only exctract the last itearations, assuming it converged
    chain_flat = reader.get_chain(discard=0, flat=True)
    burnin = np.clip(reader.iteration - 10*Number_rand_mcmc//nwalkers,0,None)
    chain_flat = reader.get_chain(discard=burnin, flat=True)

    
    if (SPF_MODEL == 'hg_1g'):
        norm_chain = np.exp(chain_flat[:, 7])
        g1_chain = chain_flat[:, 8]

        bestmodel_Norm = np.percentile(norm_chain, 50)
        bestmodel_g1 = np.percentile(g1_chain, 50)

        Normalization = Norm_90_inplot

        if save == True:
            Normalization = bestmodel_Norm

        best_hg_mcmc = hg_1g(scattered_angles, bestmodel_g1, Normalization)

    elif SPF_MODEL == 'hg_2g':
        norm_chain = np.exp(chain_flat[:, 7])
        g1_chain = chain_flat[:, 8]
        g2_chain = chain_flat[:, 9]
        alph1_chain = chain_flat[:, 10]

        bestmodel_Norm = np.percentile(norm_chain, 50)
        bestmodel_g1 = np.percentile(g1_chain, 50)
        bestmodel_g2 = np.percentile(g2_chain, 50)
        bestmodel_alpha1 = np.percentile(alph1_chain, 50)
        Normalization = Norm_90_inplot


        if save == True:
            Normalization = bestmodel_Norm


        best_hg_mcmc = hg_2g(scattered_angles, bestmodel_g1, bestmodel_g2,
                             bestmodel_alpha1, Normalization)


    elif SPF_MODEL == 'hg_3g':
        # temporary, the 3g is not finish so we remove some of the
        # chains that are obvisouly bad. When 3g is finally converged,
        # we removed that
        # incl_chain = np.degrees(np.arccos(chain_flat[:, 3]))
        # where_incl_is_ok = np.where(incl_chain > 76)
        # norm_chain = np.exp(chain_flat[where_incl_is_ok, 7]).flatten()
        # g1_chain =  chain_flat[where_incl_is_ok, 8].flatten()
        # g2_chain = chain_flat[where_incl_is_ok, 9].flatten()
        # alph1_chain = chain_flat[where_incl_is_ok, 10].flatten()
        # g3_chain = chain_flat[where_incl_is_ok, 11].flatten()
        # alph2_chain = chain_flat[where_incl_is_ok, 12].flatten()
        

        # log_prob_samples_flat = reader.get_log_prob(discard=burnin,
        #                                             flat=True)
        # log_prob_samples_flat = log_prob_samples_flat[where_incl_is_ok]
        # wheremin = np.where(
        #         log_prob_samples_flat == np.max(log_prob_samples_flat))
        # wheremin0 = np.array(wheremin).flatten()[0]

        # bestmodel_g1 = g1_chain[wheremin0]
        # bestmodel_g2 = g2_chain[wheremin0]
        # bestmodel_g3 = g3_chain[wheremin0]
        # bestmodel_alpha1 = alph1_chain[wheremin0]
        # bestmodel_alpha2 = alph2_chain[wheremin0]
        # bestmodel_Norm = norm_chain[wheremin0]

        norm_chain = np.exp(chain_flat[:, 7])
        g1_chain = chain_flat[:, 8]
        g2_chain = chain_flat[:, 9]
        alph1_chain = chain_flat[:, 10]
        g3_chain = chain_flat[:, 11]
        alph2_chain = chain_flat[:, 12]

        bestmodel_Norm = np.percentile(norm_chain, 50)
        bestmodel_g1 = np.percentile(g1_chain, 50)
        bestmodel_g2 = np.percentile(g2_chain, 50)
        bestmodel_alpha1 = np.percentile(alph1_chain, 50)
        bestmodel_g3 = np.percentile(g3_chain, 50)
        bestmodel_alpha2 = np.percentile(alph2_chain, 50)
        Normalization = Norm_90_inplot

        # we normalize the best model at 90 either by the value found
        # by the MCMC if we want to save or by the value in the
        # Norm_90_inplot if we want to plot

        Normalization = Norm_90_inplot
        if save == True:
            Normalization = bestmodel_Norm

        best_hg_mcmc = hg_3g(scattered_angles, bestmodel_g1, bestmodel_g2,
                             bestmodel_g3, bestmodel_alpha1, bestmodel_alpha2,
                             Normalization)


    dico_return['best_spf'] = best_hg_mcmc

    random_param_number = np.random.randint(1,
                                            len(g1_chain) - 1,
                                            Number_rand_mcmc)

    if (SPF_MODEL == 'hg_1g') or (SPF_MODEL == 'hg_2g') or (
            SPF_MODEL == 'hg_3g'):
        g1_rand = g1_chain[random_param_number]
        norm_rand = norm_chain[random_param_number]

        if (SPF_MODEL == 'hg_2g') or (SPF_MODEL == 'hg_3g'):
            g2_rand = g2_chain[random_param_number]
            alph1_rand = alph1_chain[random_param_number]

            if SPF_MODEL == 'hg_3g':
                g3_rand = g3_chain[random_param_number]
                alph2_rand = alph2_chain[random_param_number]


    hg_mcmc_rand = np.zeros((len(best_hg_mcmc), len(random_param_number)))

    errorbar_sup = scattered_angles * 0.
    errorbar_inf = scattered_angles * 0.
    errorbar = scattered_angles * 0.

    for num_model in range(Number_rand_mcmc):

        norm_here = norm_rand[num_model]

        # we normalize the random SPF at 90 either by the value of
        # the SPF by the MCMC if we want to save or around the
        # Norm_90_inplot if we want to plot

        Normalization = norm_here * Norm_90_inplot / bestmodel_Norm
        if save == True:
            Normalization = norm_here
        
        if (SPF_MODEL == 'hg_1g'):
            g1_here = g1_rand[num_model]
            
        if (SPF_MODEL == 'hg_2g'):
            g1_here = g1_rand[num_model]
            g2_here = g2_rand[num_model]
            alph1_here = alph1_rand[num_model]
            hg_mcmc_rand[:, num_model] = hg_2g(
                scattered_angles, g1_here, g2_here, alph1_here, Normalization)

        if SPF_MODEL == 'hg_3g':
            g3_here = g3_rand[num_model]
            alph2_here = alph2_rand[num_model]
            hg_mcmc_rand[:, num_model] = hg_3g(
                scattered_angles, g1_here, g2_here, g3_here, alph1_here,
                alph2_here, Normalization)
        

    for anglei in range(len(scattered_angles)):
        errorbar_sup[anglei] = np.max(hg_mcmc_rand[anglei, :])
        errorbar_inf[anglei] = np.min(hg_mcmc_rand[anglei, :])
        errorbar[anglei] = (np.max(hg_mcmc_rand[anglei, :]) -
                            np.min(hg_mcmc_rand[anglei, :])) / 2.

    dico_return['errorbar_sup'] = errorbar_sup
    dico_return['errorbar_inf'] = errorbar_inf
    dico_return['errorbar'] = errorbar

    if save == True:
        savefortext = np.transpose([scattered_angles, best_hg_mcmc, errorbar_sup, errorbar_inf])
        path_and_name_txt = os.path.join(folder_save_pdf, file_prefix+'_spf.txt')

        np.savetxt( path_and_name_txt, savefortext, delimiter=',', fmt = '%10.2f')   # save the array in a txt
    return dico_return

if __name__ == '__main__':
    basedir = os.environ["EXCHANGE_PATH"]  # the base directory where is
    # your data (using OS environnement variable allow to use same code on
    # different computer without changing this).
    # basedir = '/Users/jmazoyer/Dropbox/ExchangeFolder/data_python/Aurora/'

    min_scat = 13.3
    max_scat = 166.7
    nb_random_models = 1000


    folder_save_pdf = os.path.join(basedir, 'Spf_plots_produced')

    scattered_angles = np.arange(np.round(max_scat - min_scat)) + np.round(
        np.min(min_scat))

    # ###########################################################################
    # ### SPF injected real value
    # ###########################################################################
    g1_injected = 0.825
    g2_injected = -0.201
    alph1_injected = 0.298
    injected_hg = hg_2g(scattered_angles,g1_injected,g2_injected,alph1_injected, 1.0)


    # ###########################################################################
    # ### SPHERE H2 exctracted by Milli et al. 2017
    # ###########################################################################
    # sphere spf extracted by Milli et al. 2017
    angles_sphere_extractJulien = np.zeros(49)
    spf_shpere_extractJulien = np.zeros(49)
    errors_sphere_extractJulien = np.zeros(49)

    i = 0
    with open(os.path.join(basedir, 'SPHERE_Hdata', 'SPHERE_extraction_Milli.csv'),
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

    dir_exctract_Pauline = os.path.join(basedir, '150403_K1_Spec',
                                        'Pauline_SPF_from_Pol')

    data_exctract_Pauline = os.path.join(dir_exctract_Pauline,
                                        'Pauline_SPF_ktot_NE_errs.txt')
    text_array = np.loadtxt(data_exctract_Pauline)
    angles_GPIPolK1_extractPauline = text_array[:, 0]
    spf_GPIPolK1_extractPaulineNE = text_array[:, 1]
    errors_GPIPolK1_extractPaulineNE = text_array[:, 2]

    data_exctract_Pauline = os.path.join(dir_exctract_Pauline,
                                        'Pauline_SPF_ktot_SW_errs.txt')

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

    spf_sphere_h = measure_spf_errors('SPHERE_Hband_MCMC', nb_random_models)
    spf_sphere_h_3g = measure_spf_errors('SPHERE_Hband_3g_MCMC', nb_random_models)
    spf_gpi_not1at90 = measure_spf_errors('GPI_Hband_MCMC',
                                        nb_random_models,
                                        Norm_90_inplot=0.93)
    spf_gpi_h_1at90 = measure_spf_errors('GPI_Hband_MCMC', nb_random_models)
    spf_gpi_j = measure_spf_errors('GPI_Jband_MCMC', nb_random_models)
    spf_gpi_k1 = measure_spf_errors('GPI_K1band_MCMC', nb_random_models)
    spf_gpi_k2 = measure_spf_errors('GPI_K2band_MCMC', nb_random_models)

    spf_gpi_h_fake = measure_spf_errors('GPI_Hband_fake_MCMC', nb_random_models)

    spf_gpi_h_save = measure_spf_errors('GPI_Hband_MCMC', nb_random_models, save = True)

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

    plot.fill_between(scattered_angles,
                    spf_sphere_h['errorbar_sup'],
                    spf_sphere_h['errorbar_inf'],
                    facecolor=color0,
                    alpha=0.1)

    plot.fill_between(scattered_angles,
                    spf_gpi_not1at90['errorbar_sup'],
                    spf_gpi_not1at90['errorbar_inf'],
                    facecolor=color1,
                    alpha=0.1)

    plot.plot(scattered_angles,
            spf_sphere_h['best_spf'],
            linewidth=2,
            color=color0,
            label="SPHERE H2 (extraction MCMC, this work)")

    plot.plot(scattered_angles,
            spf_gpi_not1at90['best_spf'],
            linewidth=2,
            color=color1,
            label="GPI H (extraction MCMC)")

    plot.errorbar(angles_sphere_extractJulien,
                1.38 * spf_shpere_extractJulien,
                yerr=errors_sphere_extractJulien,
                fmt='o',
                label='SPHERE H2 (previous extraction by M17)',
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

    plot.savefig(os.path.join(folder_save_pdf, name_pdf))

    plot.close()

    ####################################################################################
    ## plot all colors
    ####################################################################################
    name_pdf = 'compare_GPI_color.pdf'
    plot.figure()

    plot.fill_between(scattered_angles,
                    spf_gpi_j['errorbar_sup'],
                    spf_gpi_j['errorbar_inf'],
                    facecolor=color4,
                    alpha=0.1)

    plot.fill_between(scattered_angles,
                    spf_gpi_h_1at90['errorbar_sup'],
                    spf_gpi_h_1at90['errorbar_inf'],
                    facecolor=color1,
                    alpha=0.1)

    plot.fill_between(scattered_angles,
                    spf_sphere_h['errorbar_sup'],
                    spf_sphere_h['errorbar_inf'],
                    facecolor=color0,
                    alpha=0.1)

    plot.fill_between(scattered_angles,
                    spf_gpi_k1['errorbar_sup'],
                    spf_gpi_k1['errorbar_inf'],
                    facecolor=color2,
                    alpha=0.1)

    plot.fill_between(scattered_angles,
                    spf_gpi_k2['errorbar_sup'],
                    spf_gpi_k2['errorbar_inf'],
                    facecolor=color3,
                    alpha=0.1)

    plot.plot(scattered_angles,
            spf_gpi_j['best_spf'],
            linewidth=2,
            color=color4,
            label="GPI J (extraction MCMC)")

    plot.plot(scattered_angles,
            spf_gpi_h_1at90['best_spf'],
            linewidth=2,
            color=color1,
            label="GPI H (extraction MCMC)")

    plot.plot(scattered_angles,
            spf_sphere_h['best_spf'],
            linewidth=2,
            color=color0,
            label="SPHERE H2 (extraction MCMC)")

    plot.plot(scattered_angles,
            spf_gpi_k1['best_spf'],
            linewidth=2,
            color=color2,
            label="GPI K1 (extraction MCMC)")

    plot.plot(scattered_angles,
            spf_gpi_k2['best_spf'],
            linewidth=2,
            color=color3,
            label="GPI K2 (extraction MCMC)")

    plot.legend()
    plot.yscale('log')

    plot.ylim(bottom=0.3, top=30)
    plot.xlim(left=0, right=180)
    plot.xlabel('Scattering angles')
    plot.ylabel('Normalized total intensity')

    plot.tight_layout()

    plot.savefig(os.path.join(folder_save_pdf, name_pdf))

    plot.close()


    ####################################################################################
    ## 3g plot
    ####################################################################################
    name_pdf = 'compare_3g_SPF.pdf'
    plot.figure()

    plot.fill_between(scattered_angles,
                    spf_sphere_h['errorbar_sup'],
                    spf_sphere_h['errorbar_inf'],
                    facecolor=color0,
                    alpha=0.1)

    plot.fill_between(scattered_angles,
                    spf_sphere_h_3g['errorbar_sup'],
                    spf_sphere_h_3g['errorbar_inf'],
                    facecolor=color2,
                    alpha=0.1)

    plot.plot(scattered_angles,
            spf_sphere_h['best_spf'],
            linewidth=2,
            color=color0,
            label="2 HG SPF (MCMC best model)")


    plot.plot(scattered_angles,
            spf_sphere_h_3g['best_spf'],
            linewidth=2,
            color=color2,
            linestyle = '-.',
            label="3 HG SPF (MCMC best model)")


    plot.plot(scattered_angles,
            1.38 * hg3g_fitted_Milliextraction,
            linewidth=2,
            color=color3,
            label="3 HG SPF fitted to M17 SPF (MCMC initial point)")

    plot.errorbar(angles_sphere_extractJulien,
                1.38 * spf_shpere_extractJulien,
                yerr=errors_sphere_extractJulien,
                fmt='o',
                label='Previous extraction by M17',
                ms=3,
                capthick=1,
                capsize=2,
                elinewidth=1,
                markeredgewidth=1,
                color='grey')

    handles, labels = plot.gca().get_legend_handles_labels()
    order = [3, 2, 1, 0]
    plot.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    plot.yscale('log')

    plot.ylim(bottom=0.3, top=30)
    plot.xlim(left=0, right=180)
    plot.xlabel('Scattering angles')
    plot.ylabel('Normalized total intensity')

    plot.tight_layout()

    plot.savefig(os.path.join(folder_save_pdf, name_pdf))

    plot.close()


    ####################################################################################
    ## injected spf plot
    ####################################################################################
    name_pdf = 'Comparison_spf_injected_recovered.pdf'
    plot.figure()

    plot.fill_between(scattered_angles,
                    spf_gpi_h_fake['errorbar_sup'],
                    spf_gpi_h_fake['errorbar_inf'],
                    facecolor=color3,
                    alpha=0.1)


    plot.plot(scattered_angles,
            spf_gpi_h_fake['best_spf'],
            linewidth=2,
            color=color3,
            label="SPF Recoreved After MCMC")


    plot.plot(scattered_angles,
            injected_hg,
            linewidth=1.5 ,
            linestyle = '-.',
            color=color2,
            label="SPF Injected into Empty Dataset")


    handles, labels = plot.gca().get_legend_handles_labels()
    order = [1, 0]
    plot.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    plot.yscale('log')

    plot.ylim(bottom=0.3, top=30)
    plot.xlim(left=0, right=180)
    plot.xlabel('Scattering angles')
    plot.ylabel('Normalized total intensity')

    plot.tight_layout()

    plot.savefig(os.path.join(folder_save_pdf, name_pdf))

    plot.close()
