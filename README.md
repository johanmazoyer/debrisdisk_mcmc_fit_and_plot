# debrisdisk_mcmc_fit_and_plot #

A python code to model disks for coronagraphic instruments (GPI and SPHERE). 

Initial model by Max Millar Blanchaer from a James Graham's original code, but can be easily change to other disk model. If you are using this model, please cite 
Millar-Blanchaer, M. A., Graham, J. R., Pueyo, L., et al. 2015, ApJ, 811, 18

The ADI and RDI effects on the disk are modelled using Forward Modelling tools in PyKLIP and specifically the DiskFM tools. Please cite
Wang, J. J., Ruffio, J.-B., De Rosa, R. J., et al. 2015, ASCL, 1506.001
Mazoyer J., Arriaga. P. et al. 2020 SPIE , in prep

Finally we use emcee MCMC framework to explore the parameter space. Please cite :
corner package: Foreman-Mackey, D. 2016, JOSS, 1, 24
emcee package: Foreman-Mackey, D., Hogg, D. W., Lang, D., & Goodman, J. 2013, PASP, 125, 306 

