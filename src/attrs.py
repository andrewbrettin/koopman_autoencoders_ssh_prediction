"""
attrs.py

Project attributes: global variables and path names.
"""

MACHINE = 'casper'

GLOBALS = {
    'init_years': [1251, 1281, 1301],
    'members': ['011', '012', '013'],
    'earth_radius': 6378.1,
    'gravity': 9.81,
    'rho_ref': 1025,
    'variables': ['SSH', 'SST', 'MLD', 'PSL', 'UBOT', 'VBOT'],
    'train_ids': [0, 1, 2, 3, 4, 6, 7],
    'val_id': 5,
    'test_id': 8,
}

if MACHINE == 'local':
    PATHS = {}

elif MACHINE == 'casper':
    PATHS = {
        # Filesystem
        "home":          "~/",
        "scratch":       "$SCRATCH/",
        "tmp":           "/tmp/",

        # Project-specific
        "project":       "~/koopman_autoencoders_ssh_prediction/",
        "src":           "~/koopman_autoencoders_ssh_prediction/src/",
        "data":          "$SCRATCH/koopman/",
        # Subprojects
        "subprojects":             "$SCRATCH/koopman/subprojects/",
        "cnn_pacific_monthly":             "$SCRATCH/koopman/subprojects/cnn_pacific_monthly/",
        "cnn_pacific_daily_subsampled": "$SCRATCH/koopman/subprojects/cnn_pacific_daily_subsampled/",
        "cnn_north_atlantic_daily_subsampled": "$SCRATCH/koopman/subprojects/cnn_north_atlantic_daily_subsampled/",
        "cnn_north_atlantic_monthly":             "$SCRATCH/koopman/subprojects/cnn_north_atlantic_monthly/",        

        # Machine learning specific data
        "ml":            "$SCRATCH/koopman/ml/",
        "tensors":       "$SCRATCH/koopman/ml/tensors/",
        "networks":      "$SCRATCH/koopman/ml/networks/",
        "logs":          "$SCRATCH/koopman/ml/logs/",

        # Other data
        "inference":     "$SCRATCH/koopman/inference/",
        "computation":   "$SCRATCH/koopman/computation/",
        "visualization": "$SCRATCH/koopman/visualization/",

        # Casper-specific paths
        "cesm":          "/glade/campaign/cgd/cesm/CESM2-LE/",
        "atm_daily":     "/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/",
        "ocn_daily":     "/glade/campaign/cgd/cesm/CESM2-LE/ocn/proc/tseries/day_1/",
        "grid":          "$SCRATCH/grid/",
        "regridded":     "$SCRATCH/regridded/",
        "sea_level":     "$SCRATCH/sea_level/",
        "rechunked":     "$SCRATCH/rechunked/",
        "detrended_deseasonalized": "$SCRATCH/detrended_deseasonalized/",
        "anom_spatial":  "$SCRATCH/anom_spatial/",
        "coarsened":     "$SCRATCH/coarsened/",
        "monthly_anom":  "$SCRATCH/monthly_anom/",
        "5day_anom":     "$SCRATCH/5day_anom/",
    }
