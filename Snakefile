rule create_macros:
    input:
        "src/data/bspline_mass_spin_1000w_10000s_thin2_independent_bspline_ratio_reweighedKDEs_12-16-22.h5",
        "src/data/PPDS_bspline_mass_spin_1000w_10000s_thin2_independent_bspline_ratio_sigprior02_12-15-22.h5",
    output:
        "src/tex/output/macros.tex", 
        "src/data/macros.json",
    priority: 
        100
    conda:
        "environment.yml"
    script:
        "src/scripts/create_macros.py"