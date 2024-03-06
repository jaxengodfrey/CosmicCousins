rule create_macros:
    input:
    "src/data/bspline_1logpeak_marginalized_fixtau_m1-s25-z1_msig15_qsig5_ssig5_zsig1_sigp3_NeffNobs_downsample_100k_rng1-2_ppds.h5"
    "src/data/bspline_composite_marginalized_fixtau_m1-s25-z1_msig15_qsig5_ssig5_zsig1_sigp3_NeffNobs_downsample_100k_rng6-10_ppds.h5"
    "src/data/bspline_1logpeak_marginalized_fixtau_m1-s25-z1_msig15_qsig5_ssig5_zsig1_sigp3_NeffNobs_full_200ks.h5"
    "src/data/bspline_composite_marginalized_fixtau_m1-s25-z1_msig15_qsig5_ssig5_zsig1_sigp3_NeffNobs_full_500ks_rng6-10.h5"
    output:
        "src/tex/macros.tex", 
        "src/data/macros.json",
    priority: 
        100
    conda:
        "environment.yml"
    script:
        "src/scripts/create_macros.py"

