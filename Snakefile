rule create_macros:
    input:
        "src/data/bspline_1logpeak_100000s.h5",
        "src/data/bspline_1logpeak_100000s_ppds.h5",
        "src/data/bspline_1logpeak_samespin_100000s_2chains.h5",
        "src/data/b1logpeak_marginalized_50000s_2chains.h5",
    output:
        "src/tex/macros.tex", 
        "src/data/macros.json",
    priority: 
        100
    conda:
        "environment.yml"
    script:
        "src/scripts/create_macros.py"

