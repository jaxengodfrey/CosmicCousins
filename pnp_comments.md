## Amanda's Comments

Hi Amanda, thanks for your very helpful comments and review of our paper. We have copied over each point of yours (outside of fixing typos of ours) below with our responses and uploaded the second version of the manuscript to the dcc with all of the corresponding changes. 

- Only public PE and injections are used. Though it might be good to specify which PE samples and injection sets are used and link to the corresponding data releases
  - We added a paragraph in the Results section detailing the PE and injection samples that we used 
- No, but as Tom Dent pointed out I think the projection for the number of events in O4 is a bit optimistic
  - We have reworded this more appropiately
- For the most part, yes. The only references I would add are https://arxiv.org/abs/2204.00461 in the place indicated in my attached PDF, plus the FGMC paper Tom Dent mentioned in the bottom of his comment. I feel that this work is quite different than the original FGMC paper and expands on it significantly, but it would still be good to cite it when mentioning discreet latent variables.
  - We added each of these references mentioned
- Maybe add 10.1088/0004-637X/810/1/58 ?
  - Referenced
- maybe you mean “…most able to clearly distinguish between different possible CO formation histories…” or something similar? all parameters are directly linked to the formation history
  - We have changed the wording accordingly
- Fishbach and Holz 2017 don’t find peaks, but they do identify a maximum BH mass - is that why you are citing it? If so, maybe remove “i.e. peaks” from the beginning of this sentece. You could replace it with “peaks and truncations” or something like that. Also, usually people put citations in order of year published, from earliest to latest
  - We have clarified this and fixed citation ordering.
- I’m assuming this is coming in a forthcoming appendix but it would be good to specify which data products you used (PE samples with link/citation to data release and which injection set you used with link/citation to data release)
  - This was fixed (see above)
- people also usually cite Loredo 2004
  - We have added this citation
- also cite https://arxiv.org/abs/2204.00461
  - Referenced
- should the subscripts on these be k_i rather than M? After our conversation, I do think the subscripts should be k_i, and all future instances of M should be used just to mean the number of subpopulations in a given model
  - We have fixed notation to be more clear
- I wouldn’t say there is evidence for the 25 solar mass peak. Maybe “Given the recent evidence for peaks in the mass spectrum at 10M⊙ and 35M⊙ , and suggestions of a a potential feature at 25M⊙ ... 
  - This wording was changed
- please put citations in order of publication: earlier ones should go first.
  - We fixed the ordering
- This is quite high, right?  and it seems like it would have a big impact on your resutls. What happens if you fit maximum mass? you could fix the power law slope instead if you want to keep the same number of fit parameters
  - Following what was done in Cover Your Basis paper we let the basis splines cover the full range of allowable masses with very wide priors that allow the spline to drastically reduce the merger rate should the data prefer some sharper truncation like feature in the population. We do see some signs of reversion towards the prior at m1>~90 Msun at the same place where models with truncations (see O3b paper) infer the maximum mass to be, which is at the highest possible mass such that all event posteriors have non-negligible support below. All this to say if we wanted to add a truncation to our model it finds it at the same mass as other parametric models and where the full non-truncated model reverts towards the prior. 
- Okay actually I am confused about what k and M are. I thought M was the total number of latent variables in a given model, and k was the index of that number. But here it just seems like M=k+1 and the total number of latent variables is not defined?
  - We have fixed this to be more clear
- might be good to reference a specific figure number :)
  - Fixed
- Can you test this by just plotting the prior predictive distribution and seeing if it is wider or the same width?
  - We have added prior predictive dist on the spin plots and updated the corresponding discussion in the results section.
  
## Tom's Comments

Hi Tom, thanks for the thorough interest in our work and these very useful comments on the draft, each of which we have tried to address below.

- At the top level, the 10 Msun peak was identified by the O3b LVK population paper, so if the title is 'Identification of .. BBH Formed Through Isolated Binary Evolution' I expect the content will be something beyond the O3b paper which shows that the BBH in this known peak are from isolated/field binaries. What comes out of this analysis as explained in the main text is that this peak appears consistent with field binary formation, with some caveats or pinches of salt, so this is not quite the same. 
  - ...
- To maybe oversimplify a little, one could say that the properties of the 10 Msun peak, overall low spin magnitudes, slight preference for aligned over anti-aligned spin orientation - are those of the detected population as a whole except for some outliers. Or at most the peak has somewhat more of a preference towards spin alignment. 
  - ...
- So already from the O3b paper it seems likely that the 10 Msun peak contains some isolated binaries, or at least is consistent with that. 
  - ...
- It looks like the extra element here is separating the mergers into categories which are intended to have different spin properties. However, what is tough for me to see is how the results _prove_ that the properties are different, when the relevant plots/numbers seem to show a lot of overlap
  - ...
- At the most basic level, is it necessary for the 10 Msun peak to be due to only one formation channel? I am guessing no, as the specific mass ~10 Msun may be a result of stellar physics, core collapse supernova dynamics etc. independent of formation channel. In general we might expect more than one channel to contribute at any given primary mass (at least below the PISN cutoff) - a la Zevin et al 2021
  - ...
- Also, as far as I understand the literature, spin tilts above 90deg (cos theta<0) are very difficult to obtain from isolated binaries. However, all the analyses here indicate there is a nonzero fraction of anti-aligned spins even for 'Peak A' (eg 0.35+0.16?0.13). So is this indicating that there must be some nontrivial isotropic-spin contribution to the 10 Msun peak? I.e. is something in the data preventing the fraction of negative tilts going to 0, which is what we naively expect for isolated binaries? 
  - ...
- The models in this work imply that there are trends in spin (and maybe mass ratio) over primary mass. Such trends/variations in spin properties over mass were investigated in a non-parametric way by the Tiwari/Fairhurst and Tiwari 2022 papers. (These are cited already, but only for the presence of multiple peaks.) If a significant trend exists, it could show up in those works. So maybe it would be worth discussing how far the results here differ from what was found earlier. (I guess due to the different modelling of spin components it may not be easy to make a direct comparison.)
  - ...
- Just trying to understand how the results correspond to the data - do we know why the (single) 'peak' being located at 10 Mun is so highly preferred to the case where the 'peak' is at 35 Msun and the excess at 10 would be in the 'continuum'? Do you get any samples with a 'peak' at 35 not 10? If so, how far is their likelihood smaller than those with mu_m~10? 
  - ...
- Is it necessarily the case that the preference for mu_m~10 over mu_m~35 is entirely driven by differences in spin properties? What is the prior on the mu_m, eg is it uniform or uniform-in-log(mu_m) ? Could the choice make a difference in the peak location inference? 
  - ...
- Also might the preference be partly due to the spline model: for example would it be more difficult (unlikely) for the 'continuum' model to accommodate the variation in rate over the region around 10 Msun than the variation around 35 Msun? 
  - ...
- Could it be checked or quantified whether the properties of just the mass spectrum are leading to any preference on the peak location? E.g. if one were to run replacing the spin samples with fake samples drawn from the prior, or just omitting the spin dimensions? (If that makes sense ..)
  - ...
- The summary statistics given for the tilt distributions of the peak 'A' vs continuum on p. 5 are not significantly different, i.e. the uncertainty intervals have a quite large overlap. So it seems hard to see from the (spin) data why the inference on the discrete categories is so strong. 
  - ...
- On p. 10 the question of whether the isotropic/uniform distributions for the high-mass events is 'informed' or not is interesting, could it be addressed by comparing the uncertainties obtained from the prior distributions with the posteriors? Or e.g. artificially omitting some fraction of these events and seeing whether the uncertainties get larger? 
  - ...
- To quantify the differences between the distributions shown in fig.5 which (only just) overlap at 90%, might it be useful to extract the mean or median chi_eff from each and see if these differ significantly? 
  - ...
- 'lower bound on the PISN mass gap 49 [..] Msun' - I guess this number is an estimate for the gap lower edge? 'Lower bound on' might be confusing wording.
  - We have fixed the wording of this to be more clear.

LVK Review:
- In the astrophysical implications section of the O3b LVK paper it is noted that the 10 Msun peak seems inconsistent with globular cluster models: "Taking these results at face value, the inferred high merger rate of sources with <~10M? may suggest that globular clusters contribute subdomi- nantly to the detected population." This seems highly relevant to the astrophysical claims here and should be mentioned as a previous attempt towards interpretation of this peak.
  - ...
- Fig. 19 of the O3b LVK paper shows a nontrivial difference in the inference on spins for events with mchirp>40 (primary>45-50). This suggests that while lower-mass binaries are constrained to have mainly low spins, the spin properties of high-mass binaries *may* be different.  Similarly, Hoy et al. (2110.13542) find that 'most binaries with primary mass greater than 50M tend to prefer larger spins ..'.  These findings hinting at a different behaviour for masses above ~50 Msun should be mentioned. 
  - ...
- p.10 'Next observing run .. Expected to increase the catalog size tenfold' - I would not say this, it amounts to a forecast of future LIGO performance which should be referenced to official materials. Currently publicly visible estimates for the O4 BBH detection count are in the low few hundreds (200-300) - see the EM user guide, for instance - which is consistent with R/P internal estimates. 'Up to fivefold' might be more realistic/supportable by LVK. 
  - ...
- Concerning references to previous work, the discrete latent variable approach was introduced to GW population work in the now classic 'FGMC' paper (1302.5341), but without including hierarchical inference - so one could think of the method here as a much more sophisticated extension of FGMC.
  - ...
