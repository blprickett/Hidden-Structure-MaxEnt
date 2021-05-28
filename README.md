# Hidden-Structure-MaxEnt
The script "Hidden_Structure_MaxEnt.py" implements a Maximum Entropy learner (Goldwater and Johnson 2003) that can handle hidden structure learning problems (Tesar and Smolensky 2000, Jarosz 2013). The model can use three different optimization algorithms: Gradient Descent, Conjugate Gradient, and L-BFGS-B (Byrd et al. 1995), combining whichever algorithm you choose with Expectation Maximization (Dempster et al. 1977) to deal with the presence of hidden structure. For more information on this software (or why it's necessary), see the slides in "MFM Slides - Prickett and Pater 2019.pdf".

## Input Data
- Sample training data (from Tesar and Smolensky 2000) can be viewed in the "Input_Files/Tesar" directory. The directory "Input_Files" needs to keep the same name, but you can have multiple subdirectories inside of it named whatever you like (the name of the subdirectory you want to use is specified using the "LANG_SET" variable in the "Hidden_Structure_MaxEnt.py" script).
  - Each file represents a single stress pattern
  - The first four columns represent:
    - UR: the underlying representation of a word. Can be associated with multiple SR candidates.
    - SR: the surface representation that the learner has access to. Can be associated with multiple possible hidden structures.
    - p: the probability in the training data of getting the relevant SR, given the relevant UR.
    - HR: the hidden representation that the learner does not have direct access to. Includes any hidden structure.
- All subsequent columns give the violation profiles for their respective HR's. 

## Output Files
- Be sure to include a directory named "Output_Files", as the script will be expecting that to exist.
- The script creates three kinds of output files:
  - The main output files for each language are identical to the training data files, except that:
    - Constraint weights are included...
    - ...And the learner's expected probabilities (p_LE) are included in addition to the training data probabilities (p_TD in the output files). Note, however, that p_LE will be pr(HR), while p_TD will be the pr(SR|UR).
  - A "brief" output file that only prints, for each UR, information about the HR that the model gives the most probabilty to, given that UR as input. The information provided includes: 
    - p_TD (see description above)
    - p_absolute (same as p_LE in the full output files)
    - p_normed, which is the probability the model gives that HR, given the UR. That is, the probability of that HR divided by the probabilities of all the other HR's that UR can map to
    - Each HR's constraint violations
    - Each constraint's weights
  - A summary "successes" file is also printed, telling you which of the languages were successfully converged on (where success is defined as assigning >90% probability to each SR with a probability of 1 in the training data--note that this criterion only makes sense for categorical patterns!).

## Using the Model
- The model is made to run in Python 3, and requires the following packages (os, sys, and re should have been installed automatically with Python):
  - numpy
  - datetime
  - scipy
  - mpmath
  - re
  - os
- The script doesn't take command line arguments--but there are some things you can easily customize in the "USER SETTINGS" section at the beginning of the code:
  - LANG_SET: as mentioned above, this is the name of the subdirectory inside of "Input_Files" that you have all of your training data files saved in.
  - RAND_WEIGHTS: if this is "True", initial weights are randomly sampled from 1-10.
  - INIT_WEIGHT: If RAND_WEIGHTS is "False", weights are initialized to this value.
  - L2_PRIOR: If this is "True", the objective function will penalize higher constraint weights (this doesn't work if the algorithm you're using is gradient descent).
  - LAMBDA: this controls how much higher weights are penalized if L2_Prior is set to "True".
  - NEG_WEIGHTS: if this is "False", negative weights aren't allowed (this doesn't work for the Conjugate Gradient algorithm, though).
  - METHOD: which optimization algorithm do you want to use? Choices are "lbfgsb" (for L-BFGS-B; this is the recommended method), "cg" (for Conjugate Gradient), and "gd" (for Gradient Descent).

## References
- Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). A limited memory algorithm for bound constrained optimization. *SIAM Journal on Scientific Computing, 16(5)*, 1190–1208.
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B (Methodological), 39(1)*, 1–22.
- Jarosz, G. (2013). Learning with hidden structure in optimality theory and harmonic grammar: Beyond robust interpretive parsing. *Phonology, 30(1)*, 27–71.
- Tesar, B., & Smolensky, P. (2000). *Learnability in optimality theory*. Mit Press.


