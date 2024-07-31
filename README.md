# Hidden-Structure-MaxEnt
The script "Hidden_Structure_MaxEnt.py" implements a Maximum Entropy learner (Goldwater and Johnson 2003) that can handle hidden structure learning problems (Tesar and Smolensky 2000, Jarosz 2013). The model can use three different optimization algorithms: Gradient Descent, Conjugate Gradient, and L-BFGS-B (Byrd et al. 1995), combining whichever algorithm you choose with Expectation Maximization (Dempster et al. 1977) to deal with the presence of hidden structure. For more information on how this software works (and why it's necessary), see [the AMP paper](https://journals.linguisticsociety.org/proceedings/index.php/amphonology/article/view/5175) where we published some of our results.

## Input Data
- Sample training data (from Tesar and Smolensky 2000) can be viewed in the "Input_Files/TesarSmokensky-OG" directory. The directory "Input_Files" needs to keep the same name, but you can have multiple subdirectories inside of it named whatever you like (the name of the subdirectory you want to use when running a simulation is specified using the "LANG_SET" variable in the "Hidden_Structure_MaxEnt.py" script).
  - Each file represents a single stress pattern that can be gradient (probabilities on surface representations ranging between 0 and 1, with each underlying representations having a total probability of 1) or categorical (all underlying representations have exactly one surface representation whose probability is 1, with all others being zero).
  - The first four columns of each file represent:
    - UR: the underlying representation of a word. Can be associated with multiple SR candidates.
    - SR: the surface representation that the learner has access to. Can be associated with multiple possible hidden structures.
    - p: the probability in the training data of getting the relevant SR, given the relevant UR.
    - HR: the hidden representation that the learner does not have direct access to. Includes any hidden structure.
- All subsequent columns give the violation profiles for their respective HR's.

## Output Files
- Sample output data (obtained by running the model with L-BFGS-B and initial weights of 1 on the Tesar and Smolensky 2000 data) can be viewed in the "Output_Files" directory. Be sure to include a directory named "Output_Files", as the script will be expecting that to exist.
- The script creates three kinds of output files:
  - The main output files for each language have the same information as the training data files, with the addition of:
    - The constraint weights at the end of learning, beneath each constraint.
    - The learner's expected probabilities for each SR, labeled "p(SR)_LE". Note that the SR probabilities from the training data are labeled "p(SR)_TD" in this file.
    - The learner's expected probability for each HR, normalized by dividing by the other HRs' probabilities for that UR. This is labeled "p(HR)_normed".
    - The learner's expected probability for each HR without any normalization, "p(HR)_absolute". 
  - A "brief" output file that only prints, for each UR, information about the HR that the model gives the most probabilty to, given that UR as input. The information provided includes: 
    - "p_TD", which is the same as "p(SR)_TD" above.
    - "p_absolute", which is the same as "p(HR)_absolute" above.
    - "p_normed" which is the same as "p(HR)_normed" above.
    - Each HR's constraint violations.
    - Each constraint's weights.
  - A summary "successes" file is also printed, telling you which of the languages in the simulation's subdirectory were successfully learned. (Where success is defined as assigning >90% probability to each SR with a probability of 1 in the training data--note that this criterion only makes sense for categorical patterns!)
  - If you're using gradient descent, an "EpochsToConvergence" file will also be created. This records how many epochs (full passes through the training data) were required for the model to converge on each pattern it was able to successfully learn. Failed patterns have a value of -1 in this file. (Again, the convergence criterion here only makes sense for categorical patterns!)

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
  - INIT_WEIGHT: If RAND_WEIGHTS is "False", weights are initialized to this value. If RAND_WEIGHTS is "True", this variable is ignored.
  - L2_PRIOR: If this is "True", the objective function will penalize higher constraint weights (this doesn't work if the algorithm you're using is gradient descent).
  - LAMBDA: this controls how much higher weights are penalized if L2_PRIOR is set to "True". If L2_PRIOR is set to "False", this value is ignored.
  - NEG_WEIGHTS: if this is "False", negative weights aren't allowed (this doesn't work for the Conjugate Gradient algorithm, though).
  - METHOD: which optimization algorithm do you want to use? Choices are "lbfgsb" (L-BFGS-B; this is the recommended method if you want to maximize the chances that the model will converge), "cg" (Conjugate Gradient), and "gd" (Gradient Descent; this is the only method that will keep track of how many epochs it took to learn a pattern).
  - EPOCHS: how many full passes through the training data do you want the model to perform? Note that this variable only relevant if you're using Gradient Descent. For more on how the other algorithms operate, see [their documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize).
  - ETA: What learning rate do you want the model use? That is, how much should it scale each weight's gradient for each update in learning? Note that this is only relevant if you're using Gradient Descent. For more on how the other algorithms operate, see [their documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize).

## References
- Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). A limited memory algorithm for bound constrained optimization. *SIAM Journal on Scientific Computing, 16(5)*, 1190–1208.
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B (Methodological), 39(1)*, 1–22.
- Jarosz, G. (2013). Learning with hidden structure in optimality theory and harmonic grammar: Beyond robust interpretive parsing. *Phonology, 30(1)*, 27–71.
- Tesar, B., & Smolensky, P. (2000). *Learnability in optimality theory*. Mit Press.


