# Hidden-Structure-MaxEnt
The script "Hidden Structure L-BFGS-B.py" implements a Maximum Entropy learner (Goldwater and Johnson 2003) that can handle hidden structure learning problems (Tesar and Smolensky 2000, Jarosz 2013). The optimization algorithm used is L-BFGS-B (Byrd et al. 1995), combined with Expectation Maximization (Dempster et al. 1977).

## Input Data
- Sample training data (from Tesar and Smolensky 2000) can be viewed in the "Input Files" directory.
  - Each file represents a single stress pattern
  - The first four columns represent:
    - UR: the underlying representation of a word. Can be associated with multiple SR candidates.
    - SR: the surface representation that the learner has access to. Can be associated with multiple possible hidden structures.
    - p: the probability in the training data of getting the relevant SR, given the relevant UR.
    - HR: the hidden representation that the learner does not have direct access to. Includes any hidden structure.
- All subsequent columns give the violation profiles for their respective HR's. 

## Output Files
- The script creates two kinds of output files:
  - The main output files for each language are identical to the training data files, except that:
    - Constraint weights are included,
    - ...And the learner's expected probabilities (p_LE) are included in addition to the training data probabilities (p_TD in the output files)
  - A summary "successes" file is also printed, telling you which of the languages were successfully converged on (where success is defined as assigning >90% probability to each SR with a probability of 1 in the training data).

## Using the Model
- The model is made to run in Python 2, and requires the following packages:
  - numpy
  - sys
  - scipy
  - mpmath
  - re
  - os
- The script doesn't take commandline arguments--but there are some things you can easily customize in the "USER SETTINGS" section at the beginning of the code:
  - RAND_WEIGHTS: if this is "True", initial weights are randomly sampled from 1-10.
  - INIT_WEIGHT: If RAND_WEIGHTS is "False", weights are initialized to this value.

## References
- Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). A limited memory algorithm for bound constrained optimization. *SIAM Journal on Scientific Computing, 16(5)*, 1190�1208.
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B (Methodological), 39(1)*, 1�22.
- Jarosz, G. (2013). Learning with hidden structure in optimality theory and harmonic grammar: Beyond robust interpretive parsing. *Phonology, 30(1)*, 27�71.
- Tesar, B., & Smolensky, P. (2000). *Learnability in optimality theory*. Mit Press.


