# Hidden-Structure-MaxEnt
Maximum entropy learner that can handle hidden structure learning problems.

## Input Data

	-Sample training data (from Tesar and Smolensky 2000) can be viewed in the "Tesar and Smolenksy Langs" folder
	-Each file represents a single stress pattern
	-The first four columns represent:
		*UR: the underlying representation of a word. Can be associated with multiple SR candidates.
		*SR: the surface representation that the learner has access to. Can be associated with multiple possible hidden structures.
		*p: the probability in the training data of getting the relevant SR, given the relevant UR.
		*HR: the hidden representation that the learner does not have direct access to. Includes any hidden structure.
	-All subsequent columns give the violation profiles for their respective HR's. 

## Using the Model

	-The model is made to run in Python 2, and requires the following packages:
		*numpy
		*sys
		*scipy
		*mpmath
		*re
		*os

	-The script doesn't take commandline arguments--but there are some things you can easily customize in the "USER SETTINGS" section at the beginning of the code:
		*RAND_WEIGHTS: if this is "True", initial weights are randomly sampled from 1-10. If it's "False", they're always 1.
		*TD_DIR: this is the directory that the training data is in.
		*PREFIX: this is a prefix that should be at the beginning of all your training data (and that will appear in the script's output files).


