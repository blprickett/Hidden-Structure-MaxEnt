import numpy as np
from scipy.stats import entropy
from scipy.optimize import minimize
import mpmath
from re import sub, search
from os import listdir, path
from datetime import datetime

#####USER SETTINGS#####  
RAND_WEIGHTS = False #Are intial weights random (or all set to the value below)?
INIT_WEIGHT = 1.0

#####CUSTOM FUNCTIONS##### 
def get_predicted_probs (weights, viols):
    harmonies = viols.dot(weights)
    eharmonies = np.exp(harmonies)
    
    #Calculate denominators to convert eharmonies to predicted probs:
    Zs = np.zeros(viols.shape[0])
    for underlying_form in unique_urs:     #Sum of eharmonies for this UR (converts to probs)   
        this_Z = sum(eharmonies[ur2datum[underlying_form]])\
                                        *float(UR_num) #Number of UR's (normalizes the updates)
        #Deal with rounding errors:
        if this_Z == 0:
            this_Z = sum(eharmonies[ur2datum[underlying_form]])\
                        *mpmath.mpf(UR_num) #Number of UR's (normalizes the updates)
            this_Z = float(this_Z)
            
        #Save the Z:
        Zs[ur2datum[underlying_form]] = this_Z
    
    #Calculate prob for each datum:
    probs = []
    for datum_index, eharm in enumerate(eharmonies):
        if Zs[datum_index]==0:
            error_f = open("_weights.csv", "w")
            error_f.write("\n".join([str(w) for w in weights]))
            error_f.close()
            raise Exception("Rounding error! (Z=0 for "+ur[datum_index]+")")
        else:
            probs.append(eharm/Zs[datum_index])
       
    return np.array(probs)

def exact_predicted_probs (weights, viols):
    harmonies = viols.dot(weights)
    eharmonies = np.array([mpmath.exp(h) for h in harmonies])
    
    #Calculate denominators to convert eharmonies to predicted probs:
    Zs = np.zeros(viols.shape[0])
    for underlying_form in unique_urs:     #Sum of eharmonies for this UR (converts to probs)   
        this_Z = sum(eharmonies[ur2datum[underlying_form]])\
                                        *float(UR_num) #Number of UR's (normalizes the updates)
        #Deal with rounding errors:
        if this_Z == 0:
            this_Z = sum(eharmonies[ur2datum[underlying_form]])\
                        *mpmath.mpf(UR_num) #Number of UR's (normalizes the updates)
            this_Z = float(this_Z)
            
        #Save the Z:
        Zs[ur2datum[underlying_form]] = this_Z
    
    #Calculate prob for each datum:
    probs = []
    for datum_index, eharm in enumerate(eharmonies):
        if Zs[datum_index]==0:
            raise Exception("Rounding error (exact function)! (Z=0 for "+ur[datum_index]+")")
        else:
            probs.append(eharm/Zs[datum_index])
       
    return np.array(probs)    

def objective_func (weights, viols, td_probs, SRs):
    global m_t
    global v_t
    
    #Forward pass:
    le_probs = get_predicted_probs (np.array(weights), viols)
    
    #Weight the td_probs, based on what we know about the
    #different hidden structures:
    sr2totalLEProb = {form:sum(le_probs[sr2datum[form]]) for form in sr2datum.keys()} #Sums expected SR probs (merging different HR's)
    if 0.0 in sr2totalLEProb.values():
        ex_le_probs = exact_predicted_probs(np.array(weights), viols)
        sr2totalLEProb = {form:float(sum(ex_le_probs[sr2datum[form]])) for form in sr2datum.keys()}
    sr2totalTDProb = {form:sum(td_probs[sr2datum[form]]) for form in sr2datum.keys()} #Sums raining data SR probs (merging different HR's)
    weighted_tdProbs = []
    for datum_index, le_p in enumerate(le_probs):
        if sr2totalLEProb[SRs[datum_index]]==0.0:
            HR_givenSR = 0.0
        else:
            HR_givenSR = le_p/sr2totalLEProb[SRs[datum_index]] #How likely is the HR|SR, given our current grammar
        weighted_tdProbs.append(HR_givenSR * sr2totalTDProb[SRs[datum_index]]) #Weight the HR probs in the training data by our current estimation of HR probs
    
    #Calculate loss:
    return entropy(weighted_tdProbs, le_probs)
    
    
######LOOP THROUGH ALL LANGUAGES######
my_time = sub(":", ".", str(datetime.now()))
success_file = open(path.join("Output_Files", "successes_"+my_time+".csv"), "w") 
success_file.write("Language,Successful?\n")
input_files = listdir("Input_Files")
test_langs = [sub("[^0-9]", "", l) for l in input_files]

for lang_index, language in enumerate(test_langs):      
    #####TRAINING DATA##### 
    #Needs to create three numpy arrays:
    #    >w: array of weights of length C
    #    >v: 2d array of violations, height=D and width=C   
    #    >p: array of training data probabilities of length D
    #
    #...And a dictionary called "sr2datum" that maps SR's to a list of
    #   the data (i.e. indeces in v and p) that they're associated with.

    print ("Processing input file #"+str(language))
       
    #Get constraint names:
    tableaux_file = open(path.join("Input_Files", input_files[lang_index]), "r")
    headers = tableaux_file.readline().rstrip().split(",")
    CON = headers[4:]
    
    #Get violation vectors:
    v = []
    probs = []
    sr = []
    ur = []
    hr = []
    input_lines = []
    for row in tableaux_file.readlines():
        input_lines.append(row.rstrip().split(","))
        ur_line = search("^([^,]+)\n", row)
        sr_line = search("^,([^,]+),([^,]+)", row)
        hr_line = search("^,,,([^,]+),(.+)", row)
        
        if ur_line:
            my_in = ur_line.group(1)
            continue
        elif sr_line:
            my_out = sr_line.group(1)
            my_prob = float(sr_line.group(2))
            continue
        elif hr_line:
            my_hid = hr_line.group(1)
            raw_viols = hr_line.group(2).rstrip().split(",")
            my_viols = [-1 * int(viol) for viol in raw_viols]  
        else:
            raise Exception("Error in Training Data File! (line: "+row.rstrip()+")")
        
        hr.append(my_hid)
        sr.append(my_out)
        ur.append(my_in)
        v.append(my_viols)
        probs.append(my_prob)
    
    unique_srs = sorted(list(set(sr)))#Sorted list of SR's
    unique_urs = sorted(list(set(ur)))#Sorted list of UR's
    UR_num = len(unique_urs)
    
    #Dictionaries that we need:
    sr2datum = {form:[] for form in unique_srs} #SR --> [data indeces]
    for datum_index, surface_form in enumerate(sr):
        sr2datum[surface_form].append(datum_index)
    ur2datum = {form:[] for form in unique_urs} #UR --> [data indeces]
    for datum_index, underlying_form in enumerate(ur):
        ur2datum[underlying_form].append(datum_index)
    
    #Tweak the initial probabilities so that they make sense:
    new_probs = [] 
    probs = np.array(probs)
    for datum_index, this_prob in enumerate(probs):
        new_prob = this_prob/(len(sr2datum[sr[datum_index]])*UR_num)
        new_probs.append(new_prob)
      
    #Vectors that we need: 
    if RAND_WEIGHTS:
        w = list(np.random.uniform(low=0.0, high=10.0, size=len(v[0])))   #Init constraint weights = rand 1-10
        print ("Initial weights: ", w)
    else:  
        w = [INIT_WEIGHT for c in v[0]]  #Init constraint weights = INIT_WEIGHT
    v = np.array(v)                   #Constraint violations
    p = np.array(new_probs)  
    
    #####LEARNING##### 
    print (" ...Learning...")
    final_weights = minimize(objective_func, w, args=(v, p, sr), method="L-BFGS-B", bounds=[(0.0, 200) for x in w])['x']
    current_probs = get_predicted_probs(np.array(final_weights), v)
           
    #####OUTPUT##### 
    print ("...Saving output...")
    #Main Output file:
    output_file = open(path.join("Output_Files", language+"_Output_"+my_time+".csv"), "w")
    new_headers = headers[:2]+["p_TD", "p_LE"]+headers[3:]
    output_file.write(",".join(new_headers)+"\n,,,,,")
    for fw in final_weights:
        output_file.write(str(fw)+",")
    output_file.write("\n")
    datum_index = 0
    for old_line in input_lines:
        if len(old_line) < 4:
            output_file.write(",".join(old_line)+"\n") 
        else:
            new_line = old_line[:2]+["", str(current_probs[datum_index])]+old_line[3:]
            output_file.write(",".join(new_line)+"\n")
            datum_index += 1   
    output_file.close()

    #Success file:
    learned = True
    for datum_index, form in enumerate(sr):
        if probs[datum_index] != 1: 
            #We're only checking correct forms, skip the incorrect ones.
            continue
        SR_indeces = sr2datum[form] #Find all the training data that use this SR
        UR_indeces = ur2datum[ur[datum_index]] #Find all the training data that use this UR
        predicted_SRprob = sum(current_probs[SR_indeces]) #Sum the SR probs (merges different HR's)
        predicted_URprob = sum(current_probs[UR_indeces]) #Sum the UR probs (merges different SR's and HR's)
        if predicted_URprob == 0:
            raise Exception("Rounding error! pr(UR)=0")
        else:
            conditional_prob = predicted_SRprob/predicted_URprob #Find the prob of this SR, given its UR
        if conditional_prob < .9: #If >90% of prob isn't given to the correct SR...
            learned = False
                                 
    #Record whether we met the criterion:
    if learned:
        for cp in current_probs:
            if np.isnan(cp):
                raise Exception("Found a NAN! Possible rounding error in probabilities.")
        print ("Language "+language+" was successfully learned.")
        success_file.write(language+",1\n")
    else:
        print ("Language "+language+" was NOT learned.")
        success_file.write(language+",0\n")
        
#Deal with output files:
success_file.close()
