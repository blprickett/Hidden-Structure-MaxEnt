import numpy as np
from scipy.stats import entropy
from scipy.optimize import minimize
import mpmath
from re import sub, search
from os import listdir, path
from datetime import datetime

#####USER SETTINGS#####  
LANG_SET = "TesarSmolensky" #The subdirectory that you have all your training data in 
RAND_WEIGHTS = False #Are intial weights random (or all set to the value below)?
INIT_WEIGHT = 1.0 #Initial weights for the model
L2_PRIOR = False #This only works for L-BFGS-B and Conjugate Gradient
L1_PRIOR = False #This only works for L-BFGS-B and Conjugate Gradient
LAMBDA = 0.001 #Only matters if you're using the prior
NEG_WEIGHTS = False #Are negative weights allowed? (Doesn't work for Conjugate Gradient)
METHOD = "gd" #gd (for vanilla Gradient Descent), lbfgsb (for L-BFGS-B), or cg (for Conjugate Gradient)

#Things you only specify for gradient descent (the other two algorithms mostly use defaults from scipy.optimize):
EPOCHS = 10000 #Number of passes through the data
ETA = 4. #learning rate

#####CUSTOM FUNCTIONS##### 
def get_predicted_probs (weights, viols):
    harmonies = viols.dot(weights)
    eharmonies = np.exp(harmonies)
    
    #Calculate denominators to convert eharmonies to predicted probs:
    Zs = np.array([mpmath.mpf(0.0) for z in range(viols.shape[0])])
    for underlying_form in unique_urs:     #Sum of eharmonies for this UR (converts to probs)   
        this_Z = sum(eharmonies[ur2datum[underlying_form]])\
                                        *float(UR_num) #Number of UR's (normalizes the updates)
        if this_Z == 0:
            eharmonies = np.array([mpmath.exp(h) for h in harmonies])
            this_Z = sum(eharmonies[ur2datum[underlying_form]])\
                                        *float(UR_num)
            Zs[ur2datum[underlying_form]] = this_Z
        else:
            Zs[ur2datum[underlying_form]] = mpmath.mpf(this_Z)

    #Calculate prob for each datum:
    probs = []
    for datum_index, eharm in enumerate(eharmonies):
        if Zs[datum_index] == 0:
            #error_f = open("_weights.csv", "w")
            #error_f.write("\n".join([str(w) for w in weights]))
            #error_f.close()
            #print("\n\n"+remember_me+"\n\n")
            raise Exception("Rounding error! (Z=0 for "+ur[datum_index]+")")
        else:
            probs.append(float(eharm/Zs[datum_index]))
       
    return np.array(probs)

def exact_predicted_probs (weights, viols):
    harmonies = viols.dot(weights)
    eharmonies = np.array([mpmath.exp(h) for h in harmonies])
    
    #Calculate denominators to convert eharmonies to predicted probs:
    Zs = np.array([mpmath.mpf(0.0) for z in range(viols.shape[0])])
    for underlying_form in unique_urs:     #Sum of eharmonies for this UR (converts to probs)   
        this_Z = sum(eharmonies[ur2datum[underlying_form]])\
                                        *float(UR_num) #Number of UR's (normalizes the updates)
        if this_Z == 0:
            eharmonies = np.array([mpmath.exp(h) for h in harmonies])
            this_Z = sum(eharmonies[ur2datum[underlying_form]])\
                                        *float(UR_num)
            Zs[ur2datum[underlying_form]] = this_Z
        else:
            Zs[ur2datum[underlying_form]] = mpmath.mpf(this_Z)
    
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
        #raise Exception("Need to use 'exact_probs' function!")
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
    if L2_PRIOR:
        prior = np.sum(np.square(weights))
        loss = entropy(weighted_tdProbs, le_probs) + (LAMBDA * prior)
    elif L1_PRIOR:
        prior = np.sum(np.absolute(weights))
        loss = entropy(weighted_tdProbs, le_probs) + (LAMBDA * prior)    
    else:
        loss = entropy(weighted_tdProbs, le_probs)
    
    return loss

def gradient_descent (weights, viols, td_probs, SRs, epoch_num):
    if epoch_num==0:
        return weights
        
    for epoch in range(epoch_num):
        if epoch!=0:
            weights = np.copy(new_weights)
            
        #print(epoch, " out of ", epoch_num, ": ", objective_func(weights, viols, td_probs, SRs))
            
        #Forward pass:
        le_probs = get_predicted_probs (weights, viols)
        
        #Weight the td_probs, based on what we know about the
        #different hidden structures:
        sr2totalLEProb = {form:sum(le_probs[sr2datum[form]]) for form in sr2datum.keys()} #Sums expected SR probs (merging different HR's)
        sr2totalTDProb = {form:sum(td_probs[sr2datum[form]]) for form in sr2datum.keys()} #Sums remaining data SR probs (merging different HR's)
        weighted_tdProbs = []
        for datum_index, le_p in enumerate(le_probs):
            if sr2totalLEProb[SRs[datum_index]]==0.0:
                #exit("Got a zero when you didn't want one!")
                HR_givenSR = 0.0
            else:
                HR_givenSR = le_p/sr2totalLEProb[SRs[datum_index]] #How likely is the HR|SR, given our current grammar
            weighted_tdProbs.append(HR_givenSR * sr2totalTDProb[SRs[datum_index]]) #Weight the HR probs in the training data by our current estimation of HR probs

        #Backward pass:
        TD = viols.T.dot(weighted_tdProbs) #Violations present in the training data
        LE = viols.T.dot(le_probs) #Violations expected by the learner
        gradients = (TD - LE)  

        #Update weights:
        updates = gradients * ETA
        new_weights = weights + updates
        
        #Police negative weights:
        if not NEG_WEIGHTS:
            new_weights = np.maximum(new_weights, 0)
            
        #Check to see if we've learned the language already:
        learned = True
        for datum_index, form in enumerate(SRs):
            if td_probs[datum_index] <= 0:
                #We're only checking correct forms, skip the incorrect ones.
                continue
            SR_indeces = sr2datum[form] #Find all the training data that use this SR
            UR_indeces = ur2datum[ur[datum_index]] #Find all the training data that use this UR
            predicted_SRprob = sum(le_probs[SR_indeces]) #Sum the SR probs (merges different HR's)
            predicted_URprob = sum(le_probs[UR_indeces]) #Sum the UR probs (merges different SR's and HR's)
            if predicted_URprob == 0:
                raise Exception("Rounding error! pr(UR)=0")
            else:
                conditional_prob = predicted_SRprob/predicted_URprob #Find the prob of this SR, given its UR
            if conditional_prob < .9: #If >90% of prob isn't given to the correct SR...
                learned = False
                break
        if learned:
            print("\t"+str(1+epoch)+" epochs to completion!")
            ep_file.write(language+"\t"+str(epoch)+"\n")
            break
    
    if not learned:
        ep_file.write(language+"\t-1\n") #-1 epochs for learning failure, convention borrowed from Josh
        
    return new_weights 
    
######LOOP THROUGH ALL LANGUAGES######
my_time = sub(":", ".", str(datetime.now()))
success_file = open(path.join("Output_Files", "successes_"+my_time+".csv"), "w") 
success_file.write("Language,Successful?\n")
input_files = [fn for fn in listdir(path.join("Input_Files", LANG_SET)) if ".csv" in fn]
test_langs = [sub("[^0-9]", "", l) for l in input_files]

if METHOD == "gd":
    ep_file = open(path.join("Output_Files", "EpochsToConvergence_"+my_time+".txt"), "w")
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
    tableaux_file = open(path.join("Input_Files", LANG_SET, input_files[lang_index]), "r")
    headers = tableaux_file.readline().rstrip().split(",")
    CON = headers[4:]
    
    #Get violation vectors:
    v = []
    probs = []
    sr = []
    ur = []
    hr = []
    input_lines = []
    kind_of_line = []
    for row in tableaux_file.readlines():
        input_lines.append(row.rstrip().split(","))
        ur_line = search("^([^,]+),*\n", row)
        sr_line = search("^,([^,]+),([^,]+),*", row)
        hr_line = search("^,,,([^,]+),(.+)", row)
        
        if ur_line:
            kind_of_line.append("ur")
            my_in = ur_line.group(1)
            continue
        elif sr_line:
            kind_of_line.append("sr")
            my_out = sr_line.group(1)
            my_prob = float(sr_line.group(2))
            continue
        elif hr_line:
            kind_of_line.append("hr")
            my_hid = hr_line.group(1)
            raw_viols = hr_line.group(2).rstrip().split(",")
            my_viols = [-1 * float(viol) for viol in raw_viols]  
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
    
    #Normalize probabilities in the training data:
    new_probs = [] 
    probs = np.array(probs)
    for datum_index, this_prob in enumerate(probs):
        new_prob = this_prob/(len(sr2datum[sr[datum_index]])*UR_num)
        new_probs.append(new_prob)
      
    #Initializing vectors for the grammar's weights, constraint violations, 
    #and candidate probabilities: 
    if RAND_WEIGHTS:
        w = list(np.random.uniform(low=0.0, high=10.0, size=len(v[0])))   #Init constraint weights = rand 1-10
        print ("Initial weights: ", w)
    else:  
        w = [INIT_WEIGHT for c in v[0]]  #Init constraint weights = INIT_WEIGHT
    
    #If you want to start the model off with a particular set of weights, 
    #do that here (needs to be in the right order!):
    #w = [30.698, 15.175, 0.000, 19.877, 2.076, 1.630, 6.142, 0.000, 1.281, 5.335, 1.331, 24.222, 18.229]

    v = np.array(v) #Constraint violations                  
    p = np.array(new_probs) #Probabilities in training data 
    
    #####LEARNING##### 
    if METHOD == "lbfgsb":
        print (" ...Learning...")
        if NEG_WEIGHTS:
            lower_bound = None
        else:
            lower_bound = 0.0    
        final_weights = minimize(objective_func, w, args=(v, p, sr), method="L-BFGS-B", bounds=[(lower_bound, None) for x in w])['x']
    elif METHOD == "gd":       
        final_weights = gradient_descent(w, v, p, sr, EPOCHS)
    elif METHOD == "cg":
        final_weights = minimize(objective_func, w, args=(v, p, sr), method="CG", options={'maxiter':15000})['x']
    else:
        raise Exception("Unkown method! (Must be 'lbfgsb', 'cg', or 'gd'.)")
    
    current_probs = get_predicted_probs(np.array(final_weights), v)
           
    #####OUTPUT##### 
    print ("...Saving output...")    
    #More succinct output file: 
    #Find highest probability parse (according to model) for each UR (i.e. tableau):
    ur2bestParse = {}
    ur2highestProb = {}
    ur2totalProbs = {}
    mapping2prob = {}
    for datum_index, form in enumerate(ur):
        if form in ur2highestProb.keys():
            if ur2highestProb[form] < current_probs[datum_index]:
                ur2highestProb[form] = current_probs[datum_index]
                ur2bestParse[form] = hr[datum_index]
            ur2totalProbs[form] += current_probs[datum_index]
        else:
            ur2highestProb[form] = current_probs[datum_index]
            ur2bestParse[form] = hr[datum_index]
            ur2totalProbs[form] = current_probs[datum_index]
            
        if (form, sr[datum_index]) in mapping2prob.keys():
            mapping2prob[(form, sr[datum_index])] += current_probs[datum_index]
        else:
            mapping2prob[(form, sr[datum_index])] = current_probs[datum_index]
            
    #Print those parses into a CSV:
    brief_output_file = open(path.join("Output_Files", language+"_BriefOutput_"+my_time+".csv"), "w")
    new_headers = ["UR", "HR", "p_TD", "p_normed", "p_absolute"]+headers[4:]
    brief_output_file.write(",".join(new_headers)+"\n,,,,,")
    for fw in final_weights:
        brief_output_file.write(str(fw)+",")
    brief_output_file.write("\n")
    datum_index = 0
    for old_line, line_type in zip(input_lines, kind_of_line):
        if line_type == "ur":
            UR = old_line[0]
        elif line_type == "sr":
            SR = old_line[1]
            TD_prob = old_line[2]
        elif line_type == "hr":
            HR = old_line[3]
            absProb = current_probs[datum_index]
            normedProb = absProb/ur2totalProbs[UR]
            if HR == ur2bestParse[UR]:
                new_line = [UR, HR, TD_prob, str(normedProb), str(absProb)]+old_line[4:]
                brief_output_file.write(",".join(new_line)+"\n")
            datum_index += 1  
        else:
            print(old_line)
            raise Exception("Unexpected line type!")
    brief_output_file.close()

    #Main output file:
    output_file = open(path.join("Output_Files", language+"_Output_"+my_time+".csv"), "w")
    new_headers = headers[:2]+["p(SR)_TD", "p(SR)_LE","p(HR)_normed", "p(HR)_absolute"]+headers[3:]
    output_file.write(",".join(new_headers)+"\n,,,,,,,")
    for fw in final_weights:
        output_file.write(str(fw)+",")
    output_file.write("\n")
    datum_index = 0
    for old_line, line_type in zip(input_lines, kind_of_line):
        old_line = sub(",+$", "", old_line)
        if line_type == "sr":
            new_line = old_line+[str(mapping2prob[(ur[datum_index],sr[datum_index])]/ur2totalProbs[ur[datum_index]])]
            output_file.write(",".join(new_line)+"\n")            
        elif line_type == "ur":
            output_file.write(",".join(old_line)+"\n") 
        elif line_type == "hr":
            new_line = old_line[:2]+["", "", str(current_probs[datum_index]/ur2totalProbs[ur[datum_index]]), str(current_probs[datum_index])]+old_line[3:]
            output_file.write(",".join(new_line)+"\n")
            datum_index += 1
        else:
            print(old_line)
            raise Exception("Unexpected line type!")        
    output_file.close()    

    #Success file (prints out each language and whether it was successfully learned):
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
        
#Close output files:
success_file.close()
if METHOD == "gd":
    ep_file.close()
