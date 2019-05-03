import numpy as np
from sys import exit
from scipy.stats import entropy
from scipy.optimize import minimize
import mpmath
from re import sub, search
from os import listdir

#####USER SETTINGS#####  
RAND_WEIGHTS = False #Are intial weights random (or all 1?)
TD_DIR = "Tesar and Smolensky Langs" #Directory containing training data
PREFIX = "ts" #all language files should start with this, then have a numerical label, then ".csv".

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
            exit("Rounding error! (Z=0 for "+ur[datum_index]+")")
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
            exit("Rounding error (exact function)! (Z=0 for "+ur[datum_index]+")")
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
success_file = open("C:\\Users\\Brandon\\OneDrive\\Research\\Hidden Structure\\lbfgsb_successes_"+PREFIX+".csv", "w")
datumProb_file = open("C:\\Users\\Brandon\\OneDrive\\Research\\Hidden Structure\\datum_probs_"+PREFIX+".csv", "w")
test_langs = [sub("[^0-9]", "", fn) for fn in listdir(TD_DIR)]

for language in test_langs:  
    #####TRAINING DATA##### 
    #Needs to create three numpy arrays:
    #    >w: array of weights of length C
    #    >v: 2d array of violations, height=D and width=C   
    #    >p: array of training data probabilities of length D
    #
    #...And a dictionary called "sr2datum" that maps SR's to a list of
    #   the data (i.e. indeces in v and p) that they're associated with.

    print "Processing input file #"+str(language),
       
    #Get constraint names:
    tableaux_file = open(TD_DIR+"\\"+PREFIX+str(language)+".csv", "r")
    headers = tableaux_file.readline().rstrip().split(",")
    
    #Get violation vectors:
    v = []
    probs = []
    sr = []
    ur = []
    hr = []
    for row in tableaux_file.readlines():
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
            exit("Error in Training Data File! (line: "+row.rstrip()+")")
        
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
        new_prob = this_prob/(sum(probs[ur2datum[ur[datum_index]]])*UR_num)
        new_probs.append(new_prob)
      
    #Vectors that we need: 
    if RAND_WEIGHTS:
        w = list(np.random.uniform(low=0.0, high=10.0, size=len(v[0])))   #Init constraint weights = rand 1-10
        print "Initial weights: ", w
    else:  
        #w = np.array([0.0 for c in v[0]])   #Init constraint weights = 0
        w = [1.0 for c in v[0]]  #Init constraint weights = 10
    v = np.array(v)                   #Constraint violations
    p = np.array(new_probs)  
    
    #####LEARNING##### 
    print " ...Learning..."
    final_weights = minimize(objective_func, w, args=(v, p, sr), method="L-BFGS-B", bounds=[(0.0, 200) for x in w])['x']
           
    #####OUTPUT##### 
    current_probs = get_predicted_probs(np.array(final_weights), v)
    learned = True
    attested_data = []
    for datum_index, form in enumerate(sr):
        if probs[datum_index] != 1: 
            #We're only checking correct forms
            continue
        else:
            attested_data.append(form)
        SR_indeces = sr2datum[form] #Find all the training data that use this SR
        UR_indeces = ur2datum[ur[datum_index]] #Find all the training data that use this UR
        predicted_SRprob = sum(current_probs[SR_indeces]) #Sum the SR probs (merges different HR's)
        predicted_URprob = sum(current_probs[UR_indeces]) #Sum the UR probs (merges different SR's and HR's)
        if predicted_URprob == 0:
            exit("Rounding error! pr(UR)=0")
        else:
            conditional_prob = predicted_SRprob/predicted_URprob #Find the prob of this SR, given its UR
            #print hr[datum_index], ",", form, current_probs[datum_index], conditional_prob
        if conditional_prob < .5: #If the majority of prob isn't given to the correct SR...
            #print form, conditional_prob
            learned = False
            
    #Record the probability assigned to each (hidden) structure
    if learned:
        datumProb_file.write("Lang: "+str(language)+"\n")
        for attested_sr in attested_data:
            datumProb_file.write(","+attested_sr+"\n")
            crit_50 = False
            for hr_index in sr2datum[attested_sr]:
                hr_prob = current_probs[hr_index]
                ur_prob = sum(current_probs[ur2datum[ur[datum_index]]])
                my_cond_prob = hr_prob/ur_prob
                if my_cond_prob > .5:
                    crit_50 = True
                datumProb_file.write(",,"+hr[hr_index]+","+str(my_cond_prob)+"\n")
            if crit_50:
                datumProb_file.write(",Met stict criterion (>50% on a single structure)\n")
            else:
                datumProb_file.write(",Did not meet stict criterion (>50% on a single structure)\n")
                       
    #Record whether we met the criterion:
    if learned:
        for possible_bug in current_probs:
            if np.isnan(possible_bug):
                exit("Found a NAN!")
        print "Language "+str(language)+" was successfully learned."
        success_file.write("Language "+str(language)+",1\n")
    else:
        print "Language "+str(language)+" was NOT learned."
        success_file.write("Language "+str(language)+",0\n")
        
#Deal with output files:
success_file.close()
datumProb_file.close()