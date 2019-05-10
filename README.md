# Protein Classification - SG Genomic Data Sicentist Challenge
_This was my first brush with machine learning, back in mid-2017. I had applied for a position at Second Genome and was sent
 this challenge to complete as part of their technical assesment.
I've since progressed but I'm leaving it up as I think it's a good example of my reasoning process and how I write code.
The code, figures and the powerpoint presentation I gave are all in this repo.
Also: I got the job! - JD_
 
 
 ## Challenge description
Welcome to the Second Genome - Genomic Data Scientist challenge! Please
read all instructions carefully.

### Introduction

The biotech giant DrugPro has a small but profitable portfolio of bacterial-
derived proteins to treat chokeophobia, the clinical condition described by a
debilitating fear that the Chicago Cubs won't return to the World Series this
year.  While proven effective prior to the All-Star break, with the recent but 
historic win streak of the Cleveland Indians, local clinicians have begun to
identify resistance in the effected population.

DrugPro is convinced that the microbiome strongly influences susceptibility to 
chokeophobia, and they have partnered with you to find other microbiome-derived
proteins that may combat this debilitating condition. Thanks to the heroic
efforts of your R&D division, you have identified 10,000 proteins that are
over-expressed in Cubs fans who are resistant to chokeophobia. As resident
Genomic Data Scientist, you are tasked with mining this list of proteins for
those that are most likely to confer resistance to their host.

To faciliate your efforts, DrugPro has provided you a training set.  This
training set contains 264 proteins that have been evaluated by DrugPro's best-in-
class clinical assay for treating chokeophobia.  Those proteins that demonstrate
therapeutic activity have been labeled with a '1'. Those that do not have been
labeled with a '0'. 

### Problem Description

Develop and train a classifier for classifying an unknwon protein as being
either therapeutic ('1') or non-therapeutic ('0') based on its protein sequence
and the training data supplied by DrugPro. Once trained, apply the classifier
to the set of 10,000 proteins identified by your R&D team and select 200
proteins that are most likely to exhibit therapeutic function for synthesis
and testing by DrugPro.


### Materials

The following materials are provided to you.

  > sg_putative_10000.faa : the set of 10,000 protein sequences identified by R&D

  > drugpro_training.csv : the training data set, formatted as comma-separated
    values with columns 'id','seq','label'

  > protein_nomination_value.pkl : a pickled Python dictionary that returns
    the "monetary value" for proteins in sg_putative_10000.faa. See
    "Evaluation" below. 


### Methods

The problem of associating function to an unknown protein sequence is well-
known in molecular biology.  Typically this is approached using sequence
identity -- if a protein of unknown function exhibits a high degree of similarity
to a protein of known function, it is often assumed that the two proteins share
the same function.  Two tools that allow you to measure sequence identity are

  > NCBI's blastp:
    https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE=Proteins

  > vsearch - a fast heuristic-based sequence identity search
    https://github.com/torognes/vsearch

There are many others. If you choose to leverage sequence identity in your
method, feel free to use whichever tool you deem most appropriate.

In addition to sequence identity, hidden-markov models are well known to detect
"remote hology", that is, a functional relationship between two proteins whose
sequence identity may be very low.  The most-often used tool for scanning 
protein sequences against a know set of HMMs is hmmer:

  > HMMER
    http://hmmer.org/

A typical source of HMMs with which to annotate proteins using HMMER is the
Pfam database of protein domains:

  > Pfam
    http://pfam.xfam.org/

To scan a ${QUERY} of protein sequences in fasta format against the databse of
HMMs in Pfam, you may issue a command like the following:

hmmscan --domE ${QUERY} --domtblout ${OUTPUT} --noali --cpu ${NUMCPU}
${PFAMA} ${QUERY} > /dev/null

In addition to ${QUERY}, the following arguments appear in this command:

  > ${THRESHOLD} : a significance threshold. 1e-5 is very significant.
    1e-3 less so. 1e0 will find all, even insignificant domains
  > ${OUTPUT} : the name of your output file. Note that Hmmer outputs in
    fixed width text format
  > ${NUMCPU} : the number of CPUs to employ
  > ${PFAMA} : the location of your HMM database

Note that if you download the Pfam database from the Pfam website, you may
have to build the database first using hmmpress. Running the hmmscan command
above will notify you if this is the case.

Finally, in addition to sequence identity and domain similarity, multiple
publications defend the use of machine learning models trainined on 
physicochemical features to classify proteins of unknown function from their
primary sequence.  There is no such thing as "the" set of physicochemcial
features for a particular protein.  The set is effectively infinite.  However,
there are several resources for calculating a fairly comprehensive set of
physicochemcial features from a set of sequences.  They are:

  > ProFEAT (webserver)
    http://bidd2.nus.edu.sg/cgi-bin/profeat2016/main.cgi

  > ProtrWeb (R)
    http://protrweb.scbdd.com/

  > Pydpi (Python)
    https://pypi.python.org/pypi/pydpi/1.0

In addition to sequence identity, protein domains, and physicochemical features,
you are encouraged to engineer whatever features you deem necissary from the
primary sequences to accomplish the classification task, but these should
provide a sufficient start.


### Evaluation

DrugPro has agreed to synthesize and test 200 proteins of your choice in their
assay.  Because DrugPro is concerned about whether they can obtain legal rights
to well-known therapies for chokeophobia, you will be remunerated for proteins
that score positively in their assay according to how similar they are to
existing chokeophobia therapeutics in their training set:

    * >= 78% sequence identity: $1M per protein
    * < 78% but >= 45% sequence identity: $2M per protein
    * < 45% sequence identity: $4M per protein

Normally it requires several months to synthesize and test a protein, so we
will be accelerating that process with the following validation file:

  * protein_nomination_value.pkl

Once you are done training your classifier and applying it to the putative
set of 10000 proteins, use this pickled dictionary to report the monetary
value returned by your Top 200 nominations.  Obviously you could pretty
easily use this dictionary to just go and search for high-valued proteins
directly in the set of 10000, but that wouldn't make for a very good presentation.


### Mechanics of Evaluation

You are free to work on this problem up until the time of your presentation.
Prior to your presentation however, the hiring team must have your slides
and code in hand. When you present your work, please remember to discuss the
following:

  * a brief description of the problem (it helps to hear your version of it)
  
  * a brief description of the data
  
  * your choice of features engineered from the protein sequences
  
  * your feature selection methodolgy, if any
  
  * your choice of model or models
  
  * your validation method(s)
  
  * the quality of your model ($$$)

Bear in mind this exercise is not solely about building a great model.  The
hiring team will be evaluating the tractability of your code and your ability
to communicate your process and results.

Good luck and know that we are very excited to hear you present your work!
Please send us an email if you have any questions or run into any technical
difficulties.

  - Second Genome


