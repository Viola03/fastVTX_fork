# fast_vertexing_variables

# Inference

## Installation

To install the package
```
pip install --no-dependencies -e inference/src/.
```

## Updating electron smeaing in RapidSim


## Running RapidSim
```
paramsStable : M P PT PX PY PZ origX origY origZ
paramsDecaying : M P PT PX PY PZ vtxX vtxY vtxZ origX origY origZ
```

## Query tool to add vertex quality branches


## Examples


# Training

## Installation

To install the package
```
pip install --no-dependencies -e src/.
```

## Running

### Getting data files

In `/davinci` run:
```
ganga gangaSubmit.py
```
to submit dedicated MC jobs or 
```
ganga gangaSubmit_mix.py
```
to submit a mix. The file `targets.pkl` (created by `search_simulationCondition.py`) contains a list of addresses for various MC from the bookkeeping.

The above `ganga` scripts submit the `DaVinci` script `davinci_intermediates.py`. UPDATE, script is now `davinci_general_mcmatch_intermediates.py`

This script contains the conditions to run the unbiased (all PID and stripping cuts removed) three-body reconstruction. The important line in this script is
```
"& ((mcMatch('B+')) | (mcMatch('B-')) | (mcMatch('B0')) | (mcMatch('B~0')) | (mcMatch('B_s0')) | (mcMatch('B_s~0'))  | (mcMatch('B_c+')) | (mcMatch('B_c-')))"
```
which avoids running combinatorial events (which would otherwise dominate the results and just be thrown away anyway), this is essentially the same as applying `BKGCAT>60` upfront.

These `ganga` jobs will output small `*.root` files in their respective `ganga` directories which can be `hadd` by hand, `scp`'ed over to `gpu01` and processed there. 

### Processing data files

UPDATE, output from `davinci_general_mcmatch_intermediates.py` can be manipulated with `davinci/collapse_individual_tuples.py`
UPDATE UPDATE, output from `davinci_general_mcmatch_intermediates.py` is now kept on Dirac, output is manipulated with `davinci/gangaSubmit_collapse.py`, and then hadd with `davinci/mergeAccessURLs_TChain.py`, (run `gangaSubmit_collapse.py` again to collect `OutputDataAccessURLs_XXXX.pkl` using the code at the top of the file). The TChain combines ROOT files using their PFNs, some can be at sites which cause slow loading and a blacklist might be required. `mergeAccessURLs_TChain.py` requires `source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_102b_LHCB_Core x86_64-centos9-gcc11-opt`.

In `/davinci` run:
```
python cut_hierarchy.py
```
the options inside the file will need to be changed to point to the correct files. This script removes unnecessary branches and repeats a `BKGCAT>60` cut.

To compute conditional variables run:
```
python scripts/variables_root.py
```
again the options inside the files will need to be adapted.

https://twiki.cern.ch/twiki/bin/viewauth/LHCb/FAQ/GangaLHCbFAQ#How_do_I_automatically_reduce_my

### Training the vertex smearing network

On `gpu01` run:
```
python scripts/primary_vertex_generator_keep_DIRA.py
```
this will save a set of network weights and some transformers in `/networks`, these will be loaded later.

### Producing Rapidsim samples

On `gpu01` you can source Rapidsim with:
```
cd
source get_rapidsim.sh
```
For example Rapidsim configuration files exist in `/rapidsim`.

Once a sample is generated you can compute conditional variables with:
```
python scripts/variables_rapidsim.py
```
and 
```
python scripts/variables_rapidsim_PART_RECO.py
```
options inside these files will need to be edited to ensure you are pointing to the correct files. It is important the correct particles are labelled as `B_plus`, `e_plus` and `e_minus`. These scripts load up the weights and transformers used to run the vertex-smearing network (the architectures listed in the initialisation of the network must match). 

### Training the vertex quality network

On `gpu01` run:
```
python GAN_distances.py
```
there are important parameters to check in the script, such as setting `rd.latent`. 


### Testing the vertex quality network

On `gpu01` run:
```
python scripts/test_GAN_distances.py
```
the architectures listed in the initialisation of the network must match those used in the training. 

To debug or understand the workings behind the scenes you can use
```
python scripts/plot_conditional_variables.py
```
(was `play.py`). 


## Testing the network in realistic scenarios

### $B^+\to K^+e^+e^-$


### $B^+\to D^0(\to K^+\pi^-) K^+$

[Paper](https://arxiv.org/pdf/2012.09903)
[ANA](https://cds.cern.ch/record/2714688/files/LHCb-ANA-2020-024.pdf)

## Notes

error matrix per track

brem fixed maybe 

slides for thursday

vertex info out of network 

hannae K*tautau (K*uu)

stripping and BDT cuts how they affect momenta distributions 

fix rapid sim P an PT distbutions 

isolation?

mass constrained varaibles = track error matrix 

add isolation, add vertex 

- Kee isolation variables as a function of all stuff i condition on 



best so far

# test_tag = '20th_long_2000_lower_LR'

training_data_loader.reweight_for_training("fully_reco", weight_value=100., plot_variable='B_plus_M')

latent = 10
# rd.D_architecture=[int(512*1.5),int(1024*1.5),int(1024*1.5),int(512*1.5)]
# rd.G_architecture=[int(512*1.5),int(1024*1.5),int(1024*1.5),int(512*1.5)]
rd.beta = 2000.
rd.batch_size = 256
self.optimizer = Adam(learning_rate=0.00001)
rd.include_dropout = True
rd.use_beta_schedule = False
