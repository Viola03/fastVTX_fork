# fast_vertexing_variables

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

The above `ganga` scripts submit the `DaVinci` script `davinci_intermediates.py`.

This script contains the conditions to run the unbiased (all PID and stripping cuts removed) three-body reconstruction. The important line in this script is
```
"& ((mcMatch('B+')) | (mcMatch('B-')) | (mcMatch('B0')) | (mcMatch('B~0')) | (mcMatch('B_s0')) | (mcMatch('B_s~0'))  | (mcMatch('B_c+')) | (mcMatch('B_c-')))"
```
which avoids running combinatorial events (which would otherwise dominate the results and just be thrown away anyway), this is essentially the same as applying `BKGCAT>60` upfront.

These `ganga` jobs will output small `*.root` files in their respective `ganga` directories which can be `hadd` by hand, `scp`'ed over to `gpu01` and processed there. 

### Processing data files

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