# Editted fork of Alex Marshall's fast_vertexing_variables

Adapted for 'ResearchProject', which pulls various tools and model architectures from 'src'. Main edits include:

- updated to TensorFlow 2.18 for training on GPU

src/fast_vertexing_quality elements:
- added new plotting functions to output interim gen vs true samples, interim ROC curves, losses, KS distances, etc.
- minor adjustments to tools

Previous README:

# Inference

```
# grab code (includes one pre-trained model)
git clone ssh://git@gitlab.cern.ch:7999/amarshal/fast_vertexing_variables.git

# create and active a clean conda environment
conda create -n fast_vtx python=3.9
conda activate fast_vtx

# just confirm the right pip and right python and being pointed to
which pip
which python

# install all required libraries for inference 

pip install --no-dependencies -e inference/src/.
pip install --no-dependencies -e src/.

pip install numpy==1.26.4
pip install uproot==5.3.7
pip install uproot3==3.14.4
pip install matplotlib
pip install pandas
pip install vector==0.8.0
pip install onnxruntime
pip install scikit-learn
pip install str2bool
pip install particle
pip install hep_ml
pip install tensorflow
pip install mplhep

pip install tensorflow-addons

```

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

