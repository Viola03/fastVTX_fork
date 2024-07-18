# fast_vertexing_variables

## Installation

To install the package
```
pip install --no-dependencies -e src/.
```

## Running

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