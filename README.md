[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/maikia/ulfp-paper/master?filepath=index.ipynb)

Code for paper "Modeling unitary fields and the single-neuron contribution to local field potentials in the hippocampus" by Telenczuk Maria & Bartosz, Alain Destexhe.

# Licence
Documentation, examples and generated figures are licensed under the [Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/). The source code of the library is licensed under the [MIT license](https://opensource.org/licenses/mit-license.php). 

Authors: Maria Teleńczuk

# How to run the uLFP model

## 0. Pre-install
You need to have Neuron simulator for Python installed and NeuronEAP downloaded from https://github.com/maikia/neuroneap either to the same directory or you need to set the Python path

## 1. Translate your morphologies
You should choose the morphologies for your postsynaptic population (I used neuromorpho.org online database) of neurons from the area you are interested in; store all of the morphologies of the region/type in one folder in .swc format. To translate them so that the dendrites are vertical you should run [rotate_neurons.py](rotate_neurons.py.ipynb).
Choose your parameters appropriately. 

## 2. Choose parameters for your model
and save them in .json file.
You can find the template of how to do that and how the parameters should look like in [set_general_params.py](set_general_params.py.ipynb)
This will create an empty folder to store the results and the .json params file with the parameters

## 3. Choose parameters for each cell


Run [set_cell_params.py](set_cell_params.py.ipynb), this will create a json file for each cell, it will define the morphology used for each cell and where it is placed in space

## 4. Simulate cells
[simulate_cells.py](simulate_cells.py.ipynb)
This will call each cell separately (ie if you have a possibility you can run each of them on separate processor).
This is the longest step and the duration depends on the parameters you set (ie the number of postsynaptic cells, size of the grid, length of the simulation, etc)

## 5. Combine the results
There is local field potential calculated for each neuron separately. We can sum it linearly. To do that run
[sum_cell_results.py](sum_cell_results.py.ipynb) (don't forget to set the correct params')

## 6. plot the results
- here is an example of plotting results for a single cell: (activity at its synapses) [plot_cell_results.py](plot_cell_results.py.ipynb)
    
- here are few functions which plot the total field potential: [plot_summed_results.py](plot_summed_results.py.ipynb)
Many parameters are saved (hence you need a lot of memory space for larger simulations) but you can go wild with what and how you want to plot it




```python

```
