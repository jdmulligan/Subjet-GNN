# Graph neural networks for high-energy physics

This repository constructs and trains graph neural networks to classify jets in high-energy physics using the "Jet Flow Network" subjet basis ([arXiv:2305.08979](https://arxiv.org/abs/2305.08979)),
along with benchmark comparisons to alternate architectures including transformers and deep sets.

<div align="center">
<img src="https://github.com/jdmulligan/Subjet-GNN/assets/16219745/8dc896c6-1201-4d68-8cfa-932d1d79a2db" width="700" height="260">
</div>  

## Training the GNNs

The GNNs can be constructed and trained using the following:
```
cd Subjet-GNN
python analysis/steer_analysis.py -c <config> -i <input_file> -o <output_dir>
```
The `-i` path should point to a file `subjets_unshuffled.h5` containing a dataset produced by the [JFN](https://github.com/jdmulligan/JFN) repository. Locations of produced datasets can be found [here](https://docs.google.com/spreadsheets/d/1DI_GWwZO8sYDB9FS-rFzitoDk3SjfHfgoKVVGzG1j90).

Once the graphs are constructed (by the `graph_constructor` module), they will be read from file on subsequent runs. 
- If you would like to force recreate them, you can add the argument `--regenerate-graphs`.
- If you would like to use graphs that were already constructed from the JFN processing script output (in the `subjets_unshuffled.h5` file), you can add the argument `--use_precomputed_graphs`.

You can also re-run the plotting script after training the models, if you like:
```
cd Subjet-GNN
python analysis/plot_results.py -c <config> -o <output_dir>
```

## Training different models

Different architectures (transformers, deep sets) can be trained by specifying the model in the yaml config file.

To include an additional architecture, you should implement the following:
- The `model` folder contains a class for each model to handle initialization, data loading, and training: `init_model()`, `init_data()`, `train()`
- The `architecture` folder contains architecture definitions themselves: e.g. for PyTorch `init()`, `forward()`
- The `ml_analysis.py` module then will initialize and train the model, using the achitecture.

## Setup software environment – example: hiccup cluster
<details>
  <summary>Click for details</summary>
<br/> 

### Logon and allocate a node – example on hiccupgpu
  
Logon directly to hiccupgpu:
```
ssh <user>@hic.lbl.gov -p 1142
```

This is not yet integrated into the slurm queue on the hiccup system, so just beware that if someone else is using the system at the same time you will want to keep an extra eye on the memory consumption.

### Initialize environment
  
Now we need to initialize the environment: set the python version and create a virtual environment for python packages.
Since various ML packages require higher python versions than installed system-wide, we have set up an initialization script to take care of this. 
The first time you set up, you can do:
```
cd Subjet-GNN
./init_hiccup.sh --install
```
  
On subsequent times, you don't need to pass the `install` flag:
```
cd Subjet-GNN
./init_hiccup.sh
```

Now we are ready to run our scripts.


</details>

## Setup software environment – example: perlmutter cluster
<details>
  <summary>Click for details</summary>
<br/> 
  
### Logon and allocate a node
  
Logon to perlmutter:
```
ssh <user>@perlmutter-p1.nersc.gov
```

First, request an [interactive node](https://docs.nersc.gov/jobs/interactive/) from the slurm batch system:
   ```
   salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu --gpus 4 --account=alice_g
   ``` 
   which requests 4 GPUs on a node in the alice allocation. 
When you’re done with your session, just type `exit`.

### Initialize environment
  
We will only run the ML part of the pipeline on perlmutter. For now, you should copy your output file of generated jets/events:
```
scp -r /rstorage/<output_file> <user>@perlmutter-p1.nersc.gov:/pscratch/sd/<initial letter of user>/<user>/
```

Now we need to initialize the environment:
```
cd Subjet-GNN
source init_perlmutter.sh
```

Now we are ready to run our scripts.

   
</details>