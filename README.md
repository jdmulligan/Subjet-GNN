# Subjet-GNN

This repository performs graph construction and trains graph neural networks to classify jets in high-energy physics.

## Setup software environment – on hiccupgpu
<details>
  <summary>Click for details</summary>
<br/> 

### Logon and allocate a node
  
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

## Setup software environment – on perlmutter
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

## Training the GNNs

The GNNs can be constructed and trained using the following:
```
cd Subjet-GNN
python analysis/steer_analysis.py -c <config> -i <input_file> -o <output_dir>
```
The `-i` path should point to a file `subjets_unshuffled.h5` containing a dataset produced by the [JFN](https://github.com/jdmulligan/JFN) repository. Locations of produced datasets on hiccup and perlmutter can be found [here](https://docs.google.com/spreadsheets/d/1DI_GWwZO8sYDB9FS-rFzitoDk3SjfHfgoKVVGzG1j90).

Once the graphs are constructed (by the `graph_constructor` module), they will be read from file on subsequent runs. 
- If you would like to force recreate them, you can add the argument `--regenerate-graphs`.
- If you would like to use graphs that were already constructed from the JFN processing script output (in the `subjets_unshuffled.h5` file), you can add the argument `--use_precomputed_graphs`.

You can also re-run the plotting script after training the models, if you like:
```
cd Subjet-GNN
python analysis/plot_results.py -c <config> -o <output_dir>
```
