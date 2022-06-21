# FCDNet
<!-- CNN for Focal Cortical Dysplasia segmentation
```

└───fcd_analysis.ipynb - allows to assess number of connected components in the label.  
└───tensors_preparation.ipynb - used to pre-calculate PyTorch tensors from original data if needed
|
└───configs  
│   │   patch_segmentation.yaml - config for patch-based segmentation
│   │   segmentation.yaml - config for patch-based segmentation
│   
└───metadata  
│    │   unet3d.py  
│    │   v2v.py 
│
└───models  
│    │   unet3d.py  
│    │   v2v.py     
│   
└───folder2  
    │   file021.txt  
    │   file022.txt  
``` -->

 -  `fcd_analysis.ipynb` - allows to assess number of connected components in the label.  
    We do not use samples with more than one connected component by default.
 - `tensors_preparation.ipynb` used to pre-calculate PyTorch tensors from original data if needed
 - `metadata/*` - stores train-test splits for different experiments\datasets
 - `configs/*` - config 
 - `run_experiment_*.sh` - scripts for starting experiment
 - `visualization_*.ipynb` - visualize results of the trained models (from `best_val_preds` folder in corresponding logdir)

For example, to run full-brain segmentation using UNet3D:
1) Prepare fcd-label components dictionary with `fcd_analysis.ipynb`
2) Prepare tensors for training by running `tensors_preparation.ipynb`
3) Set desired training parameters in `configs/segmentation.yaml`
4) Run `run_experiment_seg.sh`
5) Observe training process by looking at corresponding log file using `tensorboardX`
