# FCDNet
<!-- CNN for Focal Cortical Dysplasia segmentation
```
|
└───configs  
│   │   config.ini - config for patch-based segmentation
│   
└───metadata  
│    │   metadata.npy   
│
└───models  
│    │   v2v.py     
│   
└───folder2  
    │   file021.txt  
    │   file022.txt  
``` -->

 - `metadata/*` - stores train-test splits for different experiments\datasets
 - `configs/*` - config 
 - `visualization_seg.ipynb` - visualize results of the trained models (from `best_val_preds` folder in corresponding logdir)

For example, to run full-brain segmentation using V2V-Net:
1) Set desired training parameters in `configs/config.ini`
2) Run `python train_seg.py`
3) Observe training process by looking at corresponding log file using `tensorboardX`
