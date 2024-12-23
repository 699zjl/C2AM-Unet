<<<<<<< HEAD
# C2AM-Unet
=======

# C2AM-UNet
This repository is the official PyTorch code for the paper ['C2AM-Unet :Coordinate and Channel Attention Mixing Flexible Architecture for Retinal Vessel Segmentation).


<div align="center">
  <img src="Fig1.png" width="100%">
</div>

 
## Prerequisites
 

 
Download our repo:
```
git clone https://github.com/699zjl/C2AM-Unet.git
cd C2AM-Unet
```
Install packages from requirements.txt
```
pip install -r requirements.txt
```
 
## Datasets processing
Choose a path to create a folder with the dataset name and download datasets [DRIVE](https://www.dropbox.com/sh/z4hbbzqai0ilqht/AAARqnQhjq3wQcSVFNR__6xNa?dl=0),[CHASEDB1](https://blogs.kingston.ac.uk/retinal/chasedb1/),[STARE](https://cecas.clemson.edu/~ahoover/stare/probing/index.html),[CHUAC](https://figshare.com/s/4d24cf3d14bc901a94bf), and [DCA1](http://personal.cimat.mx:8181/~ivan.cruz/DB_Angiograms.html). Type this in terminal to run the data_process.py file
 
```
python data_process.py -dp DATASET_PATH -dn DATASET_NAME
```
 
## Training
Type this in terminal to run the train.py file
 
```
python train.py -dp DATASET_PATH
```
## Test
Type this in terminal to run the test.py file
 
```
python test.py -dp DATASET_PATH -wp WEIGHT_FILE_PATH
```
We have prepared the pre-trained models for both datasets in the folder 'pretrained_weights'. To replicate the results in the paper, directly run the following commands
```
python test.py -dp DATASET_PATH -wp pretrained_weights/DATASET_NAME
```


 
## License
 
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
 
>>>>>>> b802479 (Initial commit)
