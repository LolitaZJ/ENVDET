Usage

Date: 2022.08.17

Ji Zhang

# ENVDET 

ENVDET is a pipline to achieve earthquake detection.  
ENVDET consists of ENVnet and DETnet.

`Zhang J. and Zhang J. 2022. Detect small earthquakes by waveform envelope using machine learning.`

### ENVnet and DETnet architectures
![ENVDET_NET](https://user-images.githubusercontent.com/41277021/216076038-0800f5f3-c4b6-4810-abd1-bc8989bead59.png)

### Analysis of signal-to-noise ratio and frequency band
![Analysis of signal-to-noise ratio and frequency band](https://user-images.githubusercontent.com/41277021/216076593-a2a969cd-55a8-4a2a-bd8f-bb6adb9e0150.png)

## ENVnet

ENVnet can get envelopes of waveforms while enhancingdata features and suppressing noise.  

###Usage

The envelopes of seismic data can also be used to `locate events` and `infer source focal mechanisms`, and so on. 

## DETnet

DETnet can detect events with the envelopes as input.

## 1. Set envirnoment 
> tensorflow >= 2.0  
> python >= 3.8  

## 2. Load data
> load <code>[STEAD Data](https://github.com/smousavi05/STEAD)</code> or use `https://github.com/smousavi05/STEAD` to get dataset.  

> mv STEAD data to `./data/` 
 
Of course! You can use your dataset.  
**Notes:** save dataset to hdf5 file and record the index to csv file like STEAD dataset.

## 3. Train ENVDET
### a) train ENVnet
> train   
`python ENVDET_ENV_MAIN.py --mode=train --data_dir='./your STEAD hdf5 file path/'--csv_dir='./your STEAD csv file path/'`


> test  
`python ENVDET_ENV_MAIN.py --mode=test --data_dir='./your STEAD hdf5 file path/'--csv_dir='./your STEAD csv file path/`

### b) train DETnet
> train 
`python ENVDET_DET_MAIN.py --mode=train --data_dir='./your STEAD hdf5 file path/'--csv_dir='./your STEAD csv file path/'`

> test  
`python ENVDET_DET_MAIN.py --mode=test --data_dir='./your STEAD hdf5 file path/'--csv_dir='./your STEAD csv file path/`

## 4. Prediction
> predict testing dataset
>  
`python ENVDET_MAIN.py --mode=test --data_dir='./your STEAD hdf5 file path/'--csv_dir='./your STEAD csv file path/' --save_result`

> Save the prediction figures to `./save_result/`

> predict new datset  
> 
`python ENVDET_MAIN.py --mode=predict --pred_dir='./your  hdf5 file path/'--pred_csv='./your  csv file path/ --pred_name=New_data --save_result`

> Save the prediction results to `./Res/`

## 5. Folder
> `./data`: Store the data and csv files used for training, testing, and prediction.

>>`stalta_pk_noise.npz` Store the picking of noise data.
   
> `./model`: Store the trained models.   

> `./model_configure`: Save the parameters of models.

> `./acc_loss_fig`: Save the training curves of loss and accuracy. 

> `./Res`: Save the prediction figures.

