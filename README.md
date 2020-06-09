# Art Inspired Fashion - High Resolution
### Master Thesis - Started: 18.07.2018

## Instructions

- Install required dependencies from requirements.txt

### Training data
- Make sure images are located in the data folder folder
- An assoicated data loader should be provided in utils/dataloader.py

## Training model

- Model can trained using train.py and the options available for each model are stored in the options folder.
- Example of how to train the model can be found in the run_model.sh file. Here, various models with their default settings can be run.

## Overview of directory structure
- See below for insturctions on the directory structure
```
│   ---Analyis of results and visualisation of datasets will be found here---  
├─analysis  
│   ---DEPOSIT IMAGES HERE---  
├─data
│   │
│   │   ---Where training data will be stored---
│   └─redbubble (When downloaded)
│     
│   ---Directory for models---  
├─models  
│   │   
│   │  
│   └─directory for each model
│     │
│     │
│     └─code for that model
│       │
│       │
│       └──Logs for that model 
│
│   ---The options available for command line arguments      
├─options  
│   ---General helper functions---
├─utils  
```

Each model is contained with in its own folder within the model directory. When training begins for that model, a log is created with that models directory to log sample images, stats and checkpoints related to the model. Below is an overview of what will be created:
```
       └─model
          │
          │
          └─logs
            │
            │   ---Dataset used---
            └─name of dataset
              │
              │   ---Data and time of run---
              └─yyyy.mm.dd-hh.mm.ss
                │
                │   ---Parameters used for run---
                ├─model_params.json
                │
                │   ---Saved models from run---
                ├─model
                │
                │   ---Losses and other tensorboard stats related to run---
                ├─results
                │
                │   ---Image samples from run---
                └─samples
``` 
 

## Experiments carried out so far 06/12/18:
       * Compared DiscoGAN, CycleGAN and Pix2Pix at resolutions 64px, 128px, 256px, 512,px (in Thesis Design)
       * Noticed (G_A(G_B(A))) is very accurate at high resolutions. Higher resolution better the recreation.
       * Added an L1 loss between (G_A(G_B(A))) and G_A(B) (will cal rec-fake loss). Seems to help convergence on dress shape.
       * Added content loss from Neural Style Transfer instead of L1 loss when comparing real_A and fake_A. Dress shape is generated much quicker, cuts a dress shape within the painting.
       * Added content loss instead of L1 for cycle loss, seems to add additional limbs to the model
       * Added a content loss instead of L1 loss for rec-fake loss. Seems to  converge on dress shape quicker.
       * Added WGAN, seems to be able to identify the location of content within a painting but can't approximate it in any detail. Quite noisy when generating dresses
       * Made painting as small as the dress to see if the GAN has an easier time mapping the image to the dress. Early results indicate it does. The colors generated on dress are much more accurate and occasionally the content is a much clearer.
       * Masked the model from the dress and combined it with the smaller paintings. Mode collapse occurs very frequently but the GAN is able to create actual dresses from the training set but it will generate the same dress for all paintings. The painting generation on the other hand has less success. It captures very simples paintings like a phrase.
