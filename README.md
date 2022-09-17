# Caps-cGAN:Capsule Conditional Generative Adversarial Network for Speckle Noise Reduction in Retinal OCT Images
Project directory structure
Data is stored in two folders under Datasets:  
-/OCT_Super  
------/train/  
--------/img  
----------f1  
----------f2  
----------f3  
------/mask  
----------f1  
----------f2  
----------f3  
  
--/valid/  
--------/img  
----------f1  
----------f2  
----------f3  
------/mask  
----------f1  
----------f2  
----------f3  
  
  
-/OCT_Data  
-----/train/
-------/img
---------f1
---------f2
---------f3
---------f4
-----/mask
---------f1
---------f2
---------f3
---------f4

--/valid/  
-------/img  
---------f1  
---------f2  
---------f3  
---------f4  
-----/mask  
---------f1  
---------f2  
---------f3  
---------f4  
  
Among them, f1 and f2 respectively store half of the labeled data, f3 and f4 are unlabeled data,   
f3 participates in training, and f4 only participates in testing.  
The default code is that the images in img and mask are one-to-one correspondence, because mask stores labels, and f3 and f4 are unlabeled,  
so directly put the images in img into f3 and f4 corresponding to the mask to avoid reporting bugs.   
f3 and f4 in the mask are useless, just to make ensure not reported errors when reading data.  
  
In the patients_to_slices function of the train file:  
ref_dict = {"f1": 512, "f2": 512,
                    "f3": 768,"f4": 640}  
Respectively indicate how much data there is in each corresponding folder.   
This can be replaced by your own data amount according to your actual data situation.  
