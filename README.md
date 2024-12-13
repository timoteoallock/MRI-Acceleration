# MRI-Acceleration

This repository contains the work done during my internship at Institut Fresnel in Marseille in 2023, where I worked on the acceleration of MRI acquisition times via Deep-Learning tecniques using fastmri library and Pytorch Lightning. 
The novelty of my work consists of the implementation of a new framework that supports 2 channel input images (i.e. complex k-space information) that to the best of our knowledge has never been adopted in this context. \
The repository contains: 
1. Tutorial on how to train the 1 input channel network 
2. Tutorial on how to train the 2 input channel network
3. Classes that were modified to make 2 input network training possible (data loading, loss calculations, image and metrics logging to Tensorboard) while still supporting 1 input network, thus creating a unifying framework for future MRI acceleration research.
   
   
We obtained an approximately 20% increase in key metrics such as PSNR, MSE and SSIM compared to baseline non deep learning models. 
