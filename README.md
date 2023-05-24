# Casimir-programming

As PhD candidates of the bionanoscience department we are constantly analyising microscopy images and trying to obtain quantitative information from them. 
For this reason we develope the notebook ImageAnalysis.ipynb that implements some basic image analysis tools such as changing the brightness and contrast of images with linear and gamma filters, a function to transform raw images into binary images given an intensity treshold (masking) and then count the objects above or below the threshold, and a function to detect boundaries and calculate the curvature of objects in an image. 

This respository contains the aforementioned notebook. Additionally, the respository contains the files masking_mod.py, counting_module.py and curvature_module.py which conatin functions that are called in the main notebook ImageAnalysis.ipynb.
Finally, the images (.tif files) used for testing the filters can be found in the folder Images
