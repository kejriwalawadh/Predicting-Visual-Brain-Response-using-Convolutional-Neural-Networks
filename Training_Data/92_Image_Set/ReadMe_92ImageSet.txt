This is 92 ImageSet training data that contains the image files and fMRI brain data.


“92images.mat” contains a MATLAB structure with two fields:

-	visual_stimuli.names contains the images names in order.
-	visual_stimuli.pixels contains the pixel values of the corresponding images



“target_fmri.mat” is the fMRI brain data. On MATLAB prompt, load the file. 
>> load(“target_fmri.mat”)

This loads two fields to the MATLAB workspace:

“EVC_RDMs”
A 3-dimensional matrix of dissimilarities (1-Pearson corr) created from fMRI responses in Early Visual Cortex (EVC) with dimensions:
Subjects (15) * Images (92) * Images (92)

The last 2 dimensions form representational dissimilarity matrices, symmetric across the diagonal, with the diagonal zero(0).

“IT_RDMs”
A 3-dimensional matrix of dissimilarities (1-Pearson corr) created from fMRI responses in Inferior Temporal (IT) cortex with dimensions:
Subjects (15) * Images (92) * Images (92)

The last 2 dimensions form representational dissimilarity matrices, symmetric across the diagonal, with the diagonal zero(0).




For details about experiment design and data collection, please refer to reference [1].


References:
[1] Cichy, R.M., Pantazis , D., & Oliva, A. (2014). Resolving human object recognition in space and time.Nature Neuroscience, 17(3), 455-464.

[2] Cichy, R.M., Pantazis , D., & Oliva, A. (2016). Similarity-based fusion of MEG and fMRI reveals spatio-temporal dynamics in human cortex during visual object recognition. Cerebral Cortex, 26 (8): 3563-3579.

[3] Mohsenzadeh, Y., Mullin, C., Lahner, B., Cichy, R.M., & Oliva, A. (2018). Reliability and generalizability of similarity-based fusion of fMRI and MEG data in the ventral and dorsal visual streams. Vision 2019.




