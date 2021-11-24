Consists of the script, “testSub_fmri.py” to compare the participant’s model RDM to fMRI brain data.


“testSub_fmri.py”
Run as: python testSub_fmri.py --rdms_dir '../Feature_Extract/rdms/92images_rdms'
Replace '../Feature_Extract/rdms/92images_rdms' by the path containing rdms folder of different CNNs

The brain data is limited by the measurement noise and the amount of data. Therefore, we do not expect that a model RDM reaches a correlation of 1 with the brain data RDMs. The noise ceiling is the expected RDM correlation achieved by the (unknown) ideal model, given the noise in the data. 

The EVC and IT correlations are obtained from the comparison of respective model RDMs to the corresponding fMRI RDMs. The significance values are obtained from a t-test over fMRI data of 15 participants. The score is obtained by averaging the two correlations.

