import matplotlib.pyplot as plt
import scipy.io

def main():
    
    mat = scipy.io.loadmat('submit_fmri.mat')
    mat = mat['EVC_RDMs']
    plt.figure()
    plt.imshow(mat)
    plt.colorbar()
    plt.savefig('RDM.png')
    

if __name__ == '__main__':
    main()
    