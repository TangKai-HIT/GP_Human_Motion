# Studing GPLVM-like models for human motion analysis
**originated from Jack Wang's source codes of GPDM and Neil Lawrence's gplvm toolbox. Some modification to the code was made.**
## Jack Wang's Notations
This is a version of the GPDM code that you are free to use for academic purposes.  It is obviously not 'production code', and I do not guarantee its correctness or efficiency.  

See example.m for the code to learn a single walker model and generate samples from the learned model.  I only included a single .amc file from the CMU mocap data base, if you are interested in more mocap data, go to mocap.cs.cmu.edu.  
More generally, if you want to use your own data, pass the data as row-vectors in the Y matrix to gpdmfitFull. 

The code in src/gplvm is curtesy of Neil Lawrence, and src/netlab is simply a copy of the netlab library you can get at http://www.ncrg.aston.ac.uk/netlab/ . 

Cheers,
Jack Wang

## My Notations
Add MOCAP(modified version), NDLUTIL toolbox(By Neil Lawrence) to src, enabling visualization(animation) of asf/amc files.   
Add some modified functions to /modified.  
Add some useful functions to /myutil:  
`remakeAmcAnimation.m`: make amc & gif files for newly generated motions for predicted motion data matrix.(give the mean of the data if it's been centered)   

### Experiments  
1.`Experiment1`: compare PCA, KPCA, GPLVM, GDPM results: using MAP + SCG to train GPLVM & GPDM. KPCA is not able yet  
2.`Experiment2`: compare different learning methods: MAP + SCG/BFGS,  B-GPDM + SCG/BFGS (strange difference was seen when use BFGS to train MAP, results of which turned out to be different from that of SCG)   
3.`Experiment3`: try 2-stage-MAP(hmc-EM) method to train GPDM model    
4.`Experiment4`: Hold out test of multi-squence 1-order GPDM for different learning methods: MAP+BFGS, B-GPDM+BFGS  

Kai Tang
