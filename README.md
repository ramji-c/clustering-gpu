# clustering-gpu
Document Clustering using GPUs with Tensorflow

**Pre-requisites** 

A CUDA enabled GPU, cudaToolkit v8.0, cudaDNN v5.1 and tensorflow-gpu python package

**Usage Notes**

The input to this program is a numpy Ndarray (.npy file); text documents must be vectorized using tf-idf or other
such vectorizers externally before being input to this script. Input should be a **dense matrix.**
 
The output is a trained model of 'k' centroids which is compatible with scikit-learn, and hence be used as the 'init' param in 
http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html

**Training**

To perform Mini-Batch K-means clustering on the input, execute the script as below

_python MiniBatchKmeans.py <<input .npy file>> --model-dir <<path where model should be stored>> 
-k <<num clusters>> --n-iter <<# iterations>> --batch-size <<mini-batch size>>_