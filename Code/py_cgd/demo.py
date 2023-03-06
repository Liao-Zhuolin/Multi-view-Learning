import numpy as np
import torch
from pro import load_data, NormalizeFea, bs_convert2sim_knn, spectral, bestMap, evaluate#, PridictLabel, ClusteringMeasure
from CGD import CGD
import warnings
warnings.filterwarnings('ignore')

def L2_distance(Z):
    Z = Z.astype('float64')
    num = np.size(Z,0)
    AA = np.sum(Z**2,axis=1)
    AB = np.matmul(Z,Z.T)
    A = np.tile(AA,(num,1))
    B = A.T
    d = A + B -2*AB
    return d

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
	torch.cuda.manual_seed(42)

dataFile = './MVL_data/MSRCV1.mat'
X, Y = load_data(dataFile)
kmeansK = len(np.unique(Y)) #clusters
ViewN = len(X)  #views
label = Y
TotalSampleNo = len(Y)
Dist = []  #distance adj
for vIndex in range(0, ViewN):
    TempvData = X[vIndex]#.todense().getA()
    NorTempvData = NormalizeFea(TempvData)
    tempDM = L2_distance(NorTempvData)
    #tempN, tempD = len(TempvData), len(TempvData[0])
    #tempDM = np.zeros((tempN,tempN))
    #for tempi in range(0, tempN):
    #    for tempj in range(0, tempN):
    #        tempDM[tempi,tempj] = np.linalg.norm(NorTempvData[tempi, :] - NorTempvData[tempj,:])**2
    Dist.append(tempDM)

knn = 16  # this parameter can be adjusted to obtain better results
sigma = 0.5
Sim = []
for ii in range(0, len(Dist)):
    Sim.append(bs_convert2sim_knn(Dist[ii], knn, sigma))

para_mu = 0.3
para_max_iter_diffusion = 10
para_max_iter_alternating = 10
para_thres = 1e-3
I = np.eye(len(Sim[0]))
para_beta = np.ones((len(Sim), 1))/len(Sim)
A, out_beta = CGD(Sim, I, para_mu, para_max_iter_diffusion, para_max_iter_alternating, para_thres, para_beta)

predY = spectral(A, kmeansK)
gnd_Y = bestMap(predY, Y.T[0])
AC = np.sum(gnd_Y == predY)/gnd_Y.shape[0]
nmi, ari, f, Precision, Recall, Purity= evaluate(gnd_Y, predY)
#out = PridictLabel(A, label.T, kmeansK)
#result, Con = ClusteringMeasure(label, out.T) #[8: ACC MIhat Purity ARI F-scoresultre Precision Recall Contingency]
print('Result:NMI:%1.4f || ACC:%1.4f || ARI:%1.4f || F-score:%1.4f || Precision:%1.4f \
|| Recall:%1.4f || Purity:%1.4f'%(nmi, AC, ari, f, Precision, Recall, Purity))
