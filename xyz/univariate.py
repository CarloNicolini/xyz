import numpy as np
from scipy.spatial.distance import pdist

def entropy_linear(A: np.ndarray) -> float:
	"""
	Linear Gaussian Estimation of the Shannon Entropy
	Computes the shannon entropy of a multivariate dataset A
	A is N*M multivariate data (N observations, M variables)
	"""
	C = np.cov(A.T)

	# Entropy for the multivariate Gaussian case:
	N,M = A.shape
	# e.g., Barnett PRL 2009
	e = 0.5*np.log(np.linalg.det(C))+0.5*N*np.log(2*np.pi*np.exp(1))
	return e


def entropy_kernel(Y: np.ndarray,r: float, norm: str="c") -> float:
	"""
	Computes the entropy of the M-dimensional variable Y (matrix N*M)
	uses step kernel and maximum (Chebyshev) distance
	Note: uses "SampEn -like" entropy approach (exclude self-matches and compute -log of average probability)

	function Hy=its_Eker(Y,r,norma)

		if ~exist('norma','var') 
		    norma='c'; %default chebishev
		end

		% Y=[(1:10)' (11:20)'];% Y=(1:10)';
		N=size(Y,1);

		X=Y';
		p=nan*ones(N,1);
		for n=1:N
		   Xtmp=X; Xtmp(:,n)=[]; % exclude self-matches
		   if norma=='c'
		       dist = max(abs( repmat(Y(n,:)',1,N-1) - Xtmp ),[],1); % maximum norm
		   else % euclidian norm - for dimension 1 is equivalent
		       dist=nan*ones(1,size(Xtmp,2));
		       for q=1:size(Xtmp,2)
		           dist(q)=norm(Y(n,:)'-Xtmp(:,q),2);
		       end
		   end
		    
		   D = (dist < r);

		   p(n) = sum(D)/(N-1);
		end

		Hy = - log( mean(p) ); % "SampEn -like" version of entropy estimate 

		end
	"""
	N,M = Y.shape
	X = Y.T
	np.repeat()
	p = np.nan * np.ones(size=[N,1])
	for n in range(N):
		Xtmp = np.delete(X, indices=[n], axis=1)  # exclude self-matches
		if norm == 'c':
			dist = np.max(
				np.abs(
					np.tile(Y[n,:].T,repeats=[1, N-1]) - Xtmp
				),
				axis=2
			)
		D = dist < r
		p[n] = D.sum() / (N-1)
	Hy =  -np.log(np.mean(p))  # SampEn -like" version of entropy estimate
	return Hy