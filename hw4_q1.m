delta = 1e-5;
K=4;
imdata = imread('fig2.jpg');
imdata = im2double(imdata);
[R,C,RGB] = size(imdata);
N=R*C;
features = zeros(5,N);
for i=1:R
    for j=1:C
        features(1,(i-1)*C+j)=i/R;
        features(2,(i-1)*C+j)=j/C;
        features(3:5,(i-1)*C+j)=imdata(i,j,:);
    end
end
labels = zeros(R,C);

% Initialize the GMM to randomly selected samples
alpha = ones(1,K)/K;
shuffledIndices = randperm(N);
mu = features(:,shuffledIndices(1:K));
[~,assignedCentroidLabels] = min(pdist2(mu',features'),[],1);
for i = 1:K % use sample covariances of initial assignments as initial covariance estimates
    Sigma(:,:,i) = cov(features(:,find(assignedCentroidLabels==i))') + 1e-10*eye(5);
end

 
 Converged = 0; % Not converged at the beginning
for t=1:100
     for i = 1:K
         temp(i,:) = repmat(alpha(i),1,N).*evalGaussian(features,mu(:,i),Sigma(:,:,i));
     end
    plgivenx = temp./sum(temp,1);
    alphaNew = mean(plgivenx,2);
    w = plgivenx./repmat(sum(plgivenx,2),1,N);
    muNew = features*w';
    for l = 1:K
        v = features-repmat(muNew(:,l),1,N);
        u = repmat(w(l,:),5,1).*v;
        SigmaNew(:,:,l) = u*v' + 1e-10*eye(5); % adding a small regularization term
    end
    Dalpha = sum(abs(alphaNew-alpha'));
    Dmu = sum(sum(abs(muNew-mu)));
    DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
    Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
    alpha = alphaNew; mu = muNew; Sigma = SigmaNew; 
    %displayProgress(t,x,alpha,mu,Sigma);
end

for i=1:R
    for j=1:C
        labels(i,j)=evalGMM(features(:,(i-1)*C+j),alpha, mu, Sigma);
    end
end

figure(2),
s=pcolor(1:C,R-(1:R)+1, labels);
set(s, 'EdgeColor', 'none');
title("GMM Cluster Result For Figure 2")

function k = evalGMM(x,alpha,mu,Sigma)
gmm = -inf;
k=1;
for m = 1:length(alpha) % evaluate the GMM on the grid
    p=alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
    if gmm < p
        gmm=p;
        k=m;
    end
end
end


function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end