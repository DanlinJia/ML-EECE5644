%function EMforGMM(N)
% Generates N samples from a specified GMM,
N=10;
K=10;
close all,
delta = 1e-5; % tolerance for EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates

% Generate samples from a 3-component GMM
alpha_true = [0.35,0.2,0.15,0.3];
mu_true = [-12 0 10 2;0 -14 0 -2];
Sigma_true(:,:,1) = [3 1;1 20];
Sigma_true(:,:,2) = [7 1;1 2];
Sigma_true(:,:,3) = [4 1;1 16];
Sigma_true(:,:,4) = [6 1;1 10];
x = randGMM(N,alpha_true,mu_true,Sigma_true);
[d,M] = size(mu_true); % determine dimensionality of samples and number of GMM components
bic=zeros(K,6);
for M=1:6 
    for kford=1:K
        x_test=getTestData(x,kford);
        x_train = getTrainData(x,kford);
        % Initialize the GMM to randomly selected samples
        alpha = ones(1,M)/M;
        shuffledIndices = randperm(N-N/K);
        mu = x_train(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
        [~,assignedCentroidLabels] = min(pdist2(mu',x_train'),[],1); % assign each sample to the nearest mean
        for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
            Sigma(:,:,m) = cov(x_train(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
        end
        t = 0; %displayProgress(t,x,alpha,mu,Sigma);
        Converged = 0; % Not converged at the beginning
        %while ~Converged
        for i=1:100
            for l = 1:M
                temp(l,:) = repmat(alpha(l),1,N-N/K).*evalGaussian(x_train,mu(:,l),Sigma(:,:,l));
            end
            plgivenx = temp./sum(temp,1);
            alphaNew = mean(plgivenx,2);
            w = plgivenx./repmat(sum(plgivenx,2),1,N-N/K);
            muNew = x_train*w';
            for l = 1:M
                v = x_train-repmat(muNew(:,l),1,N-N/K);
                u = repmat(w(l,:),d,1).*v;
                SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
            end
            Dalpha = sum(abs(alphaNew-alpha'));
            Dmu = sum(sum(abs(muNew-mu)));
            DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
            Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
            alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
            t = t+1; 
            displayProgress(t,x,alpha,mu,Sigma);
            if Converged
                break
            end
        end
        bic(kford,M)=-2*sum(log(evalGMM(x_test,alpha,mu,Sigma)))+M*(N/100); 
    end
end

[R,C]=size(bic);
for r=1:R
    for c=1:C
        if (isnan(bic(r,c)) + isinf(bic(r,c))>0)
            bic(r,c)=0;
        end  
    end
end
figure(1),plot(1:6,mean(bic,1,'omitnan')),xlabel(sprintf("Model Order")), ylabel("BIC"),title(sprintf("Order Selection-%d",N));
%%%
function test = getTestData(x,kthford)
K=10;
[~,N]=size(x);
test = x(:,(kthford-1)*N/K+1:kthford*N/K);
end

function train = getTrainData(x,kthford)
K=10;
[~,N]=size(x);
x(:,(kthford-1)*N/K+1:kthford*N/K)=[];
train=x;
end

function displayProgress(t,x,alpha,mu,Sigma)
figure(1),
if size(x,1)==2
    subplot(1,2,1), cla,
    plot(x(1,:),x(2,:),'b.'); 
    xlabel('x_1'), ylabel('x_2'), title('Data and Estimated GMM Contours'),
    axis equal, hold on;
    rangex1 = [min(x(1,:)),max(x(1,:))];
    rangex2 = [min(x(2,:)),max(x(2,:))];
    [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2);
    contour(x1Grid,x2Grid,zGMM); axis equal, 
    subplot(1,2,2), 
end
logLikelihood = sum(log(evalGMM(x,alpha,mu,Sigma)));
plot(t,logLikelihood,'b.'); hold on,
xlabel('Iteration Index'), ylabel('Log-Likelihood of Data'),
drawnow; pause(0.1),
end


%%%
function x = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
    
end
end

%%%
function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end

%%%
function [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2)
x1Grid = linspace(floor(rangex1(1)),ceil(rangex1(2)),101);
x2Grid = linspace(floor(rangex2(1)),ceil(rangex2(2)),91);
[h,v] = meshgrid(x1Grid,x2Grid);
GMM = evalGMM([h(:)';v(:)'],alpha, mu, Sigma);
zGMM = reshape(GMM,91,101);
%figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
end

%%%
function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end

%%%
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end