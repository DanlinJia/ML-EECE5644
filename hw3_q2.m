N=999;
alpha_true = [0.3,0.7];
mu_true = [8 10 ;-10 0];
Sigma_true(:,:,1) = [4 5;8 12];
Sigma_true(:,:,2) = [12 2;4 20];
x = randGMM(N,alpha_true,mu_true,Sigma_true);

x_1=x(:,:,find(x(:,:,:)==1));
figure(1), plot(x(1,:),x(2,:),'b.');


function x = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d+1,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
    x(1:d,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
    labels(ind) = 1;
    ind
end
x(d+1,:) = labels;
end

function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end

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