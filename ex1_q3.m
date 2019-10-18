N=10;
x = zeros(1,N);
sigma=0.1;
wTrue=[4 0 -2 0];
for i=1:N
    x(i)=2*rand()-1;
end
v = zeros(1,N);
for i=1:N
    v(i) = normrnd(0,sigma);
end
y = polyval(w,x)+v
lambda=1;

for gamma=10^-2:10^2
    gamma
end



function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,4,N)).*(inv(Sigma)*(x-repmat(mu,4,N))),1);
g = C*exp(E);
end
