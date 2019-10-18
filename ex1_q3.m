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
y = polyval(wTrue,x)+v;
input = zeros(3,N);
for order=1:4
    a = x;
    if order==4
        a = ones(1,N);
    else
        for i=1:order
            a = a.*x;
        end
    end
    input(order,:)=a;
end
input = [input ]
lambda=1;
input;

percent = zeros(4,
for gamma=
    result=zeros(1,100);
    for i=1:100
        map = @(w)(w)*(w)'/gamma^2+sum(y-w*input/sigma^2);
        x0 = rand(1,4);
        [w,mval] = fminunc(map,x0);
        result(:,i) = norm(wTrue-w);
    end
    
    percen = 
    prctile(result,)
end

figure(1),
plot




function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
