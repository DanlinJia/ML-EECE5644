N=10;
x = zeros(1,N);
sigma=0.1;
wTrue=[1 0 -0.25 0];
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
lambda=1;
input;


index=0;
steps=(-3:1:3);
percent = zeros(4,length(steps));
xplot=zeros(1, length(steps));
for s=steps
    index = index+1;
    gamma = 10^s;
    xplot(index) = gamma;
    l2=zeros(1,10);
    for i=1:10
        map = @(w)(w)*(w)'*gamma^(-2)+sum((y-w*input).^2*sigma^(-2));
        x0 = rand(1,4);
        [w,mval] = fminsearch(map,x0);
        l2(:,i) = norm(wTrue-w)^2;
    end
    percent(:,index) = [prctile(l2,0),prctile(l2,25),prctile(l2,75),prctile(l2,100),];

end



figure(1),
plot(xplot,percent(1,:),'g'),hold on,
plot(xplot,percent(2,:),'r'),hold on,
plot(xplot,percent(3,:),'y'),hold on,
plot(xplot,percent(4,:),'b'),hold on,
legend("0%",'25%','75%','100%'),
title("Squared L2 with gamma"),%ylim([0,2]);
xlabel('gamma'), ylabel('L2^2');




function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
