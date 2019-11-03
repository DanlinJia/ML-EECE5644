N=999;
p = [0.3,0.7];
mu = [8 0 ;10 0];
Sigma(:,:,1) = [8 5;7 12];
Sigma(:,:,2) = [13 2;4 7];
x = randGMM(N,p,mu,Sigma);

x_1=x(1:2,find(x(3,:,:)==1));
x_2=x(1:2,find(x(3,:,:)==2));
figure(1), plot(x_1(1,:),x_1(2,:),'bo'), hold on; plot(x_2(1,:),x_2(2,:),'r*'), title("Data with True label");
xlabel("x1"),ylabel("x2");
legend("class 1","class 2");

e = map(x,mu,Sigma,p);
e2 = fisherLDA(x_1,x_2,mu,Sigma);
e3 = logsticLearning(x_1,x_2,x);


function f = logsticLearning(x_1,x_2,x)
x_temp1=x_1;
x_temp1(3,:)=1;
x_temp2=x_2;
x_temp2(3,:)=1;
model = @(w)sum(-log((1+exp(w*x_temp1)).^-1))+sum(-log(1-(1+exp(w*x_temp2)).^-1));   % w = [w1 w2 b]
w0 = [1 2 3];
[w,mval] = fminsearch(model,w0);
b=w(3);
w=w(1:2);
y_true1=(1+exp(w*x_1+b)).^(-1);
y_true2=(1+exp(w*x_2+b)).^(-1);
ind11=find(y_true1>0.5);
ind10=find(y_true1<=0.5);
ind00=find(y_true2<0.5);
ind01=find(y_true2>=0.5);
f=length(ind10)+length(ind01);
figure(4), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'.g'); hold on,
plot(x(1,ind01),x(2,ind01),'or'); hold on,
plot(x(1,ind10),x(2,ind10),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'.g'); 
axis equal,

% including the contour at level 0 which is the decision boundary
legend('True Class 1','False Class 1','False Class 2','True Class 2' ), 
title(sprintf('Linear Logistic Learning (error:%d)',f)),
xlabel('x_1'), ylabel('x_2');
end


function f=map(x,mu,Sigma,p)
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x(1:2,:),mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x(1:2,:),mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

ind00 = find(decision==0 & x(3,:)==1);
ind10 = find(decision==1 & x(3,:)==1); 
ind01 = find(decision==0 & x(3,:)==2); 
ind11 = find(decision==1 & x(3,:)==2); 

f = length(ind01)+length(ind10);
figure(2), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

% including the contour at level 0 which is the decision boundary
legend('True Class 1','False Class 1','False Class 2','True Class 2' ), 
title(sprintf('MAP (error:%d)',f)),
xlabel('x_1'), ylabel('x_2');
end

function f = fisherLDA(x_1,x_2,mu,Sigma)
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector
c=w'*0.5*(mu(:,1)-mu(:,2));
y_1 = w'*x_1;
y_2 = w'*x_2;
ind00=find(y_1<c);
ind11=find(y_2>c);
ind10=find(y_1>=c);
ind01=find(y_2<=c);
figure(3),
plot(y_1(1,ind00),zeros(1,length(ind00)),'g.');
hold on;
plot(y_1(1,ind10),zeros(1,length(ind10)),'bo');
hold on;
plot(y_2(1,ind11),zeros(1,length(ind11)),'g.');
hold on;
plot(y_2(1,ind01),zeros(1,length(ind01)),'r*');
f=length(ind10)+length(ind01);
xlabel("Data Projection"), title(sprintf("Fisher LDA (error:%d)",f));
legend("true class 1","false class 1","true class 2", "false class 2");
end

function y=labelData(x,mu,sigma)
[~,s]=size(x);
y=zeros(2,s);
for class=1:s
    if x(3,class)==1
        y(1,class)=evalGaussian(x(1:2,class),mu(:,1),sigma(:,:,1));
        y(2,class)=1;
    else
        y(1,class)=1-evalGaussian(x(1:2,class),mu(:,2),sigma(:,:,2));
        y(2,class)=2;
    end
end
end
    
function x = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d+1,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
    x(1:d,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
    labels(ind) = m;
end
x(d+1,:)=labels;
end

function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
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
