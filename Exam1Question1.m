m(:,1) = [-1;0]; Sigma(:,:,1) = 0.1*[10 -4;-4,5]; % mean and covariance of data pdf conditioned on label 3
m(:,2) = [1;0]; Sigma(:,:,2) = 0.1*[5 0;0,2]; % mean and covariance of data pdf conditioned on label 2
m(:,3) = [0;1]; Sigma(:,:,3) = 0.1*eye(2); % mean and covariance of data pdf conditioned on label 1
classPriors = [0.15,0.35,0.5]; thr = [0,cumsum(classPriors)];
N = 10000; u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
figure(1),clf, colorList = 'rbg';
record_N = zeros(1,3);
for l = 1:3
    indices = find(thr(l)<=u & u<thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices));
    x(:,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
    record_N(l) = length(indices); % count the numer of samples for each class
    figure(1), plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
end
legend('Class 1','Class 2', 'Class3'), 
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 

poss = zeros(N,3); % 	posterior probability of 3 classes
for l = 1: 3
    poss(:,l) = evalGaussian(x, m(:,l), Sigma(:,:,l))*classPriors(l);
end

decision = zeros(1, N); % make decisions
for l=1:N
    decision(l)= find(poss(l,:) == max(poss(l,:)));
end

confusion = zeros(3);
error_p = zeros(3);
for r=1:3
    for c=1:3
        confusion(r,c) = length(find(decision==r & L==c)); % confusion matrix where the decision is r and the true lable is c
        error_p(r,c) = confusion(r,c)/record_N(c);
    end
end

errors = N - (confusion(1,1)+ confusion(2,2)+ confusion(3,3)) % total errors

error_probability = errors/N % estimate probability of error


figure(2), % class 1 circle, class 2 +, class 3 x, correct green, incorrect red
plot(x(1, find(decision==1 & L==1)),x(2, find(decision==1 & L==1)),'og'); hold on,
plot(x(1,find(decision==2 & L==2)), x(2,find(decision==2 & L==2)),'+g'); hold on,
plot(x(1,find(decision==3 & L==3)), x(2,find(decision==3 & L==3)),'xg'); hold on,
plot(x(1,find(decision~=1 & L==1)), x(2,find(decision~=1 & L==1)),'or'); hold on,
plot(x(1,find(decision~=2 & L==2)), x(2,find(decision~=2 & L==2)),'+b'); hold on,
plot(x(1,find(decision~=3 & L==3)), x(2,find(decision~=3 & L==3)),'xy'); hold on,
axis equal,

% including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 1', 'Correct decisions for data from Class 2',  'Correct decisions for data from Class 3', 'Wrong decisions for data from Class 1','Wrong decisions for data from Class 2','Wrong decisions for data from Class 3'), 
title('Data and their classifier decisions versus true labels'),
xlabel('x_1'), ylabel('x_2'),




function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
