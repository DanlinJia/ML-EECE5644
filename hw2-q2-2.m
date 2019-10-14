% Class Number
n = 2; 
% Sample Number
N = 400; 
% Set mu and sigma
mu(:,1) = [0;0]; mu(:,2) = [3;3];
Sigma(:,:,1) = [1 0;0 1]; Sigma(:,:,2) = [1 0;0 1];
% Class Prior
p = [0.5,0.5];
% Generate Label
label = rand(1,N) >= p(1);
% Length of each class
Nc = [length(find(label==0)),length(find(label==1))];
x = zeros(n,N);
% Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
figure(2), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 1','Class 2'), 
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 

lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
p = [p10,p01]*Nc'/N; % probability of error, empirically estimated

figure(1), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

% including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 1','Wrong decisions for data from Class 1','Wrong decisions for data from Class 2','Correct decisions for data from Class 2' ), 
title('Data and their classifier decisions versus true labels'),
xlabel('x_1'), ylabel('x_2'), 
