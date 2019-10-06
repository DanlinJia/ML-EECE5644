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