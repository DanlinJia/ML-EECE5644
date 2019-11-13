close all, clear all,
N=1000; 
K=10;
p=[0.35 0.65];
x=zeros(1,N);
y=zeros(1,N);
for i=1:N
    if rand()<0.35
        x(1,i) = normrnd(0,1);
        x(2,i) = normrnd(0,1);
        y(i) = -1;
    else
        r = rand()+2;
        o = rand()*2*pi-pi;
        x(1,i) =cos(o)*r;
        x(2,i) =sin(o)*r;
        y(i)=1;
    end
end

% Train a Gaussian kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)
dummy = ceil(linspace(0,N,K+1));
for k = 1:K, indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; end,
CList = 10.^linspace(-1,9,11); sigmaList = 10.^linspace(-2,3,13);
for sigmaCounter = 1:length(sigmaList)
    [sigmaCounter,length(sigmaList)],
    sigma = sigmaList(sigmaCounter);
    for CCounter = 1:length(CList)
        C = CList(CCounter);
        for k = 1:K
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            xValidate = x(:,indValidate); % Using folk k as validation set
            lValidate = y(indValidate);
            if k == 1
                indTrain = [indPartitionLimits(k,2)+1:N];
            elseif k == K
                indTrain = [1:indPartitionLimits(k,1)-1];
            else
                indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1];
            end
            % using all other folds as training set
            xTrain = x(:,indTrain); lTrain = y(indTrain);
            SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','gaussian','KernelScale',sigma);
            dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
            indCORRECT = find(lValidate.*dValidate == 1); 
            Ncorrect(k)=length(indCORRECT);
        end 
        PCorrect(CCounter,sigmaCounter)= sum(Ncorrect)/N;
    end 
end
figure(2), subplot(1,2,1),
contour(log10(CList),log10(sigmaList),PCorrect',20); xlabel('log_{10} C'), ylabel('log_{10} sigma'),
title('Gaussian-SVM Cross-Val Accuracy Estimate'), axis equal,
[dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
CBest= CList(indBestC); sigmaBest= sigmaList(indBestSigma); 
SVMBest = fitcsvm(x',y','BoxConstraint',CBest,'KernelFunction','gaussian','KernelScale',sigmaBest);
d = SVMBest.predict(x')'; % Labels of training data using the trained SVM
indINCORRECT = find(y.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(y.*d == 1); % Find training samples that are correctly classified by the trained SVM
figure(2), subplot(1,2,2), 
plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
plot(x(1,indINCORRECT),x(2,indINCORRECT),'r.'), axis equal,
title('Training Data (RED: Incorrectly Classified)'),
pTrainingError = length(indINCORRECT)/N, % Empirical estimate of training error probability
Nx = 1001; Ny = 990; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
[h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
figure(2), subplot(1,2,2), contour(xGrid,yGrid,zGrid,0); xlabel('x1'), ylabel('x2'), axis equal,

x_test=zeros(1,N);
y_test=zeros(1,N);
for i=1:N
    if rand()<0.35
        x_test(1,i) = normrnd(0,1);
        x_test(2,i) = normrnd(0,1);
        y_test(i) = -1;
    else
        r = rand()+2;
        o = rand()*2*pi-pi;
        x_test(1,i) =cos(o)*r;
        x_test(2,i) =sin(o)*r;
        y_test(i)=1;
    end
end


%SVMBest_test = fitcsvm(x_test',y_test,'BoxConstraint',CBest,'KernelFunction','linear','KernelScale',sigmaBest);
d = SVMBest.predict(x_test')'; % Labels of training data using the trained SVM
indINCORRECT_test = find(y_test.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT_test = find(y_test.*d == 1); % Find training samples that are correctly classified by the trained SVM
figure(3),
plot(x(1,indCORRECT_test),x(2,indCORRECT_test),'g.'), hold on,
plot(x(1,indINCORRECT_test),x(2,indINCORRECT_test),'r.'), axis equal,
title('Test Data (RED: Incorrectly Classified)');

