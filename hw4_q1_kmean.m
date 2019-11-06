delta = 1e-5;
K=4;
imdata = imread('f2.jpg');
imdata = im2double(imdata);
[R,C,RGB] = size(imdata);
N=R*C;
features = zeros(5,N);
for i=1:R
    for j=1:C
        features(1,(i-1)*C+j)=i/R;
        features(2,(i-1)*C+j)=j/C;
        features(3:5,(i-1)*C+j)=imdata(i,j,:);
    end
end
labels = zeros(R,C);

% Initialize the GMM to randomly selected samples
shuffledIndices = randperm(N);
centriod = features(:,shuffledIndices(1:K));

for t=1:100
    [~,assignedCentroidLabels] = min(pdist2(centriod',features'),[],1);
    for i=1:K
        centriod(:,i)=mean(features(:,find(assignedCentroidLabels==i)),2);
    end
end

for i=1:R
    for j=1:C
        labels(i,j)=assignedCentroidLabels((i-1)*C+j);
    end
end



displayprogress(1,C,R,labels);

function p=displayprogress(f,C,R,labels)
figure(f),
s=pcolor(1:C,R-(1:R)+1, labels);
set(s, 'EdgeColor', 'none');
title("Kmeans Cluster Result For Figure 1"),
drawnow; pause(0.1);
end
