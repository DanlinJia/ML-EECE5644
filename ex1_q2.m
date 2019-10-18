sigma_x = 0.25; sigma_y = 0.25;
pair = zeros(2,1);
measurement = -1;
map = 1;
N = 100;
x = 4 * rand(2, N) - 2;
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);

for i=1:1
    reference=zeros(2,i);
    pair(1) = rand();
    pair(2) = rand();
    testPoint = generateTestLocation(pair, sigma_x, sigma_y);
    for k=1:i
        reference(:,k) = generateDataOnCircle();
        dTi = norm([h(:)';v(:)'] - reference(:,k))/N;
        testDistance = norm(pair - reference(:,k));
        E = evalGaussian(testDistance, dTi, 0.1);
        hv = [h(:)';v(:)'];
        [a,b] = size(hv) ;
        C =zeros(a,b);
        for v= 1:b
            C(:,v) = generateTestLocation(hv(:,v),sigma_x, sigma_y);   
        end
        %P = generateTestLocation(, sigma_x, sigma_y);
        map = map * E * C
    end
    discriminantScoreGridValues = map;
    minDSGV = -2;
    maxDSGV = 2;
    discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
    figure(i),
    plot(pair(1), pair(2), 'gx'),
    plot(reference(:,1), pair(:,2), 'ro'),
    contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]);
end



function k=generateDataOnCircle()
x=rand();
y=sqrt(1-x^2);
v(:,1) = [x;y];
k=v;
end


function p = generateTestLocation(pair, sigma_x, sigma_y)
% Generate a location based on sigmas
V(:,:,1) =[sigma_x^2, 0; 0, sigma_y^2];
C = (2*pi*sigma_x*sigma_y)^(-1);
E = -0.5*pair.*inv(V)*pair;
p = C*exp(E);
end


function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
