sigma_x = 0.25; sigma_y = 0.25;
pair = zeros(2,1);
measurement = -1;
map = 0;
horizontalGrid = linspace(floor(-2),ceil(2),101);
verticalGrid = linspace(floor(-2),ceil(2),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
hv = [h(:)';v(:)'];
[a,b] = size(hv) ; 

for i=1:4
    reference=zeros(2,i);
    pair(1) = rand();
    pair(2) = rand();
    reference(:) = generateDataOnCircle(i);
    for k=1:i   
        dTi = norm(pair - reference(:,k));
        C =zeros(1,b);
        E = zeros(1,b);
        for v= 1:b
            C(:,v) = norm(hv(:,v)-reference(:,k));  
            E(:,v) = evalPrior(hv(:,v),sigma_x, sigma_y );
        end
        map = map + log(evalGaussian(C, dTi, 0.1)) + log(E);
    end
    minDSGV = -2;
    maxDSGV = 2;
    mapGrid = reshape(map,91,101);
    subplot(2,2,i), 
    %contour(horizontalGrid,verticalGrid,mapGrid,[minDSGV*[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1],0,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]*maxDSGV]); hold on,
    contour(horizontalGrid,verticalGrid,mapGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); hold on,
    plot(pair(1), pair(2), 'gx');hold on,
    plot(reference(1,:), reference(2,:), 'ro');
    legend("Contour",'TruePoint','Reference'), 
    title('TruePoint and References'),
    xlabel('x'), ylabel('y');
    
end



function k=generateDataOnCircle(number)
v = zeros(2, number);
t1 = 0;
for i=1:number
    x=cos(t1);
    y=sin(t1);
    v(:,i) = [x;y];
    t1 = t1 + 2*pi/number;
end
k=v;
end

function p = evalPrior(pair, sigma_x, sigma_y)
% Generate a location based on sigmas
V(:,:,1) =[sigma_x^2, 0; 0, sigma_y^2];
C = (2*pi*sigma_x*sigma_y)^(-1);
E = -0.5*pair'*inv(V)*pair;
p = C*exp(E);
end


function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
