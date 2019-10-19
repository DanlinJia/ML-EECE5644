sigma_x = 0.25; sigma_y = 0.25;sigma=0.1;
pair = zeros(2,1);
ri = -1;
horizontalGrid = linspace(-2,2,101);
verticalGrid = linspace(-2,2,91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
hv = [h(:)';v(:)'];
[a,b] = size(hv) ; 
r=rand();
pair(1) = r*cos(2*pi*rand());
pair(2) = r*sin(2*pi*rand());
%     pair(1) = cos(pi/4);
%     pair(2) = sin(pi/4);
for i=1:9
    r=rand()
    pair(1) = r*cos(2*pi*rand());
    pair(2) = r*sin(2*pi*rand());
    reference=zeros(2,i);
    reference(:) = generateDataOnCircle(i);
    sumi=0;
    for k=1:i  
        dTi = norm(pair - reference(:,k));
        ni = normrnd(0,sigma);
        while ri<0
            ri=dTi+ni;
        end
        M =zeros(1,b);
        E = zeros(1,b);
        for v= 1:b
            M(:,v) = (ri - norm(hv(:,v)-reference(:,k)))^2/sigma^2;  
        end
        sumi = sumi+M;
    end
    for v=1:b
        E(:,v) = evalPrior(hv(:,v),sigma_x, sigma_y );
    end
    map = sumi + E;
    minDSGV = -2;
    maxDSGV = 2;
    mapGrid = reshape(sumi,91,101);
    subplot(3,3,i), 
    contour(horizontalGrid,verticalGrid,-mapGrid,30); hold on,
    plot(pair(1), pair(2), 'gx');hold on,
    plot(reference(1,:), reference(2,:), 'ro');
    legend("Contour",'TruePoint','Reference'), 
    title(sprintf('TruePoint and References for K=%f',i)),
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

% function p = evalPrior(pair, sigma_x, sigma_y)
% % Generate a location based on sigmas
% V(:,:,1) =[sigma_x^2, 0; 0, sigma_y^2];
% C = (2*pi*sigma_x*sigma_y)^(-1);
% E = -0.5*pair'*inv(V)*pair;
% p = C*exp(E);
% end

function p = evalPrior(pair, sigma_x, sigma_y)
% Generate a location based on sigmas
V(:,:,1) =[sigma_x^2, 0; 0, sigma_y^2];
E = pair'*inv(V)*pair;
p = E;
end
