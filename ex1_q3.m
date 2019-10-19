N=10;sigma=0.1;
x = 2*rand(1,N)-1;
wTrue=[4 0 -2 0];       % set true parameter vector
v = zeros(1,N);         % set noise
for i=1:N
    v(i) = normrnd(0,sigma);
end

input = zeros(4,N);     % set input of polynomial function
for i=1:N
    input(:,i)=[x(:,i)^3,x(:,i)^2,x(:,i),1];
end
y=wTrue*input+v;

index=0;
steps=(-3:0.1:3);
percent = zeros(5,length(steps));
xplot=zeros(1, length(steps));
for s=steps
    index = index+1;
    gamma = 10^s;
    xplot(index) = s;
    l2=zeros(1,500);
    for i=1:500
        map = @(w)(w)*(w)'*gamma^(-2)+sum((y-w*input).^2*sigma^(-2));   %map function
        w0 = 100*rand(1,4)-50;
        [w,mval] = fminsearch(map,w0);
        l2(:,i) = norm(wTrue-w)^2;
    end
    percent(:,index) = [prctile(l2,0),prctile(l2,25),prctile(l2,50),prctile(l2,75),prctile(l2,100),];

end



figure(1),
plot(xplot,percent(1,:),'g','LineWidth', 2),hold on,
plot(xplot,percent(2,:),'r','LineWidth', 1.5),hold on,
plot(xplot,percent(2,:),'k','LineWidth', 1.25),hold on,
plot(xplot,percent(3,:),'y','LineWidth', 1),hold on,
plot(xplot,percent(4,:),'b','LineWidth', 0.5),hold off,
legend("0%",'25%','50%','75%','100%'),
title("Squared L2 with gamma"),%ylim([0,2]);
xlabel('gamma = 10^x'), ylabel('L2^2');


