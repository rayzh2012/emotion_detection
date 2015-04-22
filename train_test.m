x = importdata('C:\Users\Hang\Desktop\Final Design Matlab\Assignment 3\train.mat');
t = importdata('C:\Users\Hang\Desktop\Final Design Matlab\Assignment 3\target1.mat');
t = t';
%pattern recognition network
net = patternnet(10);
net = train(net,x,t);
view (net);
y = net(x);
perf = perform(net,t,y);
classes = vec2ind(y); %%
%---------save those parameters
my_weights = getx(net);% extract the weights and the bias of the neural network
%------------------------------
%If I want to set to new weights
net = setx(net,my_weights);%set the new weights to the neural network
save net;