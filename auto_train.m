function y = auto_train(x,t)
%pattern recognition network
net = patternnet(10);
my_weights = importdata('C:\Users\Hang\Desktop\Final Design Matlab\Assignment 3\my_weights.mat');
net = setx(net,my_weights.my_weights);%set the new weights to the neural network
net = train(net,x,t);
view (net);
y = net(x);
perf = perform(net,t,y);
classes = vec2ind(y); %%
%---------save those parameters
my_weights = getx(net);% extract the weights and the bias of the neural network
%------------------------------
%If I want to set to new weights

genFunction(net,'nfunction');
edit nfunction;
save my_weights; %save the weights and bias for then next call of the train fucntion