realInput = (cell2mat(struct2cell(load('C:/Users/magshimim/Desktop/DPD/DPD/pa_data/input.mat'))));
input = transpose(create_matrix(realInput));
output = transpose(cell2mat(struct2cell(load('output.mat'))));

netPA = feedforwardnet([40],'trainlm');
netPA=train(netPA,input,output);







