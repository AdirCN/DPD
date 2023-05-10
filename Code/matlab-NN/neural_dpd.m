realInput = (cell2mat(struct2cell(load('C:/Users/magshimim/Desktop/DPD/DPD/pa_data/input.mat'))));
output = transpose(cell2mat(struct2cell(load('output.mat'))));
o = output(1,:)+(1i)*output(2,:);
scale_factor = max(realInput)/(max(o));
scaled_output = o*scale_factor;

DPDinput = transpose(create_matrix(scaled_output));
DPDouput = transpose([transpose(real(realInput)), transpose(imag(realInput))]);

netDPD = feedforwardnet([22 25 35 30 20 10 2],'trainlm');
netDPD=train(netDPD,DPDinput,DPDouput);





