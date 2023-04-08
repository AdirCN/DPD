realInput = (cell2mat(struct2cell(load('C:/Users/magshimim/Desktop/DPD/DPD/pa_data/input.mat'))));
output = transpose(cell2mat(struct2cell(load('output.mat'))));
input = transpose(create_matrix(realInput));

without_dpd = netPA(input);
without_dpd = without_dpd(1,:) + 1j*without_dpd(2,:);

with_dpd_mid = netDPD(input);
with_dpd_mid = with_dpd_mid(1,:) + 1j*with_dpd_mid(2,:);
with_dpd_mid = transpose(create_matrix(with_dpd_mid));
with_dpd = netPA(with_dpd_mid);
with_dpd = with_dpd(1,:) + 1j*with_dpd(2,:);

figure(1);
plot(abs(realInput), abs(without_dpd), ".");
hold on;
plot(abs(realInput), abs(with_dpd), ".");

figure(2);
Fs = 3*(10^6);
f= linspace(-(Fs/2), (Fs/2), length(with_dpd));
withoutDPD = fftshift(fft(without_dpd));
plot(f,db(abs(withoutDPD)));
hold on;
withDPD = fftshift(fft(with_dpd));
plot(f,db(abs(withDPD)));
%hold on;

%O = fftshift(fft(o));
%plot(f,abs(O));

%plot(abs(h));
%hold on;
%figure;
%plot(abs(o));
%save netPA;
