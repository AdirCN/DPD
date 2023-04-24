realInput = (cell2mat(struct2cell(load('C:/Users/magshimim/Desktop/DPD/DPD/pa_data/input.mat'))));
output = transpose(cell2mat(struct2cell(load('output.mat'))));
input = transpose(create_matrix(realInput));
load('(22-40-2PA).mat');
load('(22 25 35 30 20 10 2DPD).mat');

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
title("AM-AM with and without dpd");
xlabel("Amplitude");
ylabel("Amplitude");

figure(2);
Fs = 3*(10^6);
f= linspace(-(Fs/2), (Fs/2), length(with_dpd));
withoutDPD = fftshift(fft(without_dpd));
plot(f,db(abs(withoutDPD)));
hold on;
withDPD = fftshift(fft(with_dpd));
plot(f,db(abs(withDPD)));
title("Output spectrum with and without dpd");
xlabel("f [Hz]");
ylabel("Amplitude [dB]");

oobwidpd = (norm((abs(withDPD(1100:4500))).^2)+norm((abs(withDPD(6000:9400))).^2))/norm((abs(withDPD(4500:6000))).^2);
oobwodpd = (norm((abs(withoutDPD(1100:4500))).^2)+norm((abs(withoutDPD(6000:9400))).^2))/norm((abs(withoutDPD(4500:6000))).^2);
OOB = 10*log10(oobwidpd/oobwodpd);

figure(3);
epsilon = 0.0000000000000000000000001;
a = atan(imag(realInput)./(real(realInput)+epsilon));
b = atan(imag(without_dpd)./(real(without_dpd)+epsilon));
c = atan(imag(with_dpd)./(real(with_dpd)+epsilon));
plot(abs(realInput),a-b,".");
hold on;
plot(abs(realInput),a-c,".");
title("AM-PM with and without dpd");
xlabel("Amplitude");
ylabel("phase difference");

output = output(1,:) + 1j*output(2,:);

figure(4);
plot(abs(realInput), abs(without_dpd), ".");
hold on;
plot(abs(realInput), abs(output), ".");
title("NN's output and measured AM-AM");
xlabel("Amplitude");
ylabel("Amplitude");

figure(5);
epsilon = 0.0000000000000000000000001;
a = atan(imag(realInput)./(real(realInput)+epsilon));
b = atan(imag(without_dpd)./(real(without_dpd)+epsilon));
c = atan(imag(output)./(real(output)+epsilon));
plot(abs(realInput),a-b,".");
hold on;
plot(abs(realInput),a-c,".");
title("NN's output and measured AM-PM");
xlabel("Amplitude");
ylabel("phase difference");

figure(6);
Fs = 3*(10^6);
f= linspace(-(Fs/2), (Fs/2), length(with_dpd));
withoutDPD = fftshift(fft(without_dpd));
plot(f,db(abs(withoutDPD)));
hold on;
withDPD = fftshift(fft(output));
plot(f,db(abs(withDPD)));
title("NN's output and measured output spectrum");
xlabel("f [Hz]");
ylabel("Amplitude [dB]");

mdl = fitlm(abs(realInput),abs(with_dpd));
Gain = mdl.Coefficients.Estimate(2);
desired = realInput*Gain;
MMSE = 10*log10(norm(abs(desired) - abs(with_dpd))/norm(abs(desired)));
R_squared = mdl.Rsquared.Ordinary;

