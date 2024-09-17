%funct implementation for mechanistic modeling
% functional implementation for comparing M1 to M2
clear all;

N = 1000;

x1_percentage = 0.1;  % 10%
x2_percentage = 0;  % 0%

gInh = 10;
dt = 0.005;

gNP_ratio = 1.45;
gNP = gNP_ratio * gInh;

num_repeats = 1;

% for M1
tau_x1_valuesM1 = zeros(1, num_repeats);
tau_x2_valuesM1 = zeros(1, num_repeats);

for repeat_idx = 1:num_repeats
    [r_M1, sig_M1, rampingAct_M1, time_M1, I_M1, stim1_M1] = simulate_hypothalamic_dynamicsM1(x1_percentage, x2_percentage);

    thrR_M1 = 0.05;
    response_M1 = max(sig_M1, [], 2);

    plottime_M1 = (1:size(sig_M1,2)) * dt;
    r_firing_M1 = r_M1(response_M1 > thrR_M1, :);

    x1_regress_M1 = 1:length(rampingAct_M1(17000:end));
    f_ramp_x1_M1 = fit(x1_regress_M1', rampingAct_M1(17000:end)', 'exp1');
    tau_x1M1 = 1 / -f_ramp_x1_M1.b * dt;
    tau_x1_valuesM1(repeat_idx) = tau_x1M1;

    x2_activity_M1 = mean(sig_M1(201:400,:));
    x2_regress_M1 = 1:length(x2_activity_M1(17000:end));
    f_ramp_x2_M1 = fit(x2_regress_M1', x2_activity_M1(17000:end)', 'exp1');
    tau_x2M1 = 1 / -f_ramp_x2_M1.b * dt;
    tau_x2_valuesM1(repeat_idx) = tau_x2M1;
end

% for M2

tau_x1_valuesM2 = zeros(1, num_repeats);
tau_x2_valuesM2 = zeros(1, num_repeats);

for repeat_idx = 1:num_repeats
    [r_M2, sig_M2, rampingAct_M2, time_M2, I_M2, stim1_M2] = simulate_hypothalamic_dynamicsM2(x1_percentage, x2_percentage, gNP);

    thrR_M2 = 0.05;
    response_M2 = max(sig_M2, [], 2);

    plottime_M2 = (1:size(sig_M2,2)) * dt;
    r_firing_M2 = r_M2(response_M2 > thrR_M2, :);

    x1_regress_M2 = 1:length(rampingAct_M2(17000:end));
    f_ramp_x1_M2 = fit(x1_regress_M2', rampingAct_M2(17000:end)', 'exp1');
    tau_x1M2 = 1 / -f_ramp_x1_M2.b * dt;
    tau_x1_valuesM2(repeat_idx) = tau_x1M2;

    x2_activity_M2 = mean(sig_M2(201:400,:));
    x2_regress_M2 = 1:length(x2_activity_M2(17000:end));
    f_ramp_x2_M2 = fit(x2_regress_M2', x2_activity_M2(17000:end)', 'exp1');
    tau_x2M2 = 1 / -f_ramp_x2_M2.b * dt;
    tau_x2_valuesM2(repeat_idx) = tau_x2M2;
end

%%
% Graphing tau for x1 and x2 neurons for both models
% Calculate mean and standard deviation for tau integration values
mean_tau_x1_M1 = mean(tau_x1_valuesM1);
std_tau_x1_M1 = std(tau_x1_valuesM1);

mean_tau_x2_M1 = mean(tau_x2_valuesM1);
std_tau_x2_M1 = std(tau_x2_valuesM1);

mean_tau_x1_M2 = mean(tau_x1_valuesM2);
std_tau_x1_M2 = std(tau_x1_valuesM2);

mean_tau_x2_M2 = mean(tau_x2_valuesM2);
std_tau_x2_M2 = std(tau_x2_valuesM2);

% Plot bar graph of tau integration values for both experiments
clf;
close all;
figure;
hold on;

bar_width = 0.4;
x_positions = [1, 2, 4, 5];

% Bar heights
bar(x_positions(1), mean_tau_x1_M1, bar_width, 'FaceColor', 'b');
bar(x_positions(2), mean_tau_x2_M1, bar_width, 'FaceColor', 'r');
bar(x_positions(3), mean_tau_x1_M2, bar_width, 'FaceColor', 'b');
bar(x_positions(4), mean_tau_x2_M2, bar_width, 'FaceColor', 'r');

% Error bars
errorbar(x_positions(1), mean_tau_x1_M1, std_tau_x1_M1, 'k', 'LineWidth', 1);
errorbar(x_positions(2), mean_tau_x2_M1, std_tau_x2_M1, 'k', 'LineWidth', 1);
errorbar(x_positions(3), mean_tau_x1_M2, std_tau_x1_M2, 'k', 'LineWidth', 1);
errorbar(x_positions(4), mean_tau_x2_M2, std_tau_x2_M2, 'k', 'LineWidth', 1);

% Customize plot
set(gca, 'XTick', [1.5, 4.5], 'XTickLabel', {'M1', 'M2'});
xlabel('Model');
ylabel('\tau_{integration}');
title1 = sprintf('Comparison of \\tau_{integration} for M1 and M2 with gNP/gInh = %.2f : %.2f%% x1, %.2f%% x2', gNP_ratio, x1_percentage*100, x2_percentage*100);
title(title1);
legend({'\tau_x1', '\tau_x2'}, 'Location', 'northwest');
hold off;
 
%%
% plots from original code for M1 ramping act of x1 and x2 
% we finally plot the spiking activity (r), smoothed calcium signal (sig),
% the external input (stim1) and the inhibition recieved by the circuit (I)
clf;
close all;

figure
title1 = sprintf('gNP/gInh = %.2f with %.2f%% x1, %.2f%% x2 activated for M1',gNP_ratio, x1_percentage*100,x2_percentage*100);
sgtitle(title1);
subplot(1,2,1)
imagesc(r_M1)
colormap(flipud(gray))
% 
xticklabels = round(time_M1(1):15:time_M1(end));
xticks = linspace(1, size(r_firing_M1, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
xlabel('Time from stim onset(s)')
ylabel('neurons')

subplot(1,2,2)
imagesc(sig_M1)
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)


figure
sgtitle(title1);
subplot(5,1,1)
rampingAct = mean(sig_M1(1:200,:));

plot(time_M1,rampingAct,'k')

xlabel('Time from stim onset(s)')
ylabel('integration subnetwork')

subplot(5,1,2)
rampingAct = mean(sig_M1(200:400,:));
plot(time_M1,rampingAct,'k')
xlabel('Time from stim onset(s)')
ylabel('non-integration network')

subplot(5,1,3)
rampingAct = mean(sig_M1(400:end,:));
plot(time_M1,rampingAct,'k')
xlabel('Time from stim onset(s)')
ylabel('remaining neurons')

subplot(5,1,4)
plot(time_M1,(stim1_M1),'k')
xlabel('Time from stim onset(s)')
ylabel('input activity')

subplot(5,1,5)
plot(time_M1,(I_M1),'k')
xlabel('Time from stim onset(s)')
ylabel('inhibition')

%%
% plots from original code for M2 ramping act of x1 and x2 
% we finally plot the spiking activity (r), smoothed calcium signal (sig),
% the external input (stim1) and the inhibition recieved by the circuit (I)
clf;
close all;

figure
title1 = sprintf('gNP/gInh = %.2f with %.2f%% x1, %.2f%% x2 activated for M2',gNP_ratio, x1_percentage*100,x2_percentage*100);
sgtitle(title1);
subplot(1,2,1)
imagesc(r_M2)
colormap(flipud(gray))
% 
xticklabels = round(time_M2(1):15:time_M2(end));
xticks = linspace(1, size(r_firing_M2, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
xlabel('Time from stim onset(s)')
ylabel('neurons')

subplot(1,2,2)
imagesc(sig_M2)
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

figure
sgtitle(title1);
subplot(5,1,1)
rampingAct = mean(sig_M2(1:200,:));

plot(time_M2,rampingAct,'k')

xlabel('Time from stim onset(s)')
ylabel('integration subnetwork')

subplot(5,1,2)
rampingAct = mean(sig_M2(200:400,:));
plot(time_M2,rampingAct,'k')
xlabel('Time from stim onset(s)')
ylabel('non-integration network')

subplot(5,1,3)
rampingAct = mean(sig_M2(400:end,:));
plot(time_M2,rampingAct,'k')
xlabel('Time from stim onset(s)')
ylabel('remaining neurons')

subplot(5,1,4)
plot(time_M2,(stim1_M2),'k')
xlabel('Time from stim onset(s)')
ylabel('input activity')

subplot(5,1,5)
plot(time_M2,(I_M2),'k')
xlabel('Time from stim onset(s)')
ylabel('inhibition')
