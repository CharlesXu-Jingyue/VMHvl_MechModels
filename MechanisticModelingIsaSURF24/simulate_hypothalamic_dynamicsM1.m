function [r, sig, rampingAct, time, I, stim1] = simulate_hypothalamic_dynamicsM1(x1_percentage, x2_percentage)
    % simulate_hypothalamic_dynamics Simulates the dynamics of a hypothalamic line attractor network
    %   x1_percentage: Percentage of x1 neurons to be activated (0 to 1)
    %   x2_percentage: Percentage of x2 neurons to be activated (0 to 1)

    % N: number of neurons
    % thr: threshold for firing in I&F model
    % g_input: gain of external inputs to neural network

    N = 1000; thr = 0.09; [w1,w2] = deal(zeros(N,1));
    g_input = 6;

    % parameters for simulation
    dt      = 0.005; % time-step of simulation (5ms)
    tau     = 0.1; % membrane time constant (100ms)
    time    = dt:dt:200; % duration of simulation (200s)
    tOn     = 2.5; % start time of external input (s)
    tOff    = 4.5; % stop time of external input (s)

    % create an external input with three pulses of 2s each
    stimRaw = double(time>tOn & time<tOff);
    stimRaw(time>24.5 & time<26.5) = 1;
    stimRaw(time>46.5 & time<48.5) = 1;
    stimRaw(time>68.5 & time<70.5) = 1;
    stim1   = smoothts(stimRaw,'e',4/dt);stim1=stim1/max(eps+stim1);

    time    = time - tOn;

    % parameters for inhibition and time constant of specific channels
    % gInh: gain (strength) of inhibition
    % tauI: time constant of GABA (inhibition) in ms
    % tauS: time constant of Peptide channel

    gInh = 10; g = 1; tauI = 0.05; tauS = 20;

    % set the neurons receiving an external input
    % set the neurons receiving an external input for x1 neurons
    num_x1_neurons = round(N/5 * x1_percentage);
    if num_x1_neurons == 0
        w1 = 0;
    else 
        x1_indices = 1:num_x1_neurons;
        w1(x1_indices) = g_input;
    end

    % set the neurons receiving an external input for x2 neurons
    num_x2_neurons = round(N/5 * x2_percentage);
    if num_x2_neurons == 0
        w2 = 0;
    else
        x2_indices = N/5 + (1:num_x2_neurons);
        w2(x2_indices) = g_input;
    end

    % set the connectivity matrix
    Js = rand(N).*(rand(N)<0.01)/sqrt(N); % make a matrix with very sparse (1%) connectivity

    N_sub = 0.2*N; % set the percent of neurons with higher connectivity
    J_sub = rand(N_sub).*(rand(N_sub)<0.38)/sqrt(N_sub); % set sparsity of subnetwork that forms the line attractor  
    Js(1:N_sub,1:N_sub) = J_sub;

    % add known x2 to x1 connectivity
    Js(1:N_sub,(N_sub+1):(N_sub+200)) = rand(N_sub).*(rand(N_sub)<0.15)/sqrt(N_sub);

    Js(Js<0) = 0;

    % eigenvalue calculation of the connectivity matrix
    eigenvalues = eig(Js);
    lambda_max = max(real(eigenvalues));
    T_eff = tauS / abs(1 - lambda_max);

    fprintf('Individual neuron time constant T (s): %f\n', tauS);
    fprintf('Largest eigenvalue lambda_max: %f\n', lambda_max);
    fprintf('Effective network time constant T_eff (s): %f\n', T_eff);

    % set the variables for simulation
    % r is the spiking activity
    % x is the membrane potential
    % Iin is the input current in a cell
    % I is feedback inhibition
    % p is a variable that captures the dynamics of synaptic conductance

    [r,x,p,pS] = deal(zeros(N,length(stim1))); 
    I       = zeros(1,length(stim1));

    for t = 2:length(time)
   
        % synaptic conductances
        % we first calculate the change in current taking place across a channel within the cell 
        p(:,t)   = r(:,t-1) + (p(:,t-1) - r(:,t-1)) * exp(-dt/tauS);

        % membrane potentials
        % we then calculate the total input within the cell
        % this is a function of the synpases between all cells (Js), the
        % inhibition recieved by the cell (I), the external input (stim1) and
        % some random noise (rand)
        Iin     = max(g*Js*p(:,t-1) - (g*(g+gInh)/1000*I(t-1))...
                  + w1*stim1(t)  + w2*stim1(t) + rand(N,1)*0.1,0);

        % that input current is modified by the time constant of the cells
        % membrane potential
        x(:,t)  = Iin + (x(:,t-1) - Iin) * exp(-dt/tau);

        % spikes
        % we then calculate the spikes generated by thresholding the membrane
        % potential
        r(:,t)  = (x(:,t)>=thr)*1/dt/100;
        x(r(:,t)~=0,t) = 0;

        % inhibition
        % the activity of the cells also triggers inhibition within the
        % circuit, which is modeled here
        I(t) = sum(r(:,t-1)) + (I(t-1) - sum(r(:,t-1))) * exp(-dt/tauI);
    end

    sig         = smoothts(smoothts(double(r),'e',.1/dt),'e',10/dt);
    
    if (x1_percentage < .25 )
        rampingAct = mean(sig((num_x1_neurons + 1):200,:)); % ramping act is calculated for the remaining x1 neurons
    else
        rampingAct = mean(sig(1:200,:));
    end
end