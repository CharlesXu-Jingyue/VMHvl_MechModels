import numpy as np
import matplotlib.pyplot as plt
import time
import sys

class VMHvlSNNSimulation:
    def __init__(self,
        x1_percentage,
        x2_percentage,
        N=1000,
        Np_percentage=0.2,
        threshold=0.1,
        v_r=0,
        tau_m=0.02e3,
        g=1,
        tau_inh=0.05e3,
        g_inh=10,
        tau_s=0.02e3,
        s_on=2.5e3,
        s_duration=2e3,
        s_interval=20e3,
        n_pulses=1,
        g_input=1,
        dt=0.001e3,
        simulation_time=10e3,
        noise=0.1):
        """
        Initialize the hypothalamic network simulation with given parameters.

        Parameters:
        x1_percentage : float
            Percentage of x1 neurons to be activated (0 to 1).
        x2_percentage : float
            Percentage of x2 neurons to be activated (0 to 1).
        N : int
            Number of neurons (default 1000).
        Np_percentage : float
            Percentage of neurons in the subnetwork (default 0.2).
        threshold : float
            Threshold for firing in the LIF model (default 0.09).
        v_r : float
            Reversal potential in the LIF model (default 0).
        tau_m : float
            Membrane time constant in milliseconds (default 100ms).
        g : float
            Gain ratio of the synaptic input (default 1).
        tau_inh : float
            Time constant for the feedback inhibition in milliseconds (default 50ms).
        g_inh : float
            Gain ratio of the feedback inhibition (default 10).
        tau_s : float
            Time constant for the synaptic conductance in milliseconds (default 20s).
        s_on : float
            Start time of external input in milliseconds (default 2500ms).
        s_dur : float
            Duration of external input in milliseconds (default 2000ms).
        s_interval : float
            Interval between external input pulses in milliseconds (default 20000ms).
        n_pulses : int
            Number of pulses in the external input (default 1).
        g_input : float
            Strength of the external input (default 1).
        dt : float
            Time-step for the simulation in milliseconds (default 1ms).
        simulation_time : float
            Total simulation time in milliseconds (default 10000ms).
        noise : float
            Standard deviation of Gaussian noise in the system (default 0.1).
        """

        # Simulation parameters
        self.dt = dt
        self.simulation_time = simulation_time
        self.time = np.arange(dt, self.simulation_time, dt) # Time vector for simulation
        self.noise = noise

        # Network parameters
        self.x1_percentage = x1_percentage
        self.x2_percentage = x2_percentage
        self.N = N
        self.Np = round(N * Np_percentage)
        self.W = self._create_connectivity_matrix()         # Create the connectivity matrix
        self.threshold = threshold
        self.tau_m = tau_m
        self.g = g
        self.tau_inh = tau_inh
        self.g_inh = g_inh
        self.tau_s = tau_s
        
        # Stimulation parameters
        self.s_on = s_on
        self.s_duration = s_duration
        self.s_interval = s_interval
        self.n_pulses = n_pulses    
        self.g_input = g_input
        self.stim = self._generate_external_stimulus()      # Generate external stimulus
        
        # Weight matrices for neurons receiving external inputs
        self.w1 = np.zeros(N)
        self.w2 = np.zeros(N)
        self._assign_inputs()                               # Assign external input to x1 and x2 neurons
                                                            # based on the percentages provided

    def _create_connectivity_matrix(self):
        """
        Create the connectivity matrix for the hypothalamic network.

        Returns:
        W : numpy array
            Connectivity matrix for the network.
        """

        # Create the netowrk connectivity matrix W_ij ~ U(0, 1/sqrt(N))
        N = self.N
        W = np.random.rand(N, N) * (np.random.rand(N, N) < 0.01) / np.sqrt(N)   # Make a matrix with very sparse (1%) connectivity

        # Create the subnetwork connectivity matrix
        Np = self.Np
        Wp = np.random.uniform(0, 1/np.sqrt(N), (N, N))
        Wp = np.random.rand(Np, Np) * (np.random.rand(Np, Np) < 0.36) / np.sqrt(Np)  # Set sparsity of subnetwork
        Np = self.Np
        W[:Np, :Np] = Wp

        return W

    def _generate_external_stimulus(self):
        """
        Generate external input stimulus.

        Returns:
        stim : numpy array
            Smoothed stimulus signal.
        """

        stimRaw = np.zeros_like(self.time)
        s_on_interval = self.s_interval + self.s_duration   # Interval between start of two pulses
        for i in range(self.n_pulses):
            stimRaw[(self.time >= self.s_on + i * s_on_interval) & (self.time < self.s_on + i * s_on_interval + self.s_duration)] = 1
        
        # Smooth stimulus and normalize
        stim = np.convolve(stimRaw, np.ones(100), mode='same')
        stim = stim / (np.finfo(float).eps + stim.max())

        return stim

    def _assign_inputs(self):
        """
        Assign external input to x1 and x2 neurons based on the percentages provided.
        """

        # Assign x1 neurons
        num_x1_neurons = round(self.N / 5 * self.x1_percentage)
        if num_x1_neurons > 0:
            x1_indices = np.arange(0, num_x1_neurons)
            self.w1[x1_indices] = self.g_input

        # Assign x2 neurons
        num_x2_neurons = round(self.N / 5 * self.x2_percentage)
        if num_x2_neurons > 0:
            x2_indices = np.arange(int(self.N / 5), int(self.N / 5) + num_x2_neurons)
            self.w2[x2_indices] = self.g_input

    def plot_stimulus(self):
        """
        Plot the external stimulus over time.
        """

        plt.plot(self.time, self.stim)
        plt.title('Smoothed External Stimulus Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Stimulus Intensity')
        plt.show()

    def run_simulation(self):
        """
        Run the hypothalamic network simulation.
        """

        # Initialize variables
        print('Initializing simulation...')
        x = np.zeros((self.N, 1))   # Membrane potential
        p = np.zeros((self.N, 1))   # Presynaptic current
        I = np.zeros((self.N, 1))   # Inhibitory current
        s = np.zeros((self.N, 1))   # External input current
        spk = np.zeros((self.N, 1)) # Spike status

        # Run the simulation
        print('Simulating...')
        for t in range(len(self.time)):
            # Update the network state
            x, p, I, spk = self._update_network(x, p, I, s, spk)

            # Update the external input current
            s = self.stim[t] * np.ones((self.N, 1))

            # Record the spikes
            if t == 0:
                spk_history = spk
            else:
                spk_history = np.hstack((spk_history, spk))

            # Print the progress
            sys.stdout.write('\r')
            sys.stdout.write(f'Simulation time: {(t+1)*self.dt:.0f}/{(self.time[-1]+1):.0f} ms')
            sys.stdout.flush()
        
        print('\nSimulation complete.')

        # # Smooth the spike history
        # exp_smooth_alpha = 0.99999
        # spk_history_smoothed = np.zeros_like(spk_history)
        # for t in range(1, len(self.time)):
        #     spk_history_smoothed[:, t] = exp_smooth_alpha * spk_history[:, t] + (1 - exp_smooth_alpha) * spk_history_smoothed[:, t-1]

        self.spk_history = spk_history
    
    def _update_network(self, x, p, I, s, spk):
        """
        Update the network state.

        Parameters:
        x : numpy array
            Membrane potential of the neurons.
        p : numpy array
            Presynaptic current of the neurons.
        I : numpy array
            Inhibitory current of the neurons.
        s : numpy array
            External input current of the neurons.
        spk : numpy array
            Spike status of the neurons

        Returns:
        x : numpy array
            Updated membrane potential of the neurons.
        p : numpy array
            Updated presynaptic current of the neurons.
        I : numpy array
            Updated inhibitory current of the neurons.
        spk : numpy array
            Updated spike status of the neurons.
        """

        # Update the presynaptic current
        p = spk + (p - spk) * np.exp(-self.dt / self.tau_s)

        # Update the inhibitory current
        I_inh_input = np.sum(spk) / self.N  # Input to the inhibitory dynamics
        I = I_inh_input + (I - I_inh_input) * np.exp(-self.dt / self.tau_inh)

        # Total input current
        I_total = np.maximum(0, self.g * np.dot(self.W, p) - self.g_inh * I + np.dot(self.w1, s) + np.dot(self.w2, s) + self.noise * np.random.randn(self.N, 1))

        # Update the membrane potential
        x = I_total + (x - I_total) * np.exp(-self.dt / self.tau_m)

        # Update the spike status
        spk, x = self._update_spike(x)

        return x, p, I, spk


    def _update_spike(self, x):
        """
        Update the spike status of the neurons and reset the membrane potential.

        Parameters:
        x : numpy array
            Membrane potential of the neurons.

        Returns:
        spk : numpy array
            Spike status of the neurons.
        x : numpy array
            Membrane potential of the neurons after resetting
        """

        spk = x > self.threshold
        x[spk] = 0

        return spk, x

    def plot_spike_train(self, cell_idx=0):
        """
        Plot the spike train of the network.

        Parameters:
        cell_idx : int or list
            Index of the cell to plot the spike train for.
        """

        if cell_idx is None:
            cell_idx = np.arange(self.N)

        if self.spk_history is not None:
            if isinstance(cell_idx, int):
                plt.figure(figsize=(6, 3))
                spk_times = self.spk_history[cell_idx, :].nonzero()[0] * self.dt
                plt.vlines(spk_times, 0, 1, linewidth=1)
                plt.xlim([0, self.simulation_time])
                plt.title(f'Spike Train of Neuron {cell_idx}')
                plt.xlabel('Time (ms)')
                plt.ylabel('Spike')
                plt.show()
            else:
                for idx in cell_idx:
                    plt.figure(figsize=(6, 3))
                    spk_times = self.spk_history[idx, :].nonzero()[0] * self.dt
                    plt.vlines(spk_times, 0, 1, linewidth=1)
                    plt.xlim([0, self.simulation_time])
                    plt.title(f'Spike Train of Neuron {idx}')
                    plt.xlabel('Time (ms)')
                    plt.ylabel('Spike')
                    plt.show()
        else:
            print('No spike history found. Run the simulation first.')

    def plot_spike_rate(self, cell_idx=0):
        """
        Plot the spike rate of cells in the network.

        Parameters:
        cell_idx : int or list
            Index of the cell to plot the spike rate for.
        """

        if cell_idx is None:
            cell_idx = np.arange(self.N)

        if self.spk_history is not None:
            if isinstance(cell_idx, int):
                window_size = int(round(1 / self.dt * 1e3))
                spk_rate = np.convolve(self.spk_history[cell_idx, :], np.ones(window_size), mode='same') / window_size
                plt.figure(figsize=(6, 3))
                plt.plot(self.time, spk_rate)
                plt.title(f'Spike Rate of Neuron {cell_idx}')
                plt.xlabel('Time (ms)')
                plt.ylabel('Spike Rate')
                plt.show()
            else:
                for idx in cell_idx:
                    window_size = int(round(1 / self.dt * 1e3))
                    spk_rate = np.convolve(self.spk_history[idx, :], np.ones(window_size), mode='same') / window_size
                    plt.figure(figsize=(6, 3))
                    plt.plot(self.time, spk_rate)
                    plt.title(f'Spike Rate of Neuron {idx}')
                    plt.xlabel('Time (ms)')
                    plt.ylabel('Spike Rate')
                    plt.show()
        else:
            print('No spike history found. Run the simulation first.')

    def plot_raster(self, cell_indices=None):
        """
        Plot the raster plot of the network.

        Parameters:
        cell_indices : list
            List of cell indices to plot the raster plot for.
        """

        if cell_indices is None:
            cell_indices = np.arange(self.N)
        
        if self.spk_history is not None:
            plt.figure(figsize=(6, 5))
            plt.imshow(self.spk_history[cell_indices, :], aspect='auto', cmap='viridis', origin='lower')
            plt.title('Raster Plot of the Network')
            plt.xlabel('Time (ms)')
            plt.ylabel('Neuron Index')
            plt.show()
        else:
            print('No spike history found. Run the simulation first.')

    def plot_network_rate(self):
        """
        Plot the firing rate of the network.
        """
        
        if self.spk_history is not None:
            firing_rate = np.sum(self.spk_history, axis=0) / self.N
            plt.figure(figsize=(6, 5))
            plt.plot(self.time, firing_rate)
            plt.title('Firing Rate of the Network')
            plt.xlabel('Time (ms)')
            plt.ylabel('Firing Rate')
            plt.show()
        else:
            print('No spike history found. Run the simulation first.')