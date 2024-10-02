class SNNSimulation:
    def __init__(self, x1_percentage, x2_percentage, N=100, thr=0.09, g_input=6, dt=0.005, tau=0.1):
        """
        Initialize the hypothalamic network simulation with given parameters.

        Parameters:
        x1_percentage : float
            Percentage of x1 neurons to be activated (0 to 1).
        x2_percentage : float
            Percentage of x2 neurons to be activated (0 to 1).
        N : int
            Number of neurons (default 100).
        thr : float
            Threshold for firing in the LIF model (default 0.09).
        g_input : float
            Gain of external input to neural network (default 6).
        dt : float
            Time-step for the simulation in seconds (default 0.005s).
        tau : float
            Membrane time constant in seconds (default 0.1s).
        """
        # Network parameters
        self.x1_percentage = x1_percentage
        self.x2_percentage = x2_percentage
        self.N = N
        self.thr = thr
        self.g_input = g_input
        self.dt = dt
        self.tau = tau
        
        # Simulation parameters
        self.time = np.arange(dt, 200, dt)  # Time array
        self.tOn = 2.5   # Start time of external input
        self.tOff = 4.5  # Stop time of external input
        self.gInh = 10   # Inhibition strength
        self.tauI = 0.05  # GABA inhibition time constant
        self.tauS = 20    # Peptide channel time constant
        
        # Weight matrices for neurons receiving external inputs
        self.w1 = np.zeros(N)
        self.w2 = np.zeros(N)
        
        # External stimulus
        self.stim1 = self._generate_external_stimulus()

        # Assign external inputs to neurons
        self._assign_inputs()

    def _generate_external_stimulus(self):
        """
        Generate external input stimulus with three pulses of 2 seconds each.

        Returns:
        stim1 : numpy array
            Smoothed stimulus signal.
        """
        stimRaw = np.zeros_like(self.time)
        stimRaw[(self.time > self.tOn) & (self.time < self.tOff)] = 1
        stimRaw[(self.time > 24.5) & (self.time < 26.5)] = 1
        stimRaw[(self.time > 46.5) & (self.time < 48.5)] = 1
        stimRaw[(self.time > 68.5) & (self.time < 70.5)] = 1
        
        # Smooth stimulus and normalize
        stim1 = np.convolve(stimRaw, np.ones(int(4 / self.dt)) / int(4 / self.dt), mode='same')
        stim1 = stim1 / (np.finfo(float).eps + stim1.max())
        return stim1

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
        plt.plot(self.time - self.tOn, self.stim1)
        plt.title('Smoothed External Stimulus Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Stimulus Intensity')
        plt.show()