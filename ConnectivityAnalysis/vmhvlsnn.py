import numpy as np
import matplotlib.pyplot as plt

class SNN:
    def __init__(self,
        N=None,
        W=None,
        tau_m=20.0,
        V_rest=-70.0,
        V_th=-50.0,
        V_reset=-75.0,
        refractory_period=5,
        noise=0.1):
        """
        Initialize the network with given parameters.
        :param N: Number of neurons (must be provided if W is not)
        :param W: Connectivity matrix (NxN), optional
        :param tau_m: Membrane time constant (ms)
        :param V_rest: Resting potential (mV)
        :param V_th: Spike threshold (mV)
        :param V_reset: Reset potential (mV)
        :param refractory_period: Refractory period (ms)
        :param noise: Noise level (standard deviation of Gaussian noise)
        """
        if N is None and W is None:
            raise ValueError("Either N or W must be provided.")
        if W is not None:
            if not isinstance(W, np.ndarray) or W.ndim != 2 or W.shape[0] != W.shape[1]:
                raise ValueError("W must be a square matrix.")
            if N is not None and W.shape[0] != N:
                raise ValueError("W's shape must match the provided N.")
            self.N = W.shape[0]
            self.W = W
        else:
            self.N = N
            self.W = np.random.rand(N, N) * 0.1  # Random weights if not provided
        
        self.tau_m = tau_m
        self.V_rest = V_rest
        self.V_th = V_th
        self.V_reset = V_reset
        self.refractory_period = refractory_period
        self.noise = noise
        
    def run(self, T=10e3, dt=1.0, external_input=None):
        """
        Run the LIF network simulation.
        :param T: Total simulation time (ms)
        :param dt: Time step (ms)
        :param external_input: External input configuration (dict)
        :return: Spikes and time array
        """
        time = np.arange(0, T, dt)
        V = np.full(self.N, self.V_rest)
        V_trace = np.zeros((len(time), self.N))
        spikes = np.zeros((len(time), self.N))
        refractory_time = np.zeros(self.N)
        self.external_input = external_input

        # Handle external input
        if external_input is not None:
            input_type = external_input.get("type")
            input_targets = external_input.get("targets", [])
            if input_type == "logical":
                input_pattern = external_input.get("pattern", np.zeros(len(time), dtype=bool))
                input_strength = external_input.get("strength", 0.0)
            elif input_type == "time_varying":
                input_pattern = external_input.get("values", np.zeros(len(time)))
            elif input_type == "full_matrix":
                input_pattern = external_input.get("matrix", np.zeros((len(input_targets), len(time))))
            else:
                raise ValueError("Invalid external input type.")
        
        for t_idx, t in enumerate(time):
            active_neurons = refractory_time <= 0
            I_syn = np.dot(self.W, spikes[t_idx - 1] if t_idx > 0 else np.zeros(self.N))
            
            if external_input is not None:
                if input_type == "logical" and input_pattern[t_idx]:
                    I_syn[input_targets] += input_strength
                elif input_type == "time_varying":
                    I_syn[input_targets] += input_pattern[t_idx]
                elif input_type == "full_matrix":
                    I_syn[input_targets] += input_pattern[:, t_idx]
            
            noise_term = np.random.normal(0, self.noise, self.N)
            dV = (-(V - self.V_rest) + I_syn + noise_term) / self.tau_m
            V[active_neurons] += dV[active_neurons] * dt
            spiking_neurons = V >= self.V_th
            spikes[t_idx, spiking_neurons] = 1
            V[spiking_neurons] = self.V_reset
            V_trace[t_idx] = V
            refractory_time[spiking_neurons] = self.refractory_period
            refractory_time -= dt
        
        self.spikes = spikes
        self.time = time
        self.V_trace = V_trace
        # return spikes, time
    
    def plot_raster(self):
        """
        Plot the raster plot of spikes.
        """
        plt.figure(figsize=(10, 5))
        for i in range(self.N):
            spike_times = self.time[self.spikes[:, i] == 1]
            plt.vlines(spike_times, i + 0.5, i + 1.5, color="black")
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron Index")
        plt.title("Spike Raster Plot")
        plt.ylim(0.5, self.N + 0.5)
        plt.xlim(0, self.time[-1])
        plt.show()
    
    def plot_trace(self, neurons=None):
        """
        Plot the membrane potential trace for specified neurons.
        """
        plt.figure(figsize=(10, 5))
        if neurons is None:
            neurons = range(self.N)
        for neuron in neurons:
            plt.plot(self.time, np.transpose(self.V_trace)[neuron], label=f'Neuron {neuron}')
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane Potential (mV)")
        plt.title("Membrane Potential Trace")
        plt.legend()
        plt.show()
    
    def plot_rate(self, neurons=None, window=100):
        """
        Plot the population firing rate for specified neurons.
        """
        plt.figure(figsize=(10, 5))
        if neurons is None:
            neurons = range(self.N)
        spike_counts = np.sum(self.spikes[:, neurons], axis=1)
        rate = np.convolve(spike_counts, np.ones(window)/window, mode='same')
        plt.plot(self.time, rate)
        plt.xlabel("Time (ms)")
        plt.ylabel("Firing Rate (Hz)")
        plt.title("Population Firing Rate")
        plt.show()

    def plot_input(self):
        """
        Plot the external input applied to the network.
        """
        if self.external_input is None:
            print("No external input provided.")
            return
        
        input_type = self.external_input.get("type")
        plt.figure(figsize=(10, 5))
        if input_type == "logical":
            input_pattern = self.external_input.get("pattern", np.zeros(len(self.time), dtype=bool))
            input_strength = self.external_input.get("strength", 0.0)
            plt.plot(self.time, input_pattern * input_strength, label="Logical Input")
            plt.title("Logical External Input")
        elif input_type == "time_varying":
            input_pattern = self.external_input.get("values", np.zeros(len(self.time)))
            plt.plot(self.time, input_pattern, label="Time-Varying Input")
            plt.title("Time-Varying External Input")
        elif input_type == "full_matrix":
            input_pattern = self.external_input.get("matrix", np.zeros((len(self.external_input.get("targets", [])), len(self.time))))
            plt.imshow(input_pattern, aspect='auto', cmap='viridis', extent=[0, self.time[-1], 0, input_pattern.shape[0]])
            plt.colorbar(label="Input Strength")
            plt.ylabel("Neuron Index")
            plt.xlabel("Time (ms)")
            plt.title("Full Matrix External Input")
        plt.legend()
        plt.show()


# Example Usage
if __name__ == "__main__":
    network = SNN(N=10)
    network.run(T=500, dt=1.0)
    network.plot_raster()
    network.plot_trace([0, 1, 2])
    network.plot_rate(window=20)
    network.plot_input()
