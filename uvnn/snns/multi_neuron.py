import numpy as np

time_step = 1 # simulation step second
simulation_time = 1000 # seconds
D = 2 # potential degradation value

p_max = 10
p_min = -10
p_rest = 0
a_plus = 2
a_minus = 2
tau_plus = 3
tau_minus = 3


t_refr = 10

# simple snn with 1 membrane:

n_input = 4
spike_freq = [10, 15, 35, 40]
n_output = 2
windowsize = 5
margin = 2
sigma = 0.4
w_max = 2
w_min = -1

def simulate(time_step, simulation_time, D, p_max, p_min, p_rest, t_refr, 
        n_input, spike_freq, n_output, tau_plus, tau_minus, a_plus, a_minus):

    traces_all = [[p_rest] for _ in range(n_output)] # membranes potentials
    
    out_spikes_all = [set() for _ in range(n_output)] 
    # we keep the times when each neuron fires spikes 

    # initialize weights
    weights = np.ones((n_input, n_output))


    for time in range(0, simulation_time, time_step):
        for post_neuron in range(n_output):
            p_cur = traces_all[post_neuron][-1] #current potentials
            out_spikes = out_spikes_all[post_neuron]
            trace = traces_all[post_neuron]
        

            if len(out_spikes) > 0 and time - out_spikes[-1] < t_refr:
                # don't do anything
                trace.append(p_cur)
                continue

            if p_cur > p_max:
                # Fire, Spike happened in post synaptic
                out_spikes.append(time)
                # update weights
                
                # calculate dw for presynaptic spikes
                for pre_time in range(time - windowsize, time - margin):
                    delta_time = time - pre_time
                    for pre_neuron in range(n_input):
                        if pre_time % spike_freq[pre_neuron] == 0:
                            dw = a_plus * np.exp(delta_time / float(tau_plus))

                            # update weights
                            w_old = weights[pre_neuron][post_neuron]
                            weights[pre_neuron][post_neuron] += (sigma * dw * 
                                    (w_max - w_old))


                p_cur = p_rest
                trace.append(p_cur)
                continue

            if p_cur < p_min:
                p_cur = p_rest
                trace.append(p_cur)
                continue

            # increase the potential, incoming spikes
            for pre_neuron in range(n_input):
                if time % spike_freq[pre_neuron] == 0: # spike happens
                    p_cur += weights[pre_neuron][post_neuron]

                    # spike heppened in pre_synaptic neuron
                    # now update weights considering the postsynaptic 
                    # spikes which happened before this (negative dependency)
                    
                    # Let's just check the last firing times of post synaptic


            # now substract D for this time instant
            if p_cur > p_rest:
                p_cur -= D

            trace.append(p_cur)
    
    return traces_all, out_spikes_all
