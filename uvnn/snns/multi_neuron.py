import numpy as np

conf = {'time_step': 2, # simulation step second
       'simulation_time':300, # seconds
        'D':1.5, # potential degradation value
        'p_max':0.5,
        'p_min':-10,
        'p_rest':0,
        't_refr':9,
        'n_input':4,
        'n_output':2,
        'spike_periods':[1, 20, 40, 60],
        'w_max':2,
        'w_min':-1,
        'sigma':0.4,
        'margin':2,
        'windowsize':5,
        'tau_plus':3,
        'tau_minus':3,
        'a_plus':2,
        'a_minus':-2,
       }

def get_presynaptics(time, spike_periods):
    ''' Get which neurons are firing on provided time

    Right now we feed spike_freq but it can be array of spikes trains later

    Args: 
        time: current clock cycle
        spike_periods: numpy array of preiod of each spikes
    Return:
        bool array of currently active active neurons
    '''
    active = []
    for i, period in enumerate(spike_periods):
        if time % period == 0:
            active.append(i)

    return active


def simulate(time_step, simulation_time, D, p_max, p_min, p_rest, t_refr, 
        n_input, spike_periods, n_output, tau_plus, tau_minus, a_plus, a_minus,
        w_max, w_min, windowsize, margin, sigma):
        
    init_potentials = np.zeros(n_output)
    init_potentials.fill(p_rest)

    traces = [init_potentials] # membranes potentials
    
    # all output spikes for each neuron
    out_spikes = [set() for _ in range(n_output)]
    
    # we keep the times when each neuron fires spikes 
    out_spikes_last = np.zeros(n_output) 
    out_spikes_last.fill(-10000)
    
    weights = np.random.rand(n_input, n_output) # initialize weights
    
    spike_periods = np.array(spike_periods) # just convert to numpy array


    for time in range(0, simulation_time, time_step): 
        if time % 10 == 0:
            print weights
        #import ipdb; ipdb.set_trace()
        cur_potentials = traces[-1]
        new_potentials = np.copy(cur_potentials)

        # fire neurons which are above threshold
        fired = cur_potentials > p_max
        for post_neuron in np.nonzero(fired)[0]:
            out_spikes[post_neuron].add(time)
            out_spikes_last[post_neuron] = time
        new_potentials[fired] = p_rest

        # update weights for presynaptic neurons (positively correlated)
        already_updated = set() # keep already updated neurons not to update twice
        for pre_time in range(time - windowsize, time - margin):
            delta_time = time - pre_time
            pre_neurons = get_presynaptics(pre_time, spike_periods)
            # distract neurons already updated
            pre_neurons = list(set(pre_neurons) - already_updated)
            dw = a_plus * np.exp(delta_time / float(tau_plus))
            weights[pre_neurons] += sigma * dw * (w_max - weights[pre_neurons])
            already_updated = already_updated | set(pre_neurons) # unite

        
        # depolarize strongly negative neurons
        strongly_negative = cur_potentials < p_max
        new_potentials[strongly_negative] = p_rest

            
        #get postsynaptic neurons which  are not in refractory period
        # and add weights from presynaptic
        post_neurons = out_spikes_last < time - t_refr
        
        # get presynaptic neurons which fires now
        pre_neurons = get_presynaptics(time, spike_periods)

        for post_neuron in np.nonzero(post_neurons)[0]:
            for pre_neuron in pre_neurons:
                new_potentials[post_neuron] += weights[pre_neuron][post_neuron]
            new_potentials[post_neuron] -= D # decay
        
        # now update wieghts for postsyanptic neurons (negatively correlated)
        alread_updated = set()
        for pre_time in range(time - windowsize, time - margin):
            for post_neuron in range(n_output):
                # check if already updated 
                if post_neuron in alread_updated:
                    continue
                # check if post_neuron was fired before pre_neuron
                if pre_time in out_spikes[post_neuron]:
                    delta_time = time - pre_time
                    dw = a_minus * np.exp(delta_time / float(tau_minus))
                    #import ipdb; ipdb.set_trace()
                    weights[pre_neurons, post_neuron] += (sigma * dw * 
                            (weights[pre_neurons, post_neuron]-w_min))
                already_updated.add(post_neuron)

        traces.append(new_potentials)  # keep potential trace

    return traces, out_spikes


#simulate(**conf)
