import numpy as np

def process_spike(spike_triplet):

    sp_time, layer, sp_address = spike_triplet
    sp_layer += 1

    # Reconstruct the imaginary first-layer action that resulted in this spike
    if layer == 1:
        last_spiked[0][sp_address] = sp_time - axon_delay
        if thr[0][sp_address] < min_thr:
            thr[0][sp_address] = min_thr
        else:
            thr[0][sp_address] -= eta * thresh_eta
        spike_count[0] += 1
        if calc_recons[0]:
            recon[0] *= np.exp(- (spike_triplet.time - last_recon[0]) / recon_tau)
            recon[0][sp_address] += recon_imp
            last_recon[0] = sp_time - axon_delay

        # update neurons
        
        #decay membrane
        membrances[layer] *= np.exp(-(sp_time - last_update[layer]) / tau)

        #add impulse
        if layer == 0:
            # TODO: how can layer be 0
            membranes[layer][sp_address] += inp_scale
        elif layer % 2 == 0:
            membranes[layer] += weights[:, sp_address] * (refrac_end[layer] < sp_time)
        else:
            membranes[layer] += weights[sp_address, :] * (refrac_end[layer] < sp_time)


        # add noise
        add_noise(membranes[layer], layer) 
    
        # update last_update
        last_update[layer] = sp_time

        # Add firings to queue
        newspikes = np.nonzero(membranes[layer] > thr[layer % 2])[0]

        for newspike in newspikes:
            # update counts

            spike_count[layer] += 1

            # update refrac end
            refrac_end[layer][newspike] = sp_time + t_refrac

            # reset firings
            membranes[layer][newspike] = 0

            # reset time for stdp
            last_spiked[layer][newspike] = sp_time

            # stdp threshold adjustment

            thr_direction = -1 if layer < 2 else 1
            wt_direction = 1 if layer < 2 else -1
            thr[layer % 2][newspike] += thr_direction * eta * tresh_eta

            # TODO ambigious
            thr[layer % 2]


            # STDP weight adjustment
            if (layer % 2 == 1):
                weights[:, newspike] += (last_spiked[layer - 1] > sp_time - stdp_lag) * (wt_direction * eta)

            # reconstruct the layer if desired
            if calc_recons[layer]:
                recon[layer] *= np.exp(- (sp_time - last_recon[layer]) / recon_tau)
                recon[layer][newspike] += recon_imp
                last_recon[layer] = sp_time
            
            # add spikes to the queue if not in the end layer
            if (layer != 3):
                # TODO amb rng.nextfloat()
                pq.add(sp_time +  2 * axon_delay * rng_float, layer, newspike)


