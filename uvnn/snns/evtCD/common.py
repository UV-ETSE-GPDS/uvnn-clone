from collections import namedtuple
Param = namedtuple('Parameters', 
        ['eta', 'thresh_eta', 'numspikes', 'timespan', 'tau', 'thr',
            'inp_scale', 't_refrac', 'stdp_lag', 'min_thr', 'plot_things', 
            'axon_delay'])
Spike = namedtuple('Spike',
        ['time', 'layer', 'address'])
