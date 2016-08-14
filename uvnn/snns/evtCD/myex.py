
#import initExample

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import collections
from common import Spike

class DashBoard(object):
    def __init__(self, history):
        # history is an array of (time, data) see history definition in evtcd.py
        self.w_row = 10   # weights are shown in 10 x 10 grid on picture
        self.w_col = 10   # NOTE TODO(refactor) change it if you change dims 
        
        self.history = history
        self.layer_sizes = [(20, 20), (10, 10), (20, 20), (10, 10)]
        self.show_lastn_spike = 100
        self.last_spikes = [collections.deque(
            maxlen=self.show_lastn_spike) for _ in range(5)]
        self.ind = 0 # index in the history array

    def plot_thigns(self):
        app = QtGui.QApplication([])
        
        view = pg.GraphicsView()
        pg.setConfigOptions(antialias=True)
        l = pg.GraphicsLayout(border=(100, 100, 100))
        view.setCentralItem(l)
        view.setWindowTitle('rame')
        view.resize(600, 500)

        view_weights = pg.GraphicsView()
        l_weights_main = pg.GraphicsLayout(border=(100, 100, 100))
        view_weights.setCentralItem(l_weights_main)
        view_weights.show()
        view_weights.setWindowTitle('weights')
        view_weights.resize(400, 400)


        view.show()   #  main view
        # weights layout 
        l_weights = l_weights_main.addLayout(self.w_row, self.w_col)
        
        text = """
        In the spike images row, first image is the input spike train <br>
        Then comes the hidden layer, then visible and hidden again
        """

        l.addLabel(text, col=0, colspan=3)
        time_lab = l.addLabel('rama', color=(255,0,0), size='20pt')
        l.nextRow()
        
        # start spike train image
        ltrain = l.addLayout(colspan=4, border=(10, 0, 0))
        ltrain.setContentsMargins(5, 5, 5, 5)
        #ltrain.addLabel('current Spike train:')
        vb = ltrain.addViewBox(lockAspect=True)
        input_train_img = pg.ImageItem(border='w')
        self.input_train_img = input_train_img
        vb.addItem(input_train_img)
        lr_size = self.layer_sizes[0]
        vb.setRange(QtCore.QRectF(0, 0, lr_size[0], lr_size[1]))
        l.nextRow()
        # end spike train image

        # start spike images 
        lspikes = l.addLayout(colspan=4, border=(10, 0, 0))
        lspikes.setContentsMargins(5, 5, 5, 5)
        
        self.spike_imgs = []
        self.spike_datas = []
        for lr_size in self.layer_sizes:
            vb = lspikes.addViewBox(lockAspect=True)
            img = pg.ImageItem(border='w')
            self.spike_imgs.append(img)
            vb.addItem(img)
            vb.setRange(QtCore.QRectF(0, 0, lr_size[0], lr_size[1]))
            self.spike_datas.append(np.zeros(lr_size))
        # end spike images
        
        l.nextRow()
        
        # layer membrane potentials

        lmembranes = l.addLayout(colspan=4, border=(10, 0, 0))
        lmembranes.setContentsMargins(5, 5, 5, 5)


        
        self.membr_imgs = []
        self.membr_datas = []
        for lr_size in self.layer_sizes:
            vb = lmembranes.addViewBox(lockAspect=True)
            img = pg.ImageItem(border='w')
            self.membr_imgs.append(img)
            vb.addItem(img)
            vb.setRange(QtCore.QRectF(0, 0, lr_size[0], lr_size[1]))
            self.membr_datas.append(np.zeros(lr_size))

        # end membrane potentials

        ############# layer for spike imags
        #p1 = l.addPlot(title="Membrane potential1", y=np.random.normal(size=100))
        #p2 = l.addPlot(title='Membrane potential2')
        #p2.plot(y=np.random.normal(size=80))
    
        #vb = l.addViewBox(lockAspect=True)
        #img = pg.ImageItem(border='w')
        #vb.addItem(img)
        #vb.setRange(QtCore.QRectF(0, 0, 100, 100))

        #
        ## Create random image
        #self.img = img
        self.time_lab = time_lab
        self.last_upd_time = 0
        self._update_plots()


        # end image view

        import sys
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def update_membrane_plot(self, layer, layer_vals):
        self.membr_datas[layer] = layer_vals.reshape(self.layer_sizes[layer])
        self.membr_imgs[layer].setImage(self.membr_datas[layer])

    def update_train_plot(self, vals):
        layer_sz = self.layer_sizes[0]
        self.input_train_img.setImage(vals.reshape(layer_sz))

    def update_spike_plot(self, triplet):
        trip_layer = triplet.layer # original layers were(-1, 0, 1, 2, 3)
        
        # now before adding newspike to correspoding layer trace
        # first delete the last spike on the image
    
        if len(self.last_spikes[trip_layer]) == self.show_lastn_spike:
            oldest_address = self.last_spikes[trip_layer][0].address
            x, y = self.getxy(oldest_address, trip_layer)
            self.spike_datas[trip_layer][x][y] = 0

        # fade everything
        self.spike_datas[trip_layer] *= 0.95
        
        self.last_spikes[trip_layer].append(triplet)
        # turn on the latest spike
        x, y = self.getxy(triplet.address, trip_layer)
        self.spike_datas[trip_layer][x][y] = 1

        # display updated image
        self.spike_imgs[trip_layer].setImage(self.spike_datas[trip_layer])


    
    def _update_plots(self):
        #import ipdb; ipdb.set_trace()
        
        # spike img
        #for img in self.spike_imgs:
        #    img.setImage(self.data[self.i])
       
        cur_time, events = self.history[self.ind]
        self.time_lab.setText(('%2.3fs'%cur_time))
        for event in events:
            if event[0] == 'SPIKE': # spike occured update plots 
                triplet = Spike(time=cur_time, layer=event[1], address=event[2])
                self.update_spike_plot(triplet)
            elif event[0] == 'MEMBRANE':
                layer, layer_vals = event[1], event[2]
                self.update_membrane_plot(layer, layer_vals)
            elif event[0] == 'NEW_SPIKE_TRAIN':
                vals = event[1]
                self.update_train_plot(vals)

        self.ind += 1
        relative_sim = False # relative time simulation
        if relative_sim:
            delay = 100 * (cur_time - self.last_upd_time)
        else:
            delay = 10 

        if self.ind < len(self.history):
            QtCore.QTimer.singleShot(delay, self._update_plots)
        self.last_upd_time = cur_time
            
    def getxy(self, address, layer):
        # convert 1d neuron coordinate to 2d
        return (address / self.layer_sizes[layer][0], 
                address % self.layer_sizes[layer][1])

#db = DashBoard(1)
#db.plot_thigns()
#db.start()
