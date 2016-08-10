
#import initExample

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import collections

class DashBoard(object):
    def __init__(self, spike_history):
        # spike history is the named tuple form of -(time, address, layer)
        self.spike_history = spike_history
        self.layer_sizes = [(20, 20), (10, 10), (20, 20), (10, 10)]
        self.show_lastn_spike = 15
        self.last_spikes = [collections.deque(
            maxlen=self.show_lastn_spike) for _ in range(5)]
        self.cur_spike = 0

    def plot_thigns(self):
        app = QtGui.QApplication([])
        
        view = pg.GraphicsView()
        pg.setConfigOptions(antialias=True)
        l = pg.GraphicsLayout(border=(100, 100, 100))
        view.setCentralItem(l)
        view.show()
        view.setWindowTitle('rame')
        view.resize(1000, 500)

        
        text = """
        In the spike images row, first image is the input spike train <br>
        Then comes the first to fifth layers, alteranting layer sizes
        """

        l.addLabel(text, col=0, colspan=4)
        l.nextRow()
        
        # make layer for spike images 
        lspikes = l.addLayout(colspan=5, border=(20, 0, 0))
        lspikes.setContentsMargins(10, 10, 10, 10)
        
        self.spike_imgs = []
        self.spike_datas = []
        for lr_size in self.layer_sizes:
            vb = lspikes.addViewBox(lockAspect=True)
            img = pg.ImageItem(border='w')
            self.spike_imgs.append(img)
            vb.addItem(img)
            vb.setRange(QtCore.QRectF(0, 0, lr_size[0], lr_size[1]))
            self.spike_datas.append(np.zeros(lr_size))



        ############# layer for spike imags
        l.nextRow()
        p1 = l.addPlot(title="Membrane potential1", y=np.random.normal(size=100))
        p2 = l.addPlot(title='Membrane potential2')
        p2.plot(y=np.random.normal(size=80))
    
        # image view
        vb = l.addViewBox(lockAspect=True)
        img = pg.ImageItem(border='w')
        #win.add
        vb.addItem(img)
        vb.setRange(QtCore.QRectF(0, 0, 100, 100))
        #vb.autoRange()

        #
        ## Create random image
        data = np.random.normal(size=(15,100, 100), loc=1024, scale=64).astype(np.uint16)
        i = 0
        #updateTime = ptime.time()
        self.data = data
        self.i = i
        self.img = img
        self._update_spike_images()


        # end image view

        import sys
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

        
    def _update_spike_images(self):
        #import ipdb; ipdb.set_trace()
        self.img.setImage(self.data[self.i])
        
        self.i = (self.i + 1) % self.data.shape[0]
        
        # spike img
        #for img in self.spike_imgs:
        #    img.setImage(self.data[self.i])
        
        triplet = self.spike_history[self.cur_spike]
        trip_layer = triplet.layer # original layers were(-1, 0, 1, 2, 3)
        
        # now before adding newspike to correspoding layer trace
        # first delete the last spike on the image
       
        if len(self.last_spikes[trip_layer]) == self.show_lastn_spike:
            oldest_address = self.last_spikes[trip_layer][0].address
            x, y = self.getxy(oldest_address, trip_layer)
            self.spike_datas[trip_layer][x][y] = 0

        # fade everything
        self.spike_datas[trip_layer] *= 0.9
        
        self.last_spikes[trip_layer].append(triplet)
        # turn on the latest spike
        x, y = self.getxy(triplet.address, trip_layer)
        self.spike_datas[trip_layer][x][y] = 1

        # display updated image
        self.spike_imgs[trip_layer].setImage(self.spike_datas[trip_layer])

        self.cur_spike += 1
        if self.cur_spike < len(self.spike_history):
            QtCore.QTimer.singleShot(10, self._update_spike_images)
            
    def getxy(self, address, layer):
        # convert 1d neuron coordinate to 2d
        return (address % self.layer_sizes[layer][0], 
                address / self.layer_sizes[layer][1])

#db = DashBoard(1)
#db.plot_thigns()
#db.start()
