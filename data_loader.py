import numpy as np
from sys import exc_info
from numpy import save, load
# params = [sys.argv[1], 10, 20,1,True,'chords',1,"adam"]

class DataLoader(object):
    def __init__(self, datadir, validation_percentage, test_percentage,filename="",
                works_per_composer=None, pace_events=False, synthetic=None, tones_per_cell=1, 
                single_composer=None, n_samples=None, n_features=1, n_steps=1):        
        self.pointer = {}
        self.datadir = datadir
        self.pointer['validation'] = 0
        self.pointer['test'] = 0
        self.pointer['train'] = 0

        self.numsamples = n_samples
        self.num_features = n_features
        self.num_steps = n_steps


        if synthetic == 'chords':
            self.generate_chords(pace_events=pace_events)

        self.read_data(filename,validation_percentage,test_percentage)

    def read_data(self,filename,val_percentage=None,test_percentage=None):
        # self.data = np.random.randn(100,48,1)

        self.data = load(self.datadir+filename)
        # data is not in correct shape
        valid_shape = (self.numsamples,self.num_steps,self.num_features)
        if self.data.shape!= valid_shape:
            print("Reshaping...")
            try:
                self.data = self.data.reshape(valid_shape)
            except:                
                print("The data with shape {} couldn't be reshaped to {}, provide valid".format(self.data.shape, valid_shape))
                exit()        

        self.songs = {}
        self.songs['validation'] = []
        self.songs['test'] = []
        self.songs['train']  = []
        
        train_len = len(self.data)
        test_len = 0
        validation_len = 0
        
        # TODO criar excpetions para essa parte
        if val_percentage or test_percentage:
            if val_percentage:
                validation_len = int(float(val_percentage/100)*len(self.data))
                train_len = train_len - validation_len
            if test_percentage:
                test_len = int(float(test_percentage/100)*len(self.data))
                train_len = train_len - test_len
            self.songs['train'] = self.data[:train_len]
            self.songs['test'] = self.data[train_len:train_len+test_len]
            self.songs['validation'] = self.data[train_len+test_len:]

        else:
            self.songs['train'] = self.data
            self.songs['test']  = self.data
            self.songs['validation'] = self.data
        

        # pointers
        self.pointer['validation']  = 0
        self.pointer['test'] = 0
        self.pointer['train'] = 0

    # TODO remove unused parameters
    def get_batch(self,batch_size,part='train'):
        """
        Returns a batch with the dimension [batch_size,self.num_steps,self.num_features]
        """

        if (self.pointer[part]>len(self.songs[part])-batch_size):
            return [None,None]

        if len(self.songs[part])>0:

            start = self.pointer[part]
            end = self.pointer[part]+batch_size

            batch = self.songs[part][start:end]
            meta = np.random.randn(batch_size,1)
            self.pointer[part]+=batch_size        
            return [meta,batch]

        else:
            raise 'get_batch() called but self.songs is not initialized.'

    def test_sizes(self):

        print("data size: ",len(self.data))
        print("train:", len(self.songs['train']))
        print("test:", len(self.songs['test']))
        print("validation:", len(self.songs['validation']))
    
    def generate_chords(self,):
        pass
    
    
    def get_num_song_features(self):
        return self.num_features

    def get_num_meta_features(self):
        # just for test purposes
        return 1
    
    def rewind(self,part='train'):
        """
        Reset the pointer for the 'part'
        (default is 'train')
        """
        self.pointer[part] = 0

    def get_midi_pattern(self,array):
        """
        Don't need to be implemented
        """
        pass
    def save_midi_pattern(self,filename, array):
        """
        Saves the generated data
        """
        if filename is not None:
            try:
                save(filename,array)                    
            except:
                print("ERROR: {} ocurred".format(exc_info()[0]))

    def save_data(self,filename,data):
        pass








