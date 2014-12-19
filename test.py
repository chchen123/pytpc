import evtdata
import tpcplot
import matplotlib.pyplot as plt
import scipy.stats as ss

def test():
    ef = evtdata.EventFile("/Users/josh/Documents/Data/run_0128.evt")
    peds = evtdata.load_pedestals("/Users/josh/Dropbox/routing/Peds20141208-2.csv")
    pm = evtdata.load_padmap("/Users/josh/Dropbox/routing/Lookup20141208.csv")
    evt = ef.read_event_by_number(11)
    evt.traces = ss.threshold(evt.traces - peds, threshmin=10)
    hits = evt.hits(pm)
    tpcplot.pad_plot(hits)