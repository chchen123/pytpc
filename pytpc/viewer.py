import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

import numpy
from scipy.stats import threshold
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

import tkinter as tk
import tkinter.ttk as ttk

root = tk.Tk()
root.wm_title("PyEvt Viewer Menu")

files_frame = ttk.Frame(master=root, borderwidth=1)
files_frame.grid(row=0, column=0)

file_label = ttk.Label(master=files_frame, text="File")
file_label.grid(row=0, column=0)

file_entry = ttk.Entry(master=files_frame)
file_entry.grid(row=0, column=1)

peds_label = ttk.Label(master=files_frame, text="Pedestals")
peds_label.grid(row=1, column=0)

peds_entry = ttk.Entry(master=files_frame)
peds_entry.grid(row=1, column=1)

lookup_label = ttk.Label(master=files_frame, text="Lookup")
lookup_label.grid(row=2, column=0)

lookup_entry = ttk.Entry(master=files_frame)
lookup_entry.grid(row=2, column=1)

prev_button = ttk.Button(master=root, text="Previous")
prev_button.grid(row=3, column=0)

current_evt = ttk.Entry(master=root)
current_evt.grid(row=3, column=1)

next_button = ttk.Button(master=root, text="Next")
next_button.grid(row=3, column=2)


# file = evtdata.EventFile('/Users/josh/Documents/Data/alphas/run_0215.evt')
# evt = file.read_event_by_number(2)
#
# peds = evtdata.load_pedestals('/Users/josh/Dropbox/routing/Peds20141208-2.csv')
# pm = evtdata.load_padmap('/Users/josh/Dropbox/routing/Lookup20141208.csv')
#
# evt.traces = threshold(evt.traces - peds, threshmin=50)
#
# padfig = tpcplot.pad_plot(evt.hits(pm))
# padfig.set_size_inches(2, 2)
# chfig = tpcplot.chamber_plot(evt.xyzs(pm))

# f1, a1 = plt.subplots(2, 1)
# a = plt.subplot(211)
# t = arange(0.0, 3.0, 0.01)
# s = sin(2*pi*t)
# a2 = plt.subplot(212)
# c = cos(2*pi*t)

# a.plot(t, s)
# a2.plot(t, c)


# # a tk.DrawingArea
# padcanvas = FigureCanvasTkAgg(padfig, master=root)
# padcanvas.show()
# padcanvas.get_tk_widget().grid(row=0, column=0)
#
# chcanvas = FigureCanvasTkAgg(chfig, master=root)
# chcanvas.show()
# chcanvas.get_tk_widget().grid(row=0, column=1)

# w2 = Tk.Toplevel()

tk.mainloop()