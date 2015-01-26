import matplotlib
matplotlib.use('TkAgg')

import matplotlib as mpl

import numpy
from scipy.stats import threshold
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

import tkinter as tk
import tkinter.ttk as ttk

root = tk.Tk()
root.wm_title("PyEvt Viewer")

master_frame = ttk.Frame(master=root, borderwidth=1)
master_frame.pack(fill='both', expand=True)

toolbar = ttk.Frame(master=master_frame)
toolbar.pack()

open_button = ttk.Button(master=toolbar, text='Open')
open_button.pack(side='left')

prev_button = ttk.Button(master=toolbar, text="Previous")
prev_button.pack(side='left')

current_evt = ttk.Entry(master=toolbar, width=4)
current_evt.pack(side='left')

next_button = ttk.Button(master=toolbar, text="Next")
next_button.pack(side='left')

f1 = mpl.figure.Figure()
ax = f1.add_subplot(111)
ax.plot(numpy.sin(numpy.linspace(0, 4, 100)))
# a tk.DrawingArea
padcanvas = FigureCanvasTkAgg(f1, master=root)
padcanvas.show()
padcanvas.get_tk_widget().pack()

toolbar = NavigationToolbar2TkAgg(padcanvas, root)
toolbar.update()
padcanvas._tkcanvas.pack(side='top', fill='both', expand=1)

tk.mainloop()