import os
import tkinter as tk
import cProfile
import pstats

from application_interface import ImageApp

##-----------------------------------------------------------------------------------------
##                        Main Script
##-----------------------------------------------------------------------------------------

def profile_image_composition():
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()

if __name__ == '__main__':
    cProfile.run("profile_image_composition()", "image_composition_profile")
    
    # Display profiling results
    stats = pstats.Stats("image_composition_profile")
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats()

