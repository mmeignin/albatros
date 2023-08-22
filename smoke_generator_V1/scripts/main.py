import tkinter as tk
import cProfile
import pstats


from utils import ImageApp

##-----------------------------------------------------------------------------------------
##                        Main Script
##-----------------------------------------------------------------------------------------

def profile_image_composition():
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()

if __name__ == '__main__':
    cProfile.run("profile_image_composition()", "image_composition_profile.prof")
    
    # Display profiling results
    stats = pstats.Stats("image_composition_profile.prof")
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats()

