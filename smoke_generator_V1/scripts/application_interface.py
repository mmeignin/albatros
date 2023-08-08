import os
import re
import tkinter as tk
from PIL import ImageTk, Image
from image_composer import composite_smoke_with_rotation, select_background_image, select_smoke_image

##-----------------------------------------------------------------------------------------
##                        Main App for Dataset Generation
##-----------------------------------------------------------------------------------------

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Composition App")

        self.image_width = 540  # Adjust this width as needed
        self.image_height = 540  # Adjust this height as needed
        
        self.base_folder = 'D:/mploi/Documents/Albatros/albatros/smoke_generator_V1'
        self.output_folder = 'D:/mploi/Documents/Albatros/albatros/smoke_dataset_V1'
        self.images_output = os.path.join(self.output_folder, 'images/')
        self.mask_output = os.path.join(self.output_folder, 'masks/')
        if not os.path.exists(self.images_output):
            os.mkdir(self.images_output)
            print(f"Output directory '{self.images_output}' created.")
        if not os.path.exists(self.mask_output):
            os.mkdir(self.mask_output)
            print(f"Output directory '{self.mask_output}' created.")

        self.background_path = select_background_image(self.base_folder)
        if not self.background_path:
            return

        self.smoke_image_path = select_smoke_image(self.base_folder)
        if not self.smoke_image_path:
            return

        self.new_image_index = self.find_last_image_index(self.images_output) + 1

        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack()

        self.cmd_frame = tk.Frame(self.root)
        self.cmd_frame.pack()

        self.img1 = None
        self.img2 = None

        self.display_images()

    def find_last_image_index(self, output_dir):
        files = os.listdir(output_dir)
        numeric_parts = [re.findall(r'\d+', f) for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        numeric_parts = [int(part) for sublist in numeric_parts for part in sublist]
        last_index = max(numeric_parts) if numeric_parts else -1
        return last_index

    def display_images(self):
        self.background_path = select_background_image(self.base_folder)
        self.smoke_image_path = select_smoke_image(self.base_folder)
        image1, image2, mask = composite_smoke_with_rotation(self.base_folder, self.background_path, self.smoke_image_path, max_rotation_angle=360, brightness_range=(0.8, 1))
        
        image1 = image1.resize((self.image_width, self.image_height), Image.ANTIALIAS)
        image2 = image2.resize((self.image_width, self.image_height), Image.ANTIALIAS)

        self.img1 = ImageTk.PhotoImage(image=image1)
        self.img2 = ImageTk.PhotoImage(image=image2)

        if hasattr(self, 'label1'):
            self.label1.destroy()
        if hasattr(self, 'label2'):
            self.label2.destroy()

        self.label1 = tk.Label(self.image_frame, image=self.img1)
        self.label1.grid(row=0, column=0, padx=10, pady=10)
        self.label2 = tk.Label(self.image_frame, image=self.img2)
        self.label2.grid(row=0, column=1, padx=10, pady=10)

        if hasattr(self, 'save_button1'):
            self.save_button1.destroy()
        if hasattr(self, 'save_button2'):
            self.save_button2.destroy()
        if hasattr(self, 'continue_button'):
            self.continue_button.destroy()
        if hasattr(self, 'quit_button'):
            self.quit_button.destroy()

        self.save_button1 = tk.Button(self.cmd_frame, text="Save Image 1", command=lambda: self.save_image(image1,mask))
        self.save_button1.pack(side="left", padx=10, pady=10)
        self.save_button2 = tk.Button(self.cmd_frame, text="Save Image 2", command=lambda: self.save_image(image2,mask))
        self.save_button2.pack(side="left", padx=10, pady=10)
        self.continue_button = tk.Button(self.cmd_frame, text="Continue", command=self.display_images)
        self.continue_button.pack(side="left", padx=10, pady=10)
        self.quit_button = tk.Button(self.cmd_frame, text="Quit", command=self.root.quit)
        self.quit_button.pack(side="left", padx=10, pady=10)

    def save_image(self, image,mask):
        image.save(os.path.join(self.images_output, f'image_{self.new_image_index}.png'))
        mask.save(os.path.join(self.mask_output, f'mask_{self.new_image_index}.png'))
        self.new_image_index += 1
        self.display_images()  
