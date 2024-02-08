import os
import re
import tkinter as tk
from PIL import ImageTk, Image
from .image_composition import composite_smoke, select_background_image, select_smoke_image
from .image_harmonization import harmonize_smoke_with_background
##-----------------------------------------------------------------------------------------
##                        Main App for Dataset Generation
##-----------------------------------------------------------------------------------------

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Composition App")

        self.image_width = 700
        self.image_height = 700
        
        self.base_folder = 'smoke_generator_V1'
        self.output_folder = 'smoke_dataset_V1'
        self.images_output = os.path.join(self.output_folder, 'images/')
        self.mask_output = os.path.join(self.output_folder, 'masks/')

        self.new_image_index = 0
        self.initialize_new_image_index()  # Initialize new_image_index based on existing images


        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack()

        self.cmd_frame = tk.Frame(self.root)
        self.cmd_frame.pack()

        self.img1 = None
        self.image1 = None
        self.mask = None
        self.harmonized_image = None
        self.harmonized_label = None

        self.create_buttons()

    def create_buttons(self):
        tk.Button(self.cmd_frame, text="Harmonize", command=self.perform_harmonization).pack(side="left", padx=10, pady=10)
        tk.Button(self.cmd_frame, text="Save Image 1", command=self.save_image).pack(side="left", padx=10, pady=10)
        tk.Button(self.cmd_frame, text="Continue", command=self.display_images).pack(side="left", padx=10, pady=10)
        tk.Button(self.cmd_frame, text="Quit", command=self.root.quit).pack(side="left", padx=10, pady=10)

    def clear_harmonized(self):
        if self.harmonized_label:
            self.harmonized_label.destroy()
        if hasattr(self, 'save_harmonized_button'):
            self.save_harmonized_button.destroy()
            
    def display_images(self):
        self.clear_harmonized()
        self.background_path = select_background_image(self.base_folder)
        self.smoke_image_path = select_smoke_image(self.base_folder)
        self.image1, self.mask = composite_smoke(self.background_path, self.smoke_image_path)
        image1 = self.image1.resize((self.image_width, self.image_height), Image.LANCZOS)
        self.img1 = ImageTk.PhotoImage(image=image1)
        if self.img1:
            if hasattr(self, 'label1'):
                self.label1.destroy()
            self.label1 = tk.Label(self.image_frame, image=self.img1)
            self.label1.grid(row=0, column=0, padx=10, pady=10)

    def perform_harmonization(self):
        model_path = r'smoke_generator_V1\scripts\harmonization_scripts\model_path\rascv2.pth.tar'
        if self.image1 is None or self.mask is None:
            return
        
        self.harmonized_image = harmonize_smoke_with_background(self.image1, self.mask, model_path)
        harmonized_image_display = self.harmonized_image.resize((self.image_width, self.image_height), Image.LANCZOS)

        if self.harmonized_label:
            self.harmonized_label.destroy()

        self.harmonized_img = ImageTk.PhotoImage(image=harmonized_image_display)
        self.harmonized_label = tk.Label(self.image_frame, image=self.harmonized_img)
        self.harmonized_label.grid(row=0, column=1, padx=10, pady=10)

        if hasattr(self, 'save_harmonized_button'):
            self.save_harmonized_button.destroy()
        if self.harmonized_image:  # Only create the button if a harmonized image is generated
            self.save_harmonized_button = tk.Button(self.cmd_frame, text="Save Harmonized", command=self.save_harmonized)
            self.save_harmonized_button.pack(side="left", padx=10, pady=10)

    def save_image(self):
        if self.image1 and self.mask:
            self.image1.save(os.path.join(self.images_output, f'image_{self.new_image_index}.png'))
            self.mask.save(os.path.join(self.mask_output, f'mask_{self.new_image_index}.png'))
            self.new_image_index += 1
            self.clear_harmonized()
            self.display_images()

    def save_harmonized(self):
        if self.harmonized_image:
            self.harmonized_image.save(os.path.join(self.images_output, f'image_{self.new_image_index}.png'))
            self.mask.save(os.path.join(self.mask_output, f'mask_{self.new_image_index}.png'))
            self.new_image_index += 1
            self.save_harmonized_button.destroy()
            self.clear_harmonized()
            self.display_images()

    def initialize_new_image_index(self):
        existing_image_files = [f for f in os.listdir(self.images_output) if f.startswith("image_")]
        existing_mask_files = [f for f in os.listdir(self.mask_output) if f.startswith("mask_")]
        
        image_indices = set()
        for file_name in existing_image_files + existing_mask_files:
            match = re.match(r"image_(\d+)\.png", file_name) or re.match(r"mask_(\d+)\.png", file_name)
            if match:
                image_indices.add(int(match.group(1)))

        if image_indices:
            self.new_image_index = max(image_indices) + 1
        else:
            self.new_image_index = 0
