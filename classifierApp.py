import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

class SatelliteImageClassificationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Defense Terrain Intelligence through CNN-Based Satellite Imagery Classification")
        self.state('zoomed')  # Maximize window to full screen

        self.model = None
        self.model_path = tk.StringVar(value="No model loaded")

        self.init_ui()

    def init_ui(self):
        self.label_title = tk.Label(self, text="Defense Terrain Intelligence through CNN-Based Satellite Imagery Classification", font=("Helvetica", 20))
        self.label_title.pack(pady=20)

        self.frame_buttons = tk.Frame(self)
        self.frame_buttons.pack(pady=10)

        self.btn_select_model = tk.Button(self.frame_buttons, text="Select Model", command=self.load_model, font=("Helvetica", 14), bg="black", fg="white")
        self.btn_select_model.pack(side=tk.LEFT, padx=5)

        self.btn_select_image = tk.Button(self.frame_buttons, text="Select Image", command=self.load_image, font=("Helvetica", 14), bg="black", fg="white", state=tk.DISABLED)
        self.btn_select_image.pack(side=tk.LEFT, padx=5)

        self.label_model = tk.Label(self, textvariable=self.model_path, font=("Helvetica", 14))
        self.label_model.pack(pady=10)

        self.frame = tk.Frame(self)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_original = tk.Canvas(self.frame, bg=self.cget("bg"))
        self.canvas_original.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas_colored = tk.Canvas(self.frame, bg=self.cget("bg"))
        self.canvas_colored.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.legend_frame = tk.Frame(self, bg=self.cget("bg"))
        self.legend_frame.pack(fill=tk.X, pady=10)

        self.image_path = None
        self.original_image = None
        self.colored_image = None

    def load_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("Keras model files", "*.keras;*.h5")])
        if model_path:
            self.model = load_model(model_path)
            self.model_path.set(f"Current model: {model_path}")
            self.btn_select_image.config(state=tk.NORMAL)

            # If an image is already selected, process it with the new model
            if self.image_path:
                self.process_image(self.image_path)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if self.image_path:
            self.process_image(self.image_path)

    def process_image(self, image_path):
        grid_size = 64
        class_colors = {
            0: [144, 238, 144],  # Light Green for AnnualCrop
            1: [34, 139, 34],    # Dark Green for Forest
            2: [173, 255, 47],   # Yellow-Green for HerbaceousVegetation
            3: [169, 169, 169],  # Gray for Highway
            4: [211, 211, 211],  # Light Gray for Industrial
            5: [255, 255, 224],  # Light Yellow for Pasture
            6: [255, 165, 0],    # Light Orange for PermanentCrop
            7: [255, 69, 0],     # Dark Orange for Residential
            8: [0, 0, 255],      # Blue for River
            9: [135, 206, 235]   # Light Blue for SeaLake
        }
        class_names = [
            'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
            'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
        ]

        # Load and preprocess the real image
        real_image = load_img(image_path)
        real_image = img_to_array(real_image)
        real_image = real_image / 255.0  # Normalize pixel values

        # Divide the resized image into grids
        grids = self.divide_image_into_grids(real_image, grid_size)

        # Predict the class of each grid
        predictions = self.model.predict(grids)
        predicted_classes = np.argmax(predictions, axis=1)

        # Colorize the grids according to the predicted classes
        colored_image = self.colorize_grids(real_image, predicted_classes, grid_size, class_colors, real_image.shape[0], real_image.shape[1])

        # Resize the colored image back to the original size
        self.original_image = real_image
        self.colored_image = colored_image

        self.display_results(class_names, class_colors)

    def divide_image_into_grids(self, image, grid_size):
        img_height, img_width, _ = image.shape
        grids = []
        for y in range(0, img_height, grid_size):
            for x in range(0, img_width, grid_size):
                grid = image[y:y+grid_size, x:x+grid_size]
                if grid.shape[0] != grid_size or grid.shape[1] != grid_size:
                    grid = cv2.resize(grid, (grid_size, grid_size))
                grids.append(grid)
        return np.array(grids)

    def colorize_grids(self, original_image, predictions, grid_size, class_colors, img_height, img_width):
        colored_image = np.zeros_like(original_image)
        grid_idx = 0
        for y in range(0, img_height, grid_size):
            for x in range(0, img_width, grid_size):
                if grid_idx < len(predictions):
                    color = np.array(class_colors[predictions[grid_idx]]) / 255.0
                    actual_grid_height = min(grid_size, img_height - y)
                    actual_grid_width = min(grid_size, img_width - x)
                    resized_color = cv2.resize(np.full((grid_size, grid_size, 3), color), (actual_grid_width, actual_grid_height))
                    colored_image[y:y+actual_grid_height, x:x+actual_grid_width] = resized_color
                    # Draw grid lines
                    cv2.rectangle(colored_image, (x, y), (x + actual_grid_width, y + actual_grid_height), (0, 0, 0), 1)
                    # Put grid numbers
                    cv2.putText(colored_image, str(grid_idx), (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    grid_idx += 1
        return colored_image

    def display_results(self, class_names, class_colors):
        self.canvas_original.delete("all")
        self.canvas_colored.delete("all")
        for widget in self.legend_frame.winfo_children():
            widget.destroy()

        # Get image dimensions
        img_height, img_width, _ = self.original_image.shape
        canvas_width = self.canvas_original.winfo_width()
        canvas_height = self.canvas_original.winfo_height()

        # Calculate the aspect ratio
        aspect_ratio = img_width / img_height

        # Calculate dimensions for display
        display_width = min(canvas_width, int(canvas_height * aspect_ratio))
        display_height = int(display_width / aspect_ratio)

        # Convert images to PIL format and resize to fit within the canvas while preserving aspect ratio
        original_image_pil = Image.fromarray((self.original_image * 255).astype(np.uint8))
        colored_image_pil = Image.fromarray((self.colored_image * 255).astype(np.uint8))

        original_image_pil = original_image_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)
        colored_image_pil = colored_image_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)

        # Convert images to ImageTk format
        original_image_tk = ImageTk.PhotoImage(original_image_pil)
        colored_image_tk = ImageTk.PhotoImage(colored_image_pil)

        # Display original image
        self.canvas_original.create_image(0, 0, anchor="nw", image=original_image_tk)
        self.canvas_original.image = original_image_tk

        # Display colored image
        self.canvas_colored.create_image(0, 0, anchor="nw", image=colored_image_tk)
        self.canvas_colored.image = colored_image_tk

        # Display legend centered
        total_legend_width = 0
        legend_labels = []

        max_height = 2  # set a fixed height for all labels in the legend
        for i, (class_name, color) in enumerate(class_colors.items()):
            legend_label = tk.Label(self.legend_frame, text=class_names[i], bg=f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}", font=("Helvetica", 12), height=max_height)
            legend_label.pack(side=tk.LEFT, padx=5, pady=5)
            legend_labels.append(legend_label)
            total_legend_width += legend_label.winfo_reqwidth() + 10

        # Center the legend frame
        self.legend_frame.pack_configure(padx=(self.winfo_width() - total_legend_width) // 2)

if __name__ == '__main__':
    app = SatelliteImageClassificationApp()
    app.mainloop()
