import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

class SatelliteImageClassification:
    def __init__(self):
        self.model = None
        self.path = None

    def load_model_SIC(self, model_path):
        self.path = model_path
        self.model = load_model(model_path)

    def load_image(self, image_path):
        image = load_img(image_path, target_size=(256, 256))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image /= 255.0
        return image

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

        return real_image, colored_image

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
    
    def plot_images(self, original_image, colored_image):
        import matplotlib.pyplot as plt
        
        original_image = (original_image * 255).astype(np.uint8)
        colored_image = (colored_image * 255).astype(np.uint8)
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 15))
        axs[0].imshow(original_image)
        axs[0].set_title('Image')
        axs[0].axis('off')
        axs[1].imshow(colored_image)
        axs[1].set_title('Predictions')
        axs[1].axis('off')

        model_name = self.path.split('/')[-1]
        
        plt.suptitle(model_name, fontsize=12)
        plt.subplots_adjust(top=1.57)
        plt.show()