import random
import colorsys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ColorSampler:
    def __init__(self):
        """
        Defines the HSV ranges for distinct color categories.
        """
        self.categories = {
            'Red':    [(0.95, 1.0), (0.0, 0.04)], # Red wraps around 0 and 1
            'Orange': [(0.04, 0.12)],
            'Yellow': [(0.12, 0.18)],
            'Green':  [(0.20, 0.40)],
            'Cyan':   [(0.45, 0.55)],
            'Blue':   [(0.58, 0.68)],
            'Purple': [(0.70, 0.82)],
            'Pink':   [(0.85, 0.93)],
        }

    def sample_color(self, category_name=None):
        """
        1. Pick a category (randomly or specific).
        2. Sample a random Hue within that category's range.
        3. Sample random Saturation/Value for variety.
        """
        # Step 1: Pick Category
        if category_name is None:
            category_name = random.choice(list(self.categories.keys()))
        
        if category_name not in self.categories:
            raise ValueError(f"Category '{category_name}' not found. Available: {list(self.categories.keys())}")
            
        ranges = self.categories[category_name]
        
        # Step 2: Sample Hue
        # (Handle Red which might have two ranges like 0.0-0.05 and 0.95-1.0)
        hue_range = random.choice(ranges)
        hue = random.uniform(hue_range[0], hue_range[1])
        
        # Step 3: Sample Saturation & Value
        # Keep these high (0.5-1.0) to ensure the color looks "colored" and not gray/black
        saturation = random.uniform(0.5, 1.0)
        value = random.uniform(0.6, 1.0)
        
        # Convert to RGB (0-1 range for matplotlib, multiply by 255 for standard usage)
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return (r, g, b), category_name

# --- Visualization Code ---

# 1. Generate 50 random samples using the hierarchical method
if __name__ == "__main__":
    sampler = ColorSampler()
    num_samples = 50
    samples = [sampler.sample_color() for _ in range(num_samples)]

    # 2. Setup Plot
    cols = 10
    rows = (num_samples // cols) + (1 if num_samples % cols else 0)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.axis('off')

    # 3. Draw Swatches
    for i, (color_rgb, category) in enumerate(samples):
        # Grid math
        row = rows - 1 - (i // cols)
        col = i % cols
        
        # Draw the rectangle
        rect = patches.Rectangle(
            (col + 0.1, row + 0.2), 0.8, 0.6, 
            facecolor=color_rgb, edgecolor='none'
        )
        ax.add_patch(rect)
        
        # Label the category below it
        ax.text(
            col + 0.5, row + 0.05, category, 
            ha='center', fontsize=8, color='#333333'
        )

    plt.title("Hierarchical Sampling: Category First -> Then Hue", fontsize=14)
    plt.tight_layout()
    plt.savefig('color_sampling_demo.png')