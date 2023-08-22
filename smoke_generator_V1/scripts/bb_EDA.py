import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.image_composition import composite_smoke,get_bounding_boxes

##-------------------------------------------------------------------------------------------------------------------
##                         Script to perform EDA on smoke bounding boxes to determine best rescaling factor
##-------------------------------------------------------------------------------------------------------------------


def analyze_smoke_images(smoke_folder, background_path):
    # Load the fixed background image
    background = Image.open(background_path).convert("RGBA")
    background_width, background_height = background.size
    
    # Initialize a list to store bounding box characteristics
    bounding_box_data = []

    # Iterate through smoke images
    for plume_folder in os.listdir(smoke_folder):
        plume_path = os.path.join(smoke_folder, plume_folder)
        if os.path.isdir(plume_path):
            # Iterate through smoke image files
            for smoke_image_file in os.listdir(plume_path):
                if smoke_image_file.endswith('.png'):
                    smoke_image_path = os.path.join(plume_path, smoke_image_file)
                    
                    # Load the smoke image
                    smoke_image = Image.open(smoke_image_path)
                    
                    # Get bounding boxes
                    bbox = get_bounding_boxes(smoke_image)

                    if bbox :
                        # Calculate other characteristics based on the bounding box
                        x, y, width, height = 0,0, bbox.width,bbox.height
                        aspect_ratio = width / height
                        box_area = width * height
                        box_position = (x + width / 2, y + height / 2)
                        image_area = smoke_image.width * smoke_image.height
                        box_area_ratio = box_area / image_area
                        
                        bounding_box_data.append({
                            'PlumeFolder': plume_folder,
                            'ImageFile': smoke_image_file,
                            'AspectRatio': aspect_ratio,
                            'BoxArea': box_area,
                            'BoxAreaRatio': box_area_ratio
                        })

    # Create a pandas DataFrame from the collected data
    df = pd.DataFrame(bounding_box_data)

    return df


def perform_eda(df):
    # Summary statistics
    print(df.describe())

    # Box plot of Box Area Ratio per Plume Folder
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='PlumeFolder', y='BoxAreaRatio')
    plt.title('Box Plot of Box Area Ratio per Plume Folder')
    plt.xlabel('Plume Folder')
    plt.ylabel('Box Area Ratio')
    plt.xticks(rotation=45)
    plt.show()

def extremums(df):
    # Calculate extremum values per Plume Folder
    extremum_values = df.groupby('PlumeFolder')['BoxAreaRatio'].agg(['min', 'max'])

    # Create a DataFrame with extremum values and image information
    extremum_info = []
    for plume_folder in extremum_values.index:
        min_image = df[(df['PlumeFolder'] == plume_folder) & (df['BoxAreaRatio'] == extremum_values.loc[plume_folder, 'min'])].iloc[0]
        max_image = df[(df['PlumeFolder'] == plume_folder) & (df['BoxAreaRatio'] == extremum_values.loc[plume_folder, 'max'])].iloc[0]
        extremum_info.append({
            'PlumeFolder': plume_folder,
            'MinImageFile': min_image['ImageFile'],
            'MinBoxAreaRatio': extremum_values.loc[plume_folder, 'min'],
            'MaxImageFile': max_image['ImageFile'],
            'MaxBoxAreaRatio': extremum_values.loc[plume_folder, 'max']
        })
    extremum_info_df = pd.DataFrame(extremum_info)

    return extremum_info_df

def visualize_composition(df, smoke_folder, background_path):
    # Summary statistics
    print(df.describe())

    # Calculate extremum information
    extremum_info_df = extremums(df)

    # Display extremum information table
    print("\nExtremum Information for Box Area Ratio:")
    print(extremum_info_df)

    # Display composite images for each smoke plume folder
    for _, row in extremum_info_df.iterrows():
        min_image_path = os.path.join(smoke_folder, row['PlumeFolder'], row['MinImageFile'])
        max_image_path = os.path.join(smoke_folder, row['PlumeFolder'], row['MaxImageFile'])

        composite_min, _ = composite_smoke(background_path, min_image_path,rescaling_factor=0.2)
        composite_max, _ = composite_smoke(background_path, max_image_path,rescaling_factor=0.2 )

        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(composite_min)
        plt.title(f"Min Image ({row['MinImageFile']})")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(composite_max)
        plt.title(f"Max Image ({row['MaxImageFile']})")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

# Rest of the code remains the same

# Example usage
smoke_folder = r'..\blender_images'
background_path = r'..\background_images\image_0168.png'
script_dir = os.path.dirname(os.path.abspath(__file__))
#print(os.path.join(script_dir,smoke_folder),os.path.exists( os.path.join(script_dir,background_path)))

df = analyze_smoke_images(os.path.join(script_dir,smoke_folder), os.path.join(script_dir,background_path))

#perform_eda
#perform_eda(df)
# Visualize composition to determine 
visualize_composition(df, os.path.join(script_dir,smoke_folder),  os.path.join(script_dir,background_path))


