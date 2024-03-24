import numpy as np
from openslide import open_slide
import cv2
from openslide.deepzoom import DeepZoomGenerator
from numpy import savetxt

# Load slide
slide = open_slide("C:/")

# Slide dimensions
slide_dims = slide.dimensions

# Level dimensions
level_dims = slide.level_dimensions
print("Number of levels:", len(level_dims))
print("Each Level Dimensions:", level_dims)
level6 = slide.read_region((0, 0), 6, level_dims[6])
level6_rgb = level6.convert('RGB')
level6_img = np.array(level6_rgb)
# cv2.imwrite("D:/images/tumor_104.png", level6_img)


# Read masks and convert to binary
tissue_mask = cv2.imread("C:/", 0)
tumor_mask = cv2.imread("C:/", 0)
ret1, tissue_mask = cv2.threshold(tissue_mask, 5, 255, cv2.THRESH_BINARY)
ret2, tumor_mask = cv2.threshold(tumor_mask, 5, 255, cv2.THRESH_BINARY)


# Get the necessary parameters
cols, rows = 0, 0
tissue_cols, tissue_rows = 0, 0
tile_size = 512  # ****select tile size here**** #
downsample = 2 ** 6
mask_tile_size = int(tile_size/downsample)
tile_dir_name = "C:/"
tiles = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0, limit_bounds=False)
tiles_level_count = tiles.level_count - 1


rows = int(slide_dims[1]/downsample)
cols = int(slide_dims[0]/downsample)

map = np.zeros((rows, cols))
calc = 0

# For loop in masks and save patches for prediction in proper form class_number
for row in range(int(slide_dims[1]/tile_size)):
    for col in range(int(slide_dims[0]/tile_size)):
        tile_name = str(row) + "_" + str(col)
        row_start = row * mask_tile_size
        col_start = col * mask_tile_size
        row_end = row_start + mask_tile_size
        col_end = col_start + mask_tile_size
        tissue_mask_tile = tissue_mask[row_start:row_end, col_start:col_end]
        tumor_mask_tile = tumor_mask[row_start:row_end, col_start:col_end]
        if not np.any(tissue_mask_tile == 0):
            temp_tile = tiles.get_tile(tiles_level_count, (col, row))
            temp_tile_RGB = temp_tile.convert('RGB')
            temp_tile_np = np.array(temp_tile_RGB)
            map[row_start:row_end, col_start:col_end] = 1
            calc += 1
            if not np.any(tumor_mask_tile == 0):
                cv2.imwrite(tile_dir_name + "tumor_" + str(calc) + ".png", temp_tile_np)
            else:
                cv2.imwrite(tile_dir_name + "normal_" + str(calc) + ".png", temp_tile_np)

# Save data for heatmap
data = np.array([int(slide_dims[1]/tile_size), int(slide_dims[0]/tile_size), mask_tile_size, calc])
savetxt('map.csv', map, delimiter=',')
savetxt('data.csv', data, delimiter=',')
