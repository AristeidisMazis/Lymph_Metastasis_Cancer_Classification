import numpy as np
from openslide import open_slide
import cv2
from openslide.deepzoom import DeepZoomGenerator

# Load slide
slide = open_slide("D:/images/wsi/tumor_077.tif")

# Slide dimensions
slide_dims = slide.dimensions
print("Slide Dimensions:", slide_dims)

# Visualise thumbnail
slide_thumb = slide.get_thumbnail(size=(1000, 1000))
# slide_thumb.show()

# Level dimensions
level_dims = slide.level_dimensions
print("Number of levels:", len(level_dims))
print("Each Level Dimensions:", level_dims)
level5 = slide.read_region((0, 0), 5, level_dims[5])
level5_rgb = level5.convert('RGB')
level5_img = np.array(level5_rgb)
# cv2.imwrite("D:/images/saved_tiles/Downsampled images/tumor_104_downsample.png", level5_img)


# Read masks and convert to binary
tissue_mask = cv2.imread("D:/images/saved_tiles/Masks/tumor_077_tissue_mask.png", 0)
tumor_mask = cv2.imread("D:/images/saved_tiles/Masks/tumor_077_tumor_mask.png", 0)
ret1, tissue_mask = cv2.threshold(tissue_mask, 5, 255, cv2.THRESH_BINARY)
ret2, tumor_mask = cv2.threshold(tumor_mask, 5, 255, cv2.THRESH_BINARY)
# cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
# cv2.imshow('Image', tissue_mask)
# cv2.waitKey(0)

# Get the necessary parameters
cols, rows = 0, 0
tissue_cols, tissue_rows = 0, 0
tile_size = 512  # ****select tile size here**** #
downsample = 2 ** 6
mask_tile_size = int(tile_size/downsample)
normal_tile_dir_name = "D:/images/saved_tiles/normal/13_"
tumor_tile_dir_name = "D:/images/saved_tiles/tumor/12_"
# normal_tile_dir_name = "D:/images/saved_tiles/normal_256/13_"
# tumor_tile_dir_name = "D:/images/saved_tiles/tumor_256/12_"
tiles = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0, limit_bounds=False)
tiles_level_count = tiles.level_count - 1

# For loop in masks and save patches
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
            if not np.any(tumor_mask_tile == 0):
                cv2.imwrite(tumor_tile_dir_name + tile_name + "_tumor.png", temp_tile_np)
            else:
                p = np.random.rand()
                if p >= 0.6:
                    cv2.imwrite(normal_tile_dir_name + tile_name + "_normal.png", temp_tile_np)
