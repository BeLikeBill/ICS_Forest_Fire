import numpy as np
from skimage.morphology import dilation
from skimage import filters
import cv2
import sys


def scale_percent(img, percent):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def imgconvert(filename):
    """
    Convert any image to grayscale format.
    """
    #img = Image.open(filename).convert("L")
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def dilate(img):
    """
    Utilize dilation algorithm.
    """
    img = np.asarray(img) / np.max(img)

    # Dilate.
    dilated = dilation(img)
    if not dilated.any():
        print("Something went wrong while thinning. Double check your input")
        return img
    else:
        return dilated


def bitmap_trees_scaled(filename, percent):
    img = imgconvert(filename)
    result = scale_percent(np.asarray(img) / np.max(img) * 255, percent)
    val = filters.threshold_otsu(result)
    mask = result > val
    # result = np.asarray(mask) / np.max(mask)*255
    return mask
    # cv2.imwrite('./color_resize.jpeg', scale_percent(result, 20))


def bitmap_water_scaled(filename, percent):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    r, g, b = 135, 195, 190
    # print(np.where((img[:, :, 0] >= b) & (
    #     img[:, :, 1] <= g) & (img[:, :, 2] <= r)))
    bitmap = ((img[:, :, 0] >= b) & (img[:, :, 1] <= g) & (img[:, :, 2] <= r))
    img2 = scale_percent(np.asarray(bitmap) / True, percent)
    return (img2 * 255) > 100


def bitmap_road_scaled(filename, percent):
    img = imgconvert(filename)
    result = dilate(img) * 255 > 254

    result = scale_percent(result / True * 255, percent)

    return result > 80


def writefile(img, filename):
    cv2.imwrite(filename, img * 255)


def main():
    if len(sys.argv) < 1:
        print(
            "Usage: python3 read_map.py <image file>")
        sys.exit(1)

    wegenkaart = 'Atlaskaarten_Meinweg/WegenkaartPS.png'
    bomenkaart = 'Atlaskaarten_Meinweg/Bomenkaart.png'
    waterkaart = 'Atlaskaarten_Meinweg/WaterkaartPS.png'
    luchtfoto = 'Atlaskaarten_Meinweg/Luchtfoto.png'

    percent = 20

    water_bm = bitmap_water_scaled(waterkaart, percent)
    road_bm = bitmap_road_scaled(wegenkaart, percent)
    tree_bm = bitmap_trees_scaled(luchtfoto, percent)
    #moor_bm = bitmap_trees_scaled(luchtfoto, percent)

    writefile(water_bm, './Atlaskaarten_Meinweg/Waterbitmap.png')
    writefile(road_bm, './Atlaskaarten_Meinweg/Roadsbitmap.png')
    writefile(tree_bm, './Atlaskaarten_Meinweg/Treesbitmap.png')


if __name__ == '__main__':
    main()
