import os

OPENSLIDE_PATH = os.path.abspath(r'openslide-win64\bin')

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


path = r"E:\Python\findabac\Find-A-Bac\trainingData\16772.tiff"

slide = openslide.open_slide(path)
img = slide.get_thumbnail((4000, 4000))

img.save("new_image.png")
