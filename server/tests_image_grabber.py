import os

OPENSLIDE_PATH = os.path.abspath(r'openslide-win64\bin')

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


path = r"C:\Users\danie\Documents\Programming\Python\CNN_Python_Opencl\trainingData\20240524_144610.tiff"

slide = openslide.open_slide(path)
img = slide.get_thumbnail((4000, 4000))

img.save("new_image.png")
