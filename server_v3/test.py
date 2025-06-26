import os

OPENSLIDE_PATH = os.path.abspath(r'openslide-win64\bin')
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


from findabac.processor import scan_item

item = scan_item.ScanItem(openslide, "demo.tiff", "MyUserIDHere", False)

item.add_detection((1000, 2000))
item.add_detection((2000, 2000))
item.add_detection((3000, 2000))
item.add_detection((4000, 2000))


raw = item.to_bytes(archive=False)
print("Item Size:", len(raw) // 1024, "KB")


loaded = scan_item.ScanItem.from_bytes(openslide, raw)

print("Loaded Locations:", loaded.get_detections_locations())

location = loaded.get_detections_locations()[0]

item.get_detections_image(location).show()
loaded.get_detections_image(location).show()

