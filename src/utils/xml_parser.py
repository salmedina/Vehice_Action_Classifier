from bs4 import BeautifulSoup

f = open('../../data/DETRAC-Train-Annotations-XML/MVI_20011.xml', 'r').read()
e = BeautifulSoup(f, features="html5lib")
# frames in a video
frames = e.findAll('target_list')
for frame in frames:
    vehicles = frame.findAll('target')
    for vehicle in vehicles:
        b = vehicle.box.attrs
        orientation = vehicle.attribute.attrs
        print(b, orientation)
