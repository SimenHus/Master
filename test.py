
from src.util import Geometry
import numpy as np

lat0, lon0, alt0 = 63.442291563667, 10.4179306566528, 38.69057083129883
lat1, lon1, alt1 = 63.442231026938586, 10.418093132398543, 38.83717727661133
lat2, lon2, alt2 = 63.44221480817148, 10.418153727628267, 38.87126541137695

r0, p0, y0 = 0.7576443552970886, 1.1649905443191528, 17.132890701293945
r1, p1, y1 = 0.887495219707489, 1.1563706398010254, 16.787208557128906
r2, p2, y2 = 1.1721217632293701, 0.9684468507766724, 348.6752014160156


h0 = 0.09271904826164246
h1 = 0.10885725170373917
h2 = 0.07315672188997269

pos0 = Geometry.Geode.LLA_to_ECEF(lon0, lat0, alt0)
pos1 = Geometry.Geode.LLA_to_ECEF(lon1, lat1, alt1)
pos2 = Geometry.Geode.LLA_to_ECEF(lon2, lat2, alt2)

print(pos1)
print(pos2)

rel01 = pos1 - pos0
rel02 = pos2 - pos0

ang01 = [r1 - r0, p1 - p0, y1 - y0]
ang02 = [r2 - r0, p2 - p0, y2 - y0]

NED01 = Geometry.Geode.LLA_to_NED(lon1, lat1, alt1, lon0, lat0, alt0)
NED02 = Geometry.Geode.LLA_to_NED(lon2, lat2, alt2, lon0, lat0, alt0)

h01 = h1 - h0
h02 = h2 - h0

print(f'Pos step 16: {rel01}')
print(f'Pos step 17: {rel02}')

print(f'Ang step 16: {ang01}')
print(f'Ang step 17: {ang02}')

print(f'NED step 16: {NED01}')
print(f'NED step 17: {NED02}')

print(f'Heave step 16: {h01}')
print(f'Heave step 17: {h02}')