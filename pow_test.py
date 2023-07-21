import math


constant = 2.4

width_long = 430
height_long = 900
radio_long = width_long / height_long
print(f'radio_long:{radio_long}')
radio_long_5pow = math.pow(radio_long, 5)*constant
print(f'radio_long_5pow:{radio_long_5pow}')
print(f'高度增加了:{radio_long_5pow*height_long}')

width_short = 350
height_short = 510
radio_short = width_short / height_short
print(f'radio_short:{radio_short}')
radio_short_5pow = math.pow(radio_short, 5)*constant
print(f'radio_short_5pow:{radio_short_5pow}')
print(f'高度增加了:{radio_short_5pow*height_short}')
