#This script displays a gear loading animation

import board
import math
import digitalio
from PIL import Image, ImageDraw, ImageFont
import time
import adafruit_rgb_display.st7789 as st7789

# Setup the display:
BAUDRATE = 24000000
spi = board.SPI()
cs_pin = digitalio.DigitalInOut(board.CE0)
dc_pin = digitalio.DigitalInOut(board.D25)
reset_pin = digitalio.DigitalInOut(board.D24)
disp = st7789.ST7789(spi, rotation=90, cs=cs_pin, dc=dc_pin, rst=reset_pin, baudrate=BAUDRATE)

# Create an 'RGB' drawing canvas
if disp.rotation % 180 == 90:
  # Swap height/width when rotating to landscape!
  height = disp.width
  width = disp.height
else:
  width = disp.width
  height = disp.height
original = Image.new('RGBA', (width, height))
draw = ImageDraw.Draw(original)

# Draw a white filled box to clear the image.
draw.rectangle((0, 0, width, height), outline=0, fill="#FFFFFF")
disp.image(original)

# Load and scale the gear image
gear_raw = Image.open("gear.png").convert("RGBA")
gear1_sized = gear_raw.resize((75, 75))
gear2_sized = gear_raw.resize((125, 125))

# Load the TTF font sizes for draw.text()
info_font = ImageFont.truetype('Ubuntu-M.ttf', 45)


# Blank white background to clear screen
image = original.copy()
draw = ImageDraw.Draw(image)

angle = 360
dots = 0
while True:

  angle -= 8
  if angle < 0: angle += 360

  dots += 0.12
  if dots >= 4: dots = 0

  # Blank white background to clear screen for next frame
  image = original.copy()
  draw = ImageDraw.Draw(image)

  # Add the gear images (rotated to angle "a" in degrees)
  gear2 = gear2_sized.copy()
  gear2 = gear2.rotate(360-angle)
  image.paste(gear2, (60, 50), gear2)

  gear1 = gear1_sized.copy()
  gear1 = gear1.rotate(angle)
  image.paste(gear1, (175, 65), gear1)

  # Add the loading text
  loading_text = "Loading"
  for i in range(math.floor(dots)): #dot animation
    loading_text += "."

  draw.text((75, 170), loading_text, font=info_font, fill="#000000")

  # Update the screen, then pause briefly before looping back
  disp.image(image)
  time.sleep(0.05)

# Update the screen
disp.image(image)
