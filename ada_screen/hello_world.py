import board
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
original = Image.new('RGB', (width, height))
draw = ImageDraw.Draw(original)

# Draw a white filled box to clear the image.
draw.rectangle((0, 0, width, height), outline=0, fill="#FFFFFF")
disp.image(original)

# Load the TTF font sizes for draw.text()
fontlogo = ImageFont.truetype('Ubuntu-M.ttf', 32)
fontstats = ImageFont.truetype('Ubuntu-M.ttf', 20)

# Loop forever, one frame per loop
a = 200
while True:
  a+=10
  if a > 200: a=0

  # Blank white background to clear screen for next frame
  image = original.copy()
  draw = ImageDraw.Draw(image)

  # Add the ADA text
  draw.text((75, a), "Hello world!", font=fontlogo, fill="#000000")

  # Update the screen, then pause briefly before looping back
  disp.image(image)
  time.sleep(0.02)
