import board
import digitalio
from PIL import Image, ImageDraw, ImageFont
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
image = original.copy()
draw = ImageDraw.Draw(image)

# Load the TTF font sizes for draw.text()
info_font = ImageFont.truetype('Ubuntu-M.ttf', 32)

# Add the ADA text
#draw.text((140, 40), "Hi!", font=info_font, fill="#000000")
#draw.text((35, 80), "My name is ADA,", font=info_font, fill="#000000")
#draw.text((70, 120), "the Assistive", font=info_font, fill="#000000")
#draw.text((62, 160), "Dextrous Arm", font=info_font, fill="#000000")
draw.multiline_text((40, 40), "Hi!\nMy name is ADA,\nthe Assistive\nDextrous Arm", font=info_font, fill="#000000", spacing=5, align="center")

# Update the screen
disp.image(image)

