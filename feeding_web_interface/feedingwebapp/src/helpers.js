/**
 * Takes in an imageWidth and imageHeight, and returns a width and height that
 * maintains the same aspect ratio but fits within the window.
 *
 * @param {number} imageWidth the original image's width in pixels
 * @param {number} imageHeight the original image's height in pixels
 * @param {number} marginTop the desired top margin between window and image, in pixels
 * @param {number} marginBottom the desired bottom margin between window and image, in pixels
 * @param {number} marginLeft the desired left margin between window and image, in pixels
 * @param {number} marginRight the desired right margin between window and image, in pixels
 *
 * @returns {object} the width and height of the image that fits within the window and has the requested margins
 */
export function scaleWidthHeightToWindow(imageWidth, imageHeight, marginTop = 0, marginBottom = 0, marginLeft = 0, marginRight = 0) {
  // Calculate the aspect ratio of the image
  let imageAspectRatio = imageWidth / imageHeight

  // Get the aspect ratio of the available subset of the window
  let availableWidth = window.innerWidth - marginLeft - marginRight
  let availableHeight = window.innerHeight - marginTop - marginBottom
  let availableAspectRatio = availableWidth / availableHeight

  // Calculate the width and height of the image that fits within the window
  let returnWidth, returnHeight
  if (availableAspectRatio > imageAspectRatio) {
    returnHeight = availableHeight
    returnWidth = imageAspectRatio * returnHeight
  } else {
    returnWidth = availableWidth
    returnHeight = returnWidth / imageAspectRatio
  }

  // Calculate the scale factor
  let scaleFactorWidth = returnWidth / imageWidth
  let scaleFactorHeight = returnHeight / imageHeight
  let scaleFactor = (scaleFactorWidth + scaleFactorHeight) / 2

  return {
    width: returnWidth,
    height: returnHeight,
    scaleFactor: scaleFactor
  }
}

/**
 * Takes in a number of REM units and returns the equivalent number of pixels.
 * See https://blog.hubspot.com/website/css-rem form more info re. REM units.
 *
 * @param {float} rem the number of REM units
 * @returns {float} the number of pixels
 */
export function convertRemToPixels(rem) {
  return rem * parseFloat(getComputedStyle(document.documentElement).fontSize)
}
