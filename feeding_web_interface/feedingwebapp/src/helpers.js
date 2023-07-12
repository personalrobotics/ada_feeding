import { useLayoutEffect, useState } from 'react'

/**
 * Returns the window size, which gets updated every time the window is resized.
 *
 * @param {func} resizeCallback an optional function to call when the window is
 *     resized. Note that this function cannot use any react hooks.
 */
export function useWindowSize(resizeCallback = null) {
  const [windowSize, setWindowSize] = useState({ width: 0, height: 0 })
  useLayoutEffect(() => {
    // set current window size
    function updateWindowSize() {
      // There is a known bug in Chrome iOS, where the window's innerHeight
      // is smaller than it should be immediately after a rotation from
      // from landscape to portrait. To address this, we wait 100ms before
      // updating the windowSize.
      setTimeout(() => {
        setWindowSize({ width: window.innerWidth, height: window.innerHeight })
        if (resizeCallback) {
          resizeCallback()
        }
      }, 200)
    }
    window.addEventListener('resize', updateWindowSize)
    updateWindowSize()
    return () => window.removeEventListener('resize', updateWindowSize)
  }, [resizeCallback])
  return windowSize
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
