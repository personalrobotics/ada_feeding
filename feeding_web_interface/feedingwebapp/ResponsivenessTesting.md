# Methodology for Testing Responsiveness 

After developing an app feature, make sure to test it on the following setups. Total 4 devices as listed below.
- **Smartphone:** iPhone 14 Plus [Tyler]; Width: 428 & Height: 926
    - Landscape
    - Portrait 
- **Tablet:** iPad Pro (6th gen 12.9") [Tyler]; Width: 1024 & Height: 1366
    - Landscape
    - Portrait
- **External Monitor:** Width: 4096 & Height: 2304
- **Laptop:** Width: 2560 & Height: 1600
    - Full-screen 
    - Half-screen (vertical split)
    - Quarter screen 

For the smartphone and laptops, test on Chrome, Firefox, Edge, and Safari. For the tablet and monitor, pick one browser to test it on. 

In total, you will run 23 tests (4 browsers * 1 smartphone * 2 orientations + 1 tablet * 2 orientations + 4 browsers * 3 laptops + 1 monitor). 

In portrait, pages should have their content appear on one screen and not require scrolling. In landscape, scrolling may be unavoidable for some screens, but try to rearrange the content horizontally in order to minimize required scrolling. 

If you don’t have the listed devices available for testing, please “Add custom device” in your browser’s responsive design mode using the width and height of the specific device’s resolution below. Please find the appropriate link from below for each browser's documentation on how to do this *except* Safari: 

1. https://www.ios-resolution.com/
2. https://support.apple.com/kb/SP748?locale=en_US
3. Adding custom device in Chrome: https://developer.chrome.com/blog/add-a-new-custom-device-as-a-preset/
4. Adding custom device in Edge: https://learn.microsoft.com/en-us/microsoft-edge/devtools-guide-chromium/device-mode/
5. Adding custom device in Firefox:
https://firefox-source-docs.mozilla.org/devtools-user/responsive_design_mode/#creating-custom-devices



