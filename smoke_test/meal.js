/*
 * Copyright (c) 2026, Personal Robotics Laboratory
 * License: BSD 3-Clause. See LICENSE.md file in root directory.
 *
 * Drives one feeding session through the web app in dummy mode. It starts
 * feeding, pauses and resumes a motion (which cancels and resends its goal),
 * selects food, acquires it, transfers it to the mouth, and finishes the meal.
 *
 *   node meal.js <artifacts_dir> [app_url]
 *
 * Writes events.jsonl (every meal-state change), result.json, screenshots/,
 * video/, and logs/browser_console.log. Exits 0 if every step passed.
 */
const fs = require('fs')
const path = require('path')
const { chromium } = require('playwright')

const out = process.argv[2]
const appUrl = process.argv[3] || 'http://localhost:3000'
const STEP_TIMEOUT_MS = 60000

const t0 = Date.now()
const events = fs.createWriteStream(path.join(out, 'events.jsonl'))
const consoleLog = fs.createWriteStream(path.join(out, 'logs', 'browser_console.log'))
fs.mkdirSync(path.join(out, 'screenshots'), { recursive: true })

// The web app persists its state (including the meal state) to localStorage.
const readAppState = (page) =>
  page.evaluate(() => {
    const s = JSON.parse(localStorage.getItem('ada_web_app_global_state') || '{}').state || {}
    return { mealState: s.mealState, paused: s.paused }
  })

let current = {}
async function pollAppState(page) {
  for (;;) {
    let state
    try {
      state = await readAppState(page)
    } catch (e) {
      return // The page closed
    }
    if (state.mealState !== current.mealState || state.paused !== current.paused) {
      current = state
      events.write(JSON.stringify({ t_ms: Date.now() - t0, time: new Date().toISOString(), ...state }) + '\n')
    }
    await new Promise((r) => setTimeout(r, 100))
  }
}

async function waitFor(description, condition, timeoutMs = STEP_TIMEOUT_MS) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (condition()) return
    await new Promise((r) => setTimeout(r, 100))
  }
  throw new Error(`Timed out after ${timeoutMs / 1000}s waiting for ${description}; app state is ${JSON.stringify(current)}`)
}
const waitForMealState = (mealState, timeoutMs) => waitFor(`meal state ${mealState}`, () => current.mealState === mealState, timeoutMs)
const clickIcon = (page, alt) => page.locator(`button:has(img[alt="${alt}"])`).first().click()

const steps = (page) => [
  ['open the app', async () => waitForMealState('U_PreMeal')],
  [
    'start feeding',
    async () => {
      await page.getByText('Start Feeding').click()
      await waitForMealState('R_MovingAbovePlate')
    }
  ],
  [
    'pause the motion (cancels its goal)',
    async () => {
      // Wait for the robot's feedback, so the goal is active when we cancel it.
      await page.getByText(/Robot is (thinking|moving)/).waitFor({ timeout: STEP_TIMEOUT_MS })
      await page.getByRole('button', { name: 'Pause' }).click()
      await waitFor('paused', () => current.paused === true)
    }
  ],
  [
    'resume the motion',
    async () => {
      await page.getByRole('button', { name: 'Resume' }).click()
      await waitForMealState('U_BiteSelection')
    }
  ],
  [
    'select food',
    async () => {
      const video = page.locator('video')
      await page.waitForFunction(() => document.querySelector('video')?.readyState >= 2, null, { timeout: STEP_TIMEOUT_MS })
      await video.click() // The center of the plate image
      await page.locator('button[value="0"]').waitFor({ timeout: STEP_TIMEOUT_MS })
      await page.screenshot({ path: path.join(out, 'screenshots', 'food_masks.png') })
      await page.locator('button[value="0"]').click()
      await waitForMealState('R_BiteAcquisition')
    }
  ],
  ['acquire the bite', async () => waitForMealState('U_BiteAcquisitionCheck')],
  [
    'confirm food on fork',
    async () => {
      await clickIcon(page, 'move_to_mouth_image')
      await waitForMealState('R_DetectingFace')
    }
  ],
  [
    'transfer to mouth',
    async () => {
      await page.getByText('Mouth detected!').waitFor({ timeout: STEP_TIMEOUT_MS })
      await clickIcon(page, 'move_to_mouth_image')
      await waitForMealState('U_BiteDone')
    }
  ],
  [
    'finish the bite',
    async () => {
      await clickIcon(page, 'move_above_plate_image')
      await waitForMealState('U_BiteSelection')
    }
  ],
  [
    'finish the meal',
    async () => {
      await page.getByText('Done Eating').click()
      await waitForMealState('U_PostMeal')
    }
  ]
]

;(async () => {
  const browser = await chromium.launch({ args: ['--use-fake-ui-for-media-stream'] })
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    recordVideo: { dir: path.join(out, 'video'), size: { width: 1280, height: 800 } }
  })
  const page = await context.newPage()
  page.on('console', (m) => consoleLog.write(`${Date.now() - t0}ms [${m.type()}] ${m.text()}\n`))
  page.on('pageerror', (e) => consoleLog.write(`${Date.now() - t0}ms [pageerror] ${e.message}\n`))

  // Start from the app's default settings (auto-continue off).
  await page.goto(appUrl, { waitUntil: 'networkidle' })
  await page.evaluate(() => localStorage.clear())
  await page.reload({ waitUntil: 'networkidle' })
  pollAppState(page)

  const result = { passed: true, steps: [] }
  const all = steps(page)
  for (const [i, [name, run]] of all.entries()) {
    const start = Date.now()
    try {
      await run()
      result.steps.push({ step: name, passed: true, seconds: (Date.now() - start) / 1000 })
      await page.screenshot({ path: path.join(out, 'screenshots', `${String(i + 1).padStart(2, '0')}_${name.replace(/\W+/g, '_')}.png`) })
      console.log(`ok   ${i + 1}/${all.length} ${name}`)
    } catch (e) {
      result.passed = false
      result.steps.push({ step: name, passed: false, seconds: (Date.now() - start) / 1000, error: e.message })
      await page.screenshot({ path: path.join(out, 'screenshots', 'failure.png') })
      console.log(`FAIL ${i + 1}/${all.length} ${name}: ${e.message}`)
      break
    }
  }

  fs.writeFileSync(path.join(out, 'result.json'), JSON.stringify(result, null, 2) + '\n')
  await context.close() // Finishes the video
  await browser.close()
  events.end()
  consoleLog.end()
  process.exit(result.passed ? 0 : 1)
})()
