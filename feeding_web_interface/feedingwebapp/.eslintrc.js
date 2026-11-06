/*
 * Copyright (c) 2024-2025, Personal Robotics Laboratory
 * License: BSD 3-Clause. See LICENSE.md file in root directory.
 */

module.exports = {
  root: true,
  parserOptions: {
    ecmaVersion: 2020,
    sourceType: 'module',
    ecmaFeatures: {
      jsx: true
    }
  },
  settings: {
    react: {
      version: 'detect'
    }
  },
  env: {
    jest: true,
    browser: true,
    amd: true,
    node: true
  },
  extends: [
    'eslint:recommended',
    'react-app',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
    'plugin:prettier/recommended' // Make this the last element so prettier config overrides other formatting rules
  ],
  rules: {
    'no-unused-vars': ['error', { vars: 'all', args: 'after-used', ignoreRestSiblings: false }],
    'prettier/prettier': ['error', {}, { usePrettierrc: true }]
  }
}
