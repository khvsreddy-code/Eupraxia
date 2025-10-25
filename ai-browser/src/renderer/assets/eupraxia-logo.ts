// Exports a small inline SVG data URL as the app logo. This file attempts to prefer a
// local `eupraxia-logo.png` if present in the same folder (so you can drop the PNG there).
// Otherwise it falls back to an inline SVG data URL (the visual placeholder).

declare const require: any;

const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns='http://www.w3.org/2000/svg' width='512' height='512' viewBox='0 0 512 512'>
  <rect width='100%' height='100%' fill='#0b0b0f' />
  <defs>
    <radialGradient id='g' cx='50%' cy='35%'>
      <stop offset='0%' stop-color='#fff0b8'/>
      <stop offset='50%' stop-color='#ffd86b'/>
      <stop offset='100%' stop-color='#b37a18'/>
    </radialGradient>
    <filter id='glow' x='-50%' y='-50%' width='200%' height='200%'>
      <feGaussianBlur stdDeviation='8' result='coloredBlur'/>
      <feMerge>
        <feMergeNode in='coloredBlur'/>
        <feMergeNode in='SourceGraphic'/>
      </feMerge>
    </filter>
  </defs>

  <!-- outer ring -->
  <circle cx='256' cy='180' r='140' fill='none' stroke='url(#g)' stroke-width='14' filter='url(#glow)' />

  <!-- eye -->
  <ellipse cx='256' cy='180' rx='92' ry='48' fill='#000' />
  <ellipse cx='256' cy='180' rx='44' ry='22' fill='#ffd86b' />
  <circle cx='256' cy='180' r='12' fill='#000' />

  <!-- accents -->
  <g fill='none' stroke='#ffd86b' stroke-width='2' opacity='0.9'>
    <path d='M256 36 L256 0' transform='translate(0,64)' />
    <path d='M256 324 L256 360' transform='translate(0,-64)' />
  </g>

  <!-- wordmark -->
  <text x='50%' y='430' text-anchor='middle' font-family='Georgia, serif' font-size='44' fill='#ffd86b' letter-spacing='6'>EUPRAXIA</text>
</svg>`;

const dataUrl = 'data:image/svg+xml;utf8,' + encodeURIComponent(svg);

// Try to prefer a PNG file placed next to this file at runtime. If node fs isn't
// available or the PNG doesn't exist, return the SVG data URL fallback.
let logoUrl = dataUrl;
try {
    const fs = require('fs');
    const path = require('path');
    const pngPath = path.join(__dirname, 'eupraxia-logo.png');
    if (fs.existsSync(pngPath)) {
        // file:// url works in Electron renderer when loading local assets
        logoUrl = `file://${pngPath.replace(/\\/g, '/')}`;
    }
} catch (e) {
    // ignore and use dataUrl
}

export default logoUrl;
