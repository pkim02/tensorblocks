---
version: alpha
name: TensorBlocks
description: AI 입문자와 비전공자가 Teachable Machine처럼 데이터를 모으고, 모델 안을 블록으로 열어 고치고, PyTorch 코드로 가져가는 웹 앱.
colors:
  page: "kit default, tinted toward graph-paper green"
  text: "kit gray-12"
  accent: "oklch(0.5 0.11 165)"
  on-accent: "oklch(1 0 0)"
typography:
  body:
    fontFamily: IBM Plex Sans KR
    fontSize: 1rem
    lineHeight: 1.6
  h1:
    fontFamily: IBM Plex Sans KR (700)
    fontSize: 2.8125rem
rounded:
  control: 6px
  surface: 8px
---

## Overview

The world of this app: a 모눈 공책 (engineering graph paper), a webcam pointed at your desk,
the tensor shape `(32, 3, 224, 224)`, and the red pen a TA uses to circle the line where your
code broke (`mat1 and mat2 shapes cannot be multiplied`).

The page should feel like a lab notebook, not a SaaS dashboard. The one memorable element is the
**red-pen correction note**: when two blocks don't fit, the connection turns red and a short
handwritten-style note says what's wrong and offers the fix button. Everything else is quiet.

## Colors

Light mode only (index.html pins `data-mode="light"`): a lab notebook is paper. Accent is graph-paper green ink (hue 165). It marks the one primary action per view
(학습시키기, 블록 캔버스 열기), selected blocks and links. Red (kit danger) is reserved for shape
errors only, so it always means "this connection is broken". Amber (kit warning) means
"works but slow". OK connections are plain grey, never green, so green ink and "fine" don't blur.

Kit skin:

```css
:root {
  --accent-h: 165;
  --accent-c: 0.11;
  --accent-solid-l: 0.5;
  --on-accent: oklch(1 0 0);
  --neutral-h: 165;
  --neutral-c: 0.01;
  --font-body: "IBM Plex Sans KR", system-ui, sans-serif;
  --font-display: var(--font-body);
  --font-mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
  --radius-control: 6px;
  --radius-surface: 8px;
}
```

## Typography

- IBM Plex Sans KR for all UI text (Korean + Latin in one family, engineering voice).
- IBM Plex Mono for tensor shapes (`64×64×16`), parameter counts and exported code. Shapes are
  data, so mono is earned here; never use mono for decorative labels.
- Gaegu 700 only for the red-pen correction note title. Nowhere else.

## Layout

Left-aligned. Main screen follows Teachable Machine's left-to-right flow: data, model + training,
preview. The block canvas opens full-screen from the model column.

```
┌ TensorBlocks ─────────────────────────────────────────────┐
│ 데이터 (classes)   │  모델 + 학습        │  미리보기          │
│ [Class 1 ▢▢▢▢]    │  Input 64×64×3      │  [webcam]          │
│ [Class 2 ▢▢▢▢]    │   │ 64×64×16        │  Class 1 ████ 82%  │
│ + 클래스 추가       │  Conv2D …          │  모델 속 보기 ▦▦▦   │
│                    │  [블록 캔버스 열기] │  [코드 내보내기]     │
│                    │  [학습시키기]       │                    │
└────────────────────────────────────────────────────────────┘
Canvas: palette (left) · graph-paper canvas (centre) · inspector (right)
On the canvas the model runs left to right: Input ─64×64×3─▶ Conv2D ─▶ … ─▶ Output.
Correction notes hang under the broken line.
```

## Elevation & Depth

Kit default: only floating things (menus, dialogs, toasts) have shadows. Blocks on the canvas are
flat with a border; the selected block gets an accent border, not a shadow.

## Shapes

Controls 6px, blocks and panels 8px. Edge shape tags are pills.

## Components

Kit: btn, field, segmented, list, table, dialog, tabs, toast, badge, callout, empty.
Project: class card with sample strip, block node, shape edge tag, correction note, layer activation strip, training chart.

## Do's and Don'ts

- Do show the tensor shape on every connection. That's the product.
- Do explain each block in one plain Korean sentence; technical names (Conv2D, Dense) stay in English.
- Don't use red for anything except a broken connection.
- Don't put cards inside cards; the main columns are plain sections with headings.
- No emoji icons, no gradients, no invented stats.
