// A ready-made dataset (circles, triangles, squares) so the whole flow can be tried
// without a webcam, e.g. on a projector during a demo.
import { newId } from "../model/graph.js";
import { makeSample, SAMPLE_SIZE } from "./images.js";

const rand = (a, b) => a + Math.random() * (b - a);

function draw(kind) {
  const c = document.createElement("canvas");
  c.width = c.height = SAMPLE_SIZE;
  const g = c.getContext("2d", { willReadFrequently: true });
  g.fillStyle = `hsl(${rand(0, 360)} ${rand(10, 40)}% ${rand(78, 94)}%)`;
  g.fillRect(0, 0, SAMPLE_SIZE, SAMPLE_SIZE);
  for (let i = 0; i < 60; i++) {
    g.fillStyle = `hsl(0 0% ${rand(40, 100)}% / 0.25)`;
    g.fillRect(rand(0, SAMPLE_SIZE), rand(0, SAMPLE_SIZE), rand(2, 8), rand(2, 8));
  }
  const r = rand(34, 66);
  const x = rand(r + 6, SAMPLE_SIZE - r - 6),
    y = rand(r + 6, SAMPLE_SIZE - r - 6);
  g.fillStyle = `hsl(${rand(0, 360)} ${rand(45, 80)}% ${rand(25, 50)}%)`;
  g.translate(x, y);
  g.rotate(rand(-0.6, 0.6));
  g.beginPath();
  if (kind === "circle") g.arc(0, 0, r, 0, Math.PI * 2);
  if (kind === "square") g.rect(-r * 0.85, -r * 0.85, r * 1.7, r * 1.7);
  if (kind === "triangle") {
    g.moveTo(0, -r);
    g.lineTo(r * 0.95, r * 0.75);
    g.lineTo(-r * 0.95, r * 0.75);
    g.closePath();
  }
  g.fill();
  return c;
}

const KINDS = ["circle", "triangle", "square"];
export const drawShape = (kind = KINDS[Math.floor(Math.random() * KINDS.length)]) => draw(kind);

export function makeShapesDataset(perClass = 40) {
  return [
    ["동그라미", "circle"],
    ["세모", "triangle"],
    ["네모", "square"],
  ].map(([name, kind]) => ({
    id: newId("c"),
    name,
    samples: Array.from({ length: perClass }, () => makeSample(draw(kind))),
  }));
}
