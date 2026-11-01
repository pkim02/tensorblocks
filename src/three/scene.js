// 3D picture of the model: every block's output tensor drawn as a box of cells,
// left to right. Image tensors are H tall, W wide and C deep (one slice per channel);
// vectors are tall bars. Sizes grow slower than the real numbers (d^0.42) so a
// 7×7×1280 map and a 224×224×3 photo both fit on screen.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const L = (d) => 2.2 * Math.pow(d, 0.42);
const CAP = 110; // longest side drawn; longer tensors get a break mark
const GAP = 14;
const VIEW_DIR = new THREE.Vector3(0.42, 0.5, 1).normalize();

// Light-mode palette (graph-paper notebook): paper → green ink for activations.
const PAPER = [246, 248, 247];
const INK = [24, 82, 66];
const C = {
  page: 0xfafbfa,
  floor: 0xdfe6e3,
  floorMinor: 0xedf1ef,
  face: 0xe9efec,
  edge: 0x8a9893,
  edgeOn: 0x1f8a6b,
  error: 0xc8352b,
  frozen: 0x3b4650,
  link: 0xa9b5b0,
};

const ramp = (v) => PAPER.map((p, i) => Math.round(p + (INK[i] - p) * Math.max(0, Math.min(1, v))));

function dataTexture(w, h, pixel) {
  const buf = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [r, g, b] = pixel(x, y);
      const o = ((h - 1 - y) * w + x) * 4; // DataTexture rows start at the bottom
      buf[o] = r;
      buf[o + 1] = g;
      buf[o + 2] = b;
      buf[o + 3] = 255;
    }
  }
  const t = new THREE.DataTexture(buf, w, h);
  t.magFilter = THREE.NearestFilter;
  t.minFilter = THREE.NearestFilter;
  t.colorSpace = THREE.SRGBColorSpace;
  t.needsUpdate = true;
  return t;
}

const mapTexture = (data, w, h) => dataTexture(w, h, (x, y) => ramp(data[y * w + x]));

// The photo split into its red, green and blue channels, one slice each.
function rgbTextures(canvas, size) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d", { willReadFrequently: true });
  g.drawImage(canvas, 0, 0, size, size);
  const px = g.getImageData(0, 0, size, size).data;
  // White where the channel is 0, its own colour where it is 255.
  return [0, 1, 2].map((ch) =>
    dataTexture(size, size, (x, y) => {
      const fade = 255 - px[(y * size + x) * 4 + ch] * 0.78;
      return [0, 1, 2].map((i) => (i === ch ? 255 : fade));
    }),
  );
}

// Vector values as a column of cells (long vectors are averaged down to 64 cells).
function vectorTexture(values, cells) {
  const n = values.length;
  const k = Math.min(n, cells);
  const per = n / k;
  const agg = Array.from({ length: k }, (_, i) => {
    let s = 0;
    const a = Math.floor(i * per),
      b = Math.max(a + 1, Math.floor((i + 1) * per));
    for (let j = a; j < b; j++) s += Math.abs(values[j]);
    return s / (b - a);
  });
  const max = Math.max(1e-6, ...agg);
  return dataTexture(1, k, (_, y) => ramp(agg[y] / max));
}

let gridCanvas = null;
function gridTexture(cx, cy, dark = false) {
  if (!gridCanvas) {
    gridCanvas = {};
    for (const mode of ["light", "dark"]) {
      const c = document.createElement("canvas");
      c.width = c.height = 32;
      const g = c.getContext("2d");
      g.fillStyle = mode === "dark" ? "#3b4650" : "#eef3f1";
      g.fillRect(0, 0, 32, 32);
      g.strokeStyle = mode === "dark" ? "#56636d" : "#c9d3cf";
      g.lineWidth = 2;
      g.strokeRect(0, 0, 32, 32);
      gridCanvas[mode] = c;
    }
  }
  const t = new THREE.CanvasTexture(gridCanvas[dark ? "dark" : "light"]);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(cx, cy);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function label(html, cls = "") {
  const div = document.createElement("div");
  div.className = `t3-label ${cls}`;
  div.innerHTML = html;
  return new CSS2DObject(div);
}

const esc = (s) => String(s).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]);

// ── Pieces ───────────────────────────────────────────────

function imageTensor(item) {
  const [h, w, c] = item.shape;
  const W = Math.min(L(w), CAP),
    H = Math.min(L(h), CAP),
    D = Math.min(L(c), CAP);
  const g = new THREE.Group();
  const shell = new THREE.Mesh(
    new THREE.BoxGeometry(W, H, D),
    new THREE.MeshBasicMaterial({ color: C.face, transparent: true, opacity: 0.22, depthWrite: false }),
  );
  g.add(shell);

  let textures = null;
  if (item.maps?.length) textures = item.maps.map((m) => mapTexture(m, w, h));
  else if (item.input) textures = rgbTextures(item.input, Math.min(w, 128));
  const k = textures ? textures.length : Math.min(c, 4);
  for (let i = 0; i < k; i++) {
    const mat = textures
      ? new THREE.MeshBasicMaterial({ map: textures[i], side: THREE.DoubleSide })
      : new THREE.MeshBasicMaterial({ map: gridTexture(Math.min(w, 28), Math.min(h, 28)), side: THREE.DoubleSide });
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(W, H), mat);
    plane.position.z = k === 1 ? D / 2 - 0.05 : D / 2 - 0.05 - (i * (D - 0.1)) / (k - 1 || 1);
    g.add(plane);
  }
  if (k > 1 && c > k) {
    // A faint line for every channel we didn't draw, so depth still reads as "many slices".
    const lines = [];
    const n = Math.min(c, 48);
    for (let i = 1; i < n; i++) {
      const z = D / 2 - (i * D) / n;
      lines.push(-W / 2, H / 2, z, W / 2, H / 2, z, W / 2, H / 2, z, W / 2, -H / 2, z);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(lines, 3));
    g.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: C.edge, transparent: true, opacity: 0.35 })));
  }
  return { group: g, size: new THREE.Vector3(W, H, D), shell };
}

function vectorTensor(item) {
  const n = item.shape[0];
  const H = Math.min(L(n) * 1.15, CAP),
    W = 2.6,
    D = 2.6;
  const front = item.values ? vectorTexture(item.values, 64) : gridTexture(1, Math.min(n, 40));
  const side = new THREE.MeshBasicMaterial({ color: C.face });
  const mats = [side, side, side, side, new THREE.MeshBasicMaterial({ map: front }), side];
  const shell = new THREE.Mesh(new THREE.BoxGeometry(W, H, D), mats);
  const g = new THREE.Group();
  g.add(shell);
  if (L(n) * 1.15 > CAP) g.add(breakMark(W, H));
  return { group: g, size: new THREE.Vector3(W, H, D), shell };
}

function outputTensor(item) {
  const n = item.shape[0];
  const cell = 3.2,
    W = 3.2,
    D = 3.2,
    H = cell * n;
  const g = new THREE.Group();
  const probs = item.values;
  const tex = dataTexture(1, n, (_, y) => (probs ? ramp(probs[y]) : [238, 243, 241]));
  const side = new THREE.MeshBasicMaterial({ color: C.face });
  const shell = new THREE.Mesh(new THREE.BoxGeometry(W, H, D), [side, side, side, side, new THREE.MeshBasicMaterial({ map: tex }), side]);
  g.add(shell);
  (item.classNames || []).forEach((name, i) => {
    const pct = probs ? ` ${Math.round(probs[i] * 100)}%` : "";
    const l = label(`${esc(name)}<span>${pct}</span>`, "t3-class");
    l.position.set(W / 2 + 1, H / 2 - cell * (i + 0.5), D / 2);
    l.center.set(0, 0.5);
    g.add(l);
  });
  return { group: g, size: new THREE.Vector3(W, H, D), shell };
}

function ghostTensor(item) {
  const s = 7;
  const geo = new THREE.BoxGeometry(s, s, s);
  const shell = new THREE.Mesh(
    geo,
    new THREE.MeshBasicMaterial({ color: item.error ? 0xf6dcd9 : C.face, transparent: true, opacity: 0.35, depthWrite: false }),
  );
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geo),
    new THREE.LineDashedMaterial({ color: item.error ? C.error : C.edge, dashSize: 0.8, gapSize: 0.6 }),
  );
  edges.computeLineDistances();
  const g = new THREE.Group();
  g.add(shell, edges);
  return { group: g, size: new THREE.Vector3(s, s, s), shell, fixedEdges: true };
}

function frozenBlock(name) {
  const W = 16,
    H = 18,
    D = 18;
  const side = new THREE.MeshBasicMaterial({ map: gridTexture(10, 11, true) });
  const top = new THREE.MeshBasicMaterial({ map: gridTexture(10, 11, true) });
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(W, H, D), [side, side, top, top, side, side]);
  const g = new THREE.Group();
  g.add(mesh);
  g.add(new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry), new THREE.LineBasicMaterial({ color: 0x1e262c })));
  const l = label(`<b>${esc(name)}</b><span>사전학습된 가중치, 고정</span>`, "t3-frozen");
  l.position.set(0, H / 2 + 2.5, 0);
  l.center.set(0.5, 1);
  g.add(l);
  return { group: g, size: new THREE.Vector3(W, H, D) };
}

function breakMark(W, H) {
  const pts = [-W, H * 0.1, 0, W, H * 0.16, 0, -W, H * 0.04, 0, W, H * 0.1, 0].map((v, i) => (i % 3 === 2 ? 1.4 : v));
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  return new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: C.edge }));
}

function arrow(x1, x2, error) {
  const color = error ? C.error : C.link;
  const g = new THREE.Group();
  const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x1, 0, 0), new THREE.Vector3(x2 - 1.4, 0, 0)]);
  g.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color })));
  const head = new THREE.Mesh(new THREE.ConeGeometry(0.7, 1.6, 12), new THREE.MeshBasicMaterial({ color }));
  head.rotation.z = -Math.PI / 2;
  head.position.x = x2 - 0.8;
  g.add(head);
  return g;
}

// ── Scene ────────────────────────────────────────────────

export function createScene(el, { onPick } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  el.appendChild(renderer.domElement);

  const labels = new CSS2DRenderer();
  labels.domElement.className = "t3-labels";
  el.appendChild(labels.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(C.page);
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 6000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.screenSpacePanning = true;

  let model = new THREE.Group();
  scene.add(model);
  let items = []; // { group, shell, size, center, edges, labelEl, index }
  let bounds = new THREE.Box3();
  let tween = null;
  let focused; // last focus() target, re-framed when the stage changes size
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function resize() {
    const w = el.clientWidth,
      h = el.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h);
    labels.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    if (focused !== undefined && items.length) focus(focused, false);
  }
  const ro = new ResizeObserver(resize);
  ro.observe(el);
  resize();

  function disposeTree(obj) {
    obj.traverse((o) => {
      o.geometry?.dispose();
      const mats = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
      for (const m of mats) {
        m.map?.dispose();
        m.dispose();
      }
      if (o.isCSS2DObject) o.element.remove();
    });
  }

  function setModel(spec) {
    scene.remove(model);
    disposeTree(model);
    model = new THREE.Group();
    items = [];
    let x = 0;
    let maxH = 0;

    spec.items.forEach((item, index) => {
      if (item.frozen) {
        const fb = frozenBlock(item.frozen);
        if (index > 0) model.add(arrow(x, x + GAP, false));
        fb.group.position.x = x + GAP + fb.size.x / 2;
        model.add(fb.group);
        x += GAP + fb.size.x;
        maxH = Math.max(maxH, fb.size.y);
      }
      const piece =
        item.kind === "image"
          ? imageTensor(item)
          : item.kind === "vector"
            ? vectorTensor(item)
            : item.kind === "output"
              ? outputTensor(item)
              : ghostTensor(item);
      if (index > 0) model.add(arrow(x, x + GAP, item.error));
      const cx = x + GAP * (index > 0 ? 1 : 0) + piece.size.x / 2;
      piece.group.position.x = cx;
      model.add(piece.group);

      let edges = null;
      if (!piece.fixedEdges) {
        edges = new THREE.LineSegments(
          new THREE.EdgesGeometry(new THREE.BoxGeometry(piece.size.x, piece.size.y, piece.size.z)),
          new THREE.LineBasicMaterial({ color: C.edge }),
        );
        piece.group.add(edges);
      }
      const l = label(`<b>${esc(item.name)}</b><span>${esc(item.shapeText)}</span>`, item.error ? "t3-error" : "");
      l.position.set(0, -piece.size.y / 2 - 1.5, piece.size.z / 2);
      l.center.set(0.5, 0);
      piece.group.add(l);

      piece.shell.userData.index = index;
      items.push({ ...piece, edges, labelEl: l.element, center: new THREE.Vector3(cx, 0, 0), index });
      x = cx + piece.size.x / 2;
      maxH = Math.max(maxH, piece.size.y);
    });

    // Graph-paper floor under the model.
    const span = Math.max(x, 60);
    const floor = new THREE.GridHelper(Math.ceil((span + 60) / 4) * 4, Math.ceil((span + 60) / 4), C.floor, C.floorMinor);
    floor.position.set(x / 2, -maxH / 2 - 6, 0);
    model.add(floor);

    scene.add(model);
    bounds = new THREE.Box3();
    items.forEach((it) => bounds.expandByObject(it.group));
  }

  function frame(center, radius, animate = true) {
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const fit = Math.max(radius / Math.tan(fov / 2), radius / Math.tan(fov / 2) / Math.max(camera.aspect, 0.6));
    const to = center.clone().add(VIEW_DIR.clone().multiplyScalar(fit * 1.05 + 6));
    if (!animate || reduceMotion) {
      camera.position.copy(to);
      controls.target.copy(center);
      tween = null;
      return;
    }
    tween = { t0: performance.now(), dur: 750, fromPos: camera.position.clone(), toPos: to, fromTarget: controls.target.clone(), toTarget: center.clone() };
  }

  function focus(index, animate = true) {
    focused = index;
    items.forEach((it) => {
      const on = index === it.index;
      if (it.edges) it.edges.material.color.setHex(on ? C.edgeOn : C.edge);
      it.labelEl.classList.toggle("on", on);
    });
    if (index == null || !items[index]) {
      if (bounds.isEmpty()) return;
      const center = bounds.getCenter(new THREE.Vector3());
      const size = bounds.getSize(new THREE.Vector3());
      frame(center, Math.max(size.x * 0.5, size.y * 0.75), animate);
      return;
    }
    const it = items[index];
    frame(it.center.clone(), Math.max(it.size.x, it.size.y, it.size.z * 0.6) * 0.95 + 4, animate);
  }

  // Click a block to jump the tour to it (ignore drags).
  const ray = new THREE.Raycaster();
  let down = null;
  const onDown = (e) => (down = { x: e.clientX, y: e.clientY });
  const onUp = (e) => {
    if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
    const r = renderer.domElement.getBoundingClientRect();
    const p = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(p, camera);
    const hit = ray.intersectObjects(
      items.map((it) => it.shell),
      false,
    )[0];
    if (hit) onPick?.(hit.object.userData.index);
  };
  renderer.domElement.addEventListener("pointerdown", onDown);
  renderer.domElement.addEventListener("pointerup", onUp);

  let raf = 0;
  const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  function loop(now) {
    raf = requestAnimationFrame(loop);
    if (tween) {
      const t = Math.min(1, (now - tween.t0) / tween.dur);
      const e = ease(t);
      camera.position.lerpVectors(tween.fromPos, tween.toPos, e);
      controls.target.lerpVectors(tween.fromTarget, tween.toTarget, e);
      if (t >= 1) tween = null;
    }
    controls.update();
    renderer.render(scene, camera);
    labels.render(scene, camera);
  }
  raf = requestAnimationFrame(loop);

  return {
    setModel,
    focus,
    dispose() {
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.domElement.removeEventListener("pointerdown", onDown);
      renderer.domElement.removeEventListener("pointerup", onUp);
      controls.dispose();
      disposeTree(scene);
      renderer.dispose();
      el.innerHTML = "";
    },
  };
}
