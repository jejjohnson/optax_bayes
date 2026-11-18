// Draw labelled connectors between elements of a laid-out diagram.
//
//   wire("a", "b", { from: "right", to: "left", label: "GeoTensor" });
//
// Positions are measured after CSS layout, so a diagram only declares its
// boxes (HTML/CSS) and the edges between them (this call). Options:
//   from / to   anchor side: "left" | "right" | "top" | "bottom"
//   label       HTML for a chip placed on the curve (optional)
//   at          0..1 position of the label along the curve (default 0.5)
//   color       stroke colour (default slate)
//   dashed      true for an optional / weak relation
//   bend        control-point distance in px (default: half the gap)
//   fromShift / toShift  slide the anchor along its side, in px
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";

function _root() {
  return document.getElementById("diagram");
}

function _svg() {
  let svg = _root().querySelector("svg.wires");
  if (!svg) {
    svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", "wires");
    svg.appendChild(document.createElementNS(SVG_NS, "defs"));
    _root().prepend(svg);
  }
  const r = _root().getBoundingClientRect();
  svg.setAttribute("width", r.width);
  svg.setAttribute("height", r.height);
  return svg;
}

function _marker(color) {
  const id = "arrow-" + color.replace(/[^a-z0-9]/gi, "");
  const svg = _svg();
  if (!svg.querySelector("#" + id)) {
    const m = document.createElementNS(SVG_NS, "marker");
    m.setAttribute("id", id);
    m.setAttribute("viewBox", "0 0 10 10");
    m.setAttribute("refX", "8.5");
    m.setAttribute("refY", "5");
    m.setAttribute("markerWidth", "7");
    m.setAttribute("markerHeight", "7");
    m.setAttribute("orient", "auto-start-reverse");
    const p = document.createElementNS(SVG_NS, "path");
    p.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    p.setAttribute("fill", color);
    m.appendChild(p);
    svg.querySelector("defs").appendChild(m);
  }
  return id;
}

function _anchor(el, side, shift) {
  const r = el.getBoundingClientRect();
  const o = _root().getBoundingClientRect();
  const x0 = r.left - o.left;
  const y0 = r.top - o.top;
  switch (side) {
    case "left":
      return [x0, y0 + r.height / 2 + shift, -1, 0];
    case "right":
      return [x0 + r.width, y0 + r.height / 2 + shift, 1, 0];
    case "top":
      return [x0 + r.width / 2 + shift, y0, 0, -1];
    default:
      return [x0 + r.width / 2 + shift, y0 + r.height, 0, 1];
  }
}

function _bezier(t, p0, p1, p2, p3) {
  const u = 1 - t;
  return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3;
}

function wire(fromId, toId, opts = {}) {
  const from = opts.from || "right";
  const to = opts.to || "left";
  const color = opts.color || "#94a3b8";
  const [x1, y1, dx1, dy1] = _anchor(document.getElementById(fromId), from, opts.fromShift || 0);
  const [x2, y2, dx2, dy2] = _anchor(document.getElementById(toId), to, opts.toShift || 0);
  const gap = Math.hypot(x2 - x1, y2 - y1);
  const bend = opts.bend ?? Math.max(28, gap * 0.45);
  const c1 = [x1 + dx1 * bend, y1 + dy1 * bend];
  // A small stand-off so the arrowhead does not touch the border.
  const end = [x2 + dx2 * 3, y2 + dy2 * 3];
  const c2 = [end[0] + dx2 * bend, end[1] + dy2 * bend];

  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute(
    "d",
    `M ${x1} ${y1} C ${c1[0]} ${c1[1]}, ${c2[0]} ${c2[1]}, ${end[0]} ${end[1]}`,
  );
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", color);
  path.setAttribute("stroke-width", opts.width || 2);
  path.setAttribute("stroke-linecap", "round");
  if (opts.dashed) path.setAttribute("stroke-dasharray", "6 6");
  path.setAttribute("marker-end", `url(#${_marker(color)})`);
  _svg().appendChild(path);

  if (opts.label) {
    const t = opts.at ?? 0.5;
    const lx = _bezier(t, x1, c1[0], c2[0], end[0]);
    const ly = _bezier(t, y1, c1[1], c2[1], end[1]);
    const chip = document.createElement("div");
    chip.className = "wire-label";
    chip.innerHTML = `<span class="chip">${opts.label}</span>`;
    chip.style.left = `${lx + (opts.dx || 0)}px`;
    chip.style.top = `${ly + (opts.dy || 0)}px`;
    _root().appendChild(chip);
  }
}

// Diagrams call `ready()` after their last `wire(...)`; render.py waits
// for this flag before taking the screenshot.
function ready() {
  document.body.dataset.ready = "1";
}
