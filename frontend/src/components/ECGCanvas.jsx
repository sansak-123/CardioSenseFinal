import React, { useRef, useEffect, useCallback } from "react";

/**
 * ECGCanvas — Three.js live ECG line renderer
 * Draws a realistic PQRST waveform that scrolls continuously.
 * Pure WebGL via Three.js BufferGeometry — no R3F overhead.
 */
export default function ECGCanvas({ className = "", height = 120, color = "#38BDF8", risk = 0 }) {
  const mountRef = useRef(null);
  const stateRef = useRef({});

  // Generate one PQRST beat (normalised 0→1 x, -1→1 y)
  const generateBeat = useCallback((offset = 0, amplitude = 1) => {
    const pts = [];
    const push = (x, y) => pts.push(offset + x, y * amplitude, 0);

    // Baseline
    for (let x = 0; x < 0.12; x += 0.01) push(x, 0);
    // P wave (small bump)
    for (let x = 0.12; x < 0.22; x += 0.005) {
      push(x, 0.18 * Math.sin(Math.PI * (x - 0.12) / 0.10));
    }
    // PR segment
    for (let x = 0.22; x < 0.32; x += 0.01) push(x, 0);
    // Q (small dip)
    push(0.32, 0); push(0.34, -0.18); push(0.36, 0);
    // R (sharp spike — the QRS)
    push(0.37, 0); push(0.38, 1.0); push(0.39, 0);
    // S (dip below)
    push(0.40, -0.25); push(0.42, 0);
    // ST segment
    for (let x = 0.42; x < 0.52; x += 0.01) push(x, 0.04);
    // T wave (rounded hump)
    for (let x = 0.52; x < 0.72; x += 0.005) {
      push(x, 0.32 * Math.sin(Math.PI * (x - 0.52) / 0.20));
    }
    // Return to baseline
    for (let x = 0.72; x < 1.0; x += 0.01) push(x, 0);

    return pts;
  }, []);

  useEffect(() => {
    const THREE = window.__THREE__ || null;
    let animId;
    const el = mountRef.current;
    if (!el) return;

    // Lazy-load Three.js from CDN if not bundled
    const init = (THREE) => {
      const W = el.clientWidth;
      const H = height;

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(W, H);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setClearColor(0x000000, 0);
      el.appendChild(renderer.domElement);

      const scene = new THREE.Scene();
      const camera = new THREE.OrthographicCamera(0, W, H, 0, -1, 1);

      // Determine colour based on risk
      const lineColor = risk > 0.7 ? "#E11D48" : risk > 0.4 ? "#F59E0B" : color;
      const mat = new THREE.LineBasicMaterial({
        color: new THREE.Color(lineColor),
        linewidth: 1.5,
      });

      // Number of beats that fit in the canvas
      const BEAT_W_PX = W * 0.22;
      const numBeats = Math.ceil(W / BEAT_W_PX) + 2;
      const allPoints = [];
      for (let b = 0; b < numBeats; b++) {
        const amp = 0.9 + Math.random() * 0.15;
        allPoints.push(...generateBeat(b, amp));
      }

      const totalPts = allPoints.length / 3;
      const positions = new Float32Array(allPoints);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));

      const line = new THREE.Line(geo, mat);
      scene.add(line);

      // Scale to canvas
      const xScale = W / numBeats;
      const yScale = H * 0.35;
      const yMid = H * 0.5;
      line.scale.set(xScale, yScale, 1);
      line.position.set(0, yMid, 0);

      let offset = 0;
      const speed = 0.003;

      const animate = () => {
        animId = requestAnimationFrame(animate);
        offset -= speed;
        if (offset < -1) offset += 1;
        line.position.x = offset * xScale;
        renderer.render(scene, camera);
      };
      animate();

      // Handle resize
      const onResize = () => {
        const nW = el.clientWidth;
        renderer.setSize(nW, H);
        camera.right = nW;
        camera.updateProjectionMatrix();
        line.scale.setX(nW / numBeats);
      };
      window.addEventListener("resize", onResize);

      stateRef.current = { renderer, scene, camera, line, mat, onResize };
    };

    // Try import from bundled three first, else CDN
    import("three").then((THREE) => {
      init(THREE);
    }).catch(() => {
      const script = document.createElement("script");
      script.src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";
      script.onload = () => init(window.THREE);
      document.head.appendChild(script);
    });

    return () => {
      cancelAnimationFrame(animId);
      const { renderer, onResize, scene } = stateRef.current;
      if (onResize) window.removeEventListener("resize", onResize);
      if (renderer) {
        renderer.dispose();
        if (renderer.domElement.parentNode === el) {
          el.removeChild(renderer.domElement);
        }
      }
    };
  }, [color, risk, height, generateBeat]);

  return <div ref={mountRef} className={className} style={{ height }} />;
}