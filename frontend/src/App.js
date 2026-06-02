import React, { useEffect, useRef, useState } from 'react';
import * as ort from "onnxruntime-web";

export default function App() {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [session, setSession] = useState(null);
  const [allScores, setAllScores] = useState([]);
  const canvasRef = useRef(null);
  const previewRef = useRef(null);

  useEffect(() => {
    ort.InferenceSession.create("/digit_model_single.onnx").then((s) => {
      setSession(s);
      console.log("Model loaded!");
    });
  }, []);

  const getPos = (e, canvas) => {
    const rect = canvas.getBoundingClientRect();  // fixed: was canvasRef
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
    };
  };

  const startDrawing = (e) => {
    const canvas = canvasRef.current;  // fixed: was canvas.current
    const ctx = canvas.getContext('2d');
    const pos = getPos(e, canvas);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    setIsDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const pos = getPos(e, canvas);
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.strokeStyle = "white";
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    predict();
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
    setConfidence(null);
    setAllScores([]);
  };

  const predict = async () => {
    if (!session) return;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      let minX = canvas.witdht, maxX = 0, minY = canvas.height, maxY = 0;
      for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
          const i = (y * canvas.width + x) * 4;
          if (data[i] > 20) {
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
        }
      }

      if (maxX < minX) return;

      const padding = 20;
      minX = Math.max(0, minX - padding);
      minY = Math.max(0, minY - padding);
      maxX = Math.min(canvas.width, maxX + padding);
      maxY = Math.min(canvas.height, maxY + padding);

      const small = document.createElement("canvas");
      small.width = 28;
      small.height = 28;
      const smallCtx = small.getContext("2d");
      smallCtx.fillStyle = "black";
      smallCtx.fillRect(0, 0, 28, 28);
      smallCtx.drawImage(
        canvas,
        minX, minY, maxX - minX, maxY - minY,
        0, 0, 28, 28
      );

      const preview = previewRef.current;
      if (preview) {
        const previewCtx = preview.getContext("2d");
        previewCtx.drawImage(small, 0, 0, preview.width, preview.height);
      }

      const smallData = smallCtx.getImageData(0, 0, 28, 28).data;
      const input = new Float32Array(28 * 28);
      for (let i = 0; i < 28 * 28; i++) {
        const gray = smallData[i * 4] / 255;
        input[i] = (gray - 0.1307) / 0.3081;
      }

      const tensor = new ort.Tensor("float32", input, [1, 1, 28, 28]);
      const results = await session.run({ input: tensor });
      const output = results.output.data;

      const expScores = Array.from(output).map(Math.exp);
      const sumExp = expScores.reduce((a, b) => a + b, 0);
      const probs = expScores.map((e) => e / sumExp);
      const maxProb = Math.max(...probs);
      const digit = probs.indexOf(maxProb);

      setPrediction(digit);
      setConfidence((maxProb * 100).toFixed(1));
      setAllScores(probs);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  return (
    <div style={styles.container}>  
      <h1 style={styles.title}>Digit Classifier</h1>
      <p style={styles.subtitle}>Draw a digit (0-9) and the neural network will predict it</p>

      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        style={styles.canvas}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
        onTouchStart={startDrawing}
        onTouchMove={draw}
        onTouchEnd={stopDrawing}
      />

      <button onClick={clearCanvas} style={styles.button}>
        Clear
      </button>

      {prediction !== null && (
        <div style={styles.result}>
          <div style={styles.digit}>{prediction}</div>
          <div style={styles.confidence}>Confidence: {confidence}%</div>
        </div>
      )}

      {allScores.length > 0 && (
        <div style={styles.bars}>
          {allScores.map((score, i) => (
            <div key={i} style={styles.barRow}>
              <span style={styles.barLabel}>{i}</span>
              <div style={styles.barTrack}>
                <div
                  style={{
                    ...styles.barFill,
                    width: `${(score * 100).toFixed(1)}%`,
                    background: i === prediction ? "#4ade80" : "#3b82f6",
                  }}
                />
              </div>
              <span style={styles.barPct}>{(score * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    minHeight: "100vh",
    background: "#111",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "40px 20px",
    fontFamily: "monospace",
    color: "white",
  },
  title: {
    fontSize: "2rem",
    marginBottom: "8px",
  },
  subtitle: {
    color: "#888",
    marginBottom: "24px",
    fontSize: "0.9rem",
  },
  canvas: {
    border: "2px solid #444",
    borderRadius: "8px",
    cursor: "crosshair",
  },
  button: {
    marginTop: "16px",
    padding: "10px 32px",
    background: "#222",
    color: "white",
    border: "1px solid #444",
    borderRadius: "6px",
    cursor: "pointer",
    fontFamily: "monospace",
    fontSize: "1rem",
  },
  result: {
    marginTop: "24px",
    textAlign: "center",
  },
  digit: {
    fontSize: "5rem",
    fontWeight: "bold",
    color: "#4ade80",
    lineHeight: 1,
  },
  confidence: {
    color: "#888",
    marginTop: "8px",
  },
  bars: {
    marginTop: "24px",
    width: "320px",
  },
  barRow: {
    display: "flex",
    alignItems: "center",
    marginBottom: "6px",
    gap: "8px",
  },
  barLabel: {
    width: "12px",
    textAlign: "right",
    fontSize: "0.85rem",
  },
  barTrack: {
    flex: 1,
    background: "#222",
    borderRadius: "4px",
    height: "18px",
    overflow: "hidden",
  },
  barFill: {
    height: "100%",
    borderRadius: "4px",
    transition: "width 0.3s ease",
  },
  barPct: {
    width: "42px",
    fontSize: "0.75rem",
    color: "#888",
  },
};