import React, { useEffect, useState, useRef } from "react";
import CameraSelector from "../components/CameraSelector";
import createSocket from "../api";

const App = () => {
  const [image, setImage] = useState(null);
  const [detections, setDetections] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState("");
  const [socket, setSocket] = useState(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    if (selectedCamera) {
      console.log("Selected camera:", selectedCamera);
      const newSocket = createSocket(selectedCamera);
      setSocket(newSocket);
    }

    return () => {
      if (socket) {
        socket.close();
      }
    };
  }, [selectedCamera]);

  useEffect(() => {
    if (socket) {
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setImage(`data:image/jpeg;base64,${data.image}`);
        setDetections(data.detections);
      };
    }
  }, [socket]);

  useEffect(() => {
    if (image && detections.length > 0) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      const img = new Image();
      img.src = image;
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0, img.width, img.height);

        // Draw bounding boxes
        detections.forEach(([x1, y1, x2, y2]) => {
          ctx.strokeStyle = "red";
          ctx.lineWidth = 2;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        });
      };
    }
  }, [image, detections]);

  return (
    <div className="container">
      <h1>Real-Time Object Detection</h1>
      <CameraSelector onSelectCamera={setSelectedCamera} />
      <canvas ref={canvasRef}></canvas>
    </div>
  );
};

export default App;
