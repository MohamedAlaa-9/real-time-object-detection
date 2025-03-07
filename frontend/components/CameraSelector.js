// filepath: /home/sci/WSL_Space/real-time-object-detection/frontend/components/CameraSelector.js
import React, { useEffect, useState } from "react";

const CameraSelector = ({ onSelectCamera }) => {
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(null);

  useEffect(() => {
    navigator.mediaDevices.enumerateDevices().then((devices) => {
      const videoDevices = devices.filter((device) => device.kind === "videoinput");
      setCameras(videoDevices);
    });
  }, []);

  const handleChange = (event) => {
    const cameraId = event.target.value;
    setSelectedCamera(cameraId);
    onSelectCamera(cameraId);
  };

  return (
    <div>
      <label htmlFor="camera">Select Camera: </label>
      <select id="camera" onChange={handleChange} value={selectedCamera}>
        {cameras.map((camera) => (
          <option key={camera.deviceId} value={camera.deviceId}>
            {camera.label || `Camera ${camera.deviceId}`}
          </option>
        ))}
      </select>
    </div>
  );
};

export default CameraSelector;