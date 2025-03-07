// filepath: /home/sci/WSL_Space/real-time-object-detection/frontend/components/CameraSelector.js
import React, { useEffect, useState } from "react";

const CameraSelector = ({ onSelectCamera }) => {
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState("");

  useEffect(() => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      console.log("enumerateDevices() not supported.");
      return;
    }

    console.log("Calling enumerateDevices()");
    navigator.mediaDevices
      .enumerateDevices()
      .then((devices) => {
        console.log("enumerateDevices() returned:", devices);
        const videoDevices = devices.filter((device) => {
          console.log("Device:", device);
          return device.kind === "videoinput";
        });
        setCameras(videoDevices);
      })
      .catch((err) => {
        console.error(`${err.name}: ${err.message}`);
      });
  }, []);

  const handleChange = (event) => {
    const cameraId = event.target.value;
    setSelectedCamera(cameraId);
    onSelectCamera(cameraId);
  };

  const videoDevices = cameras.filter(camera => camera.kind === 'videoinput');

  return (
    <div>
      <label htmlFor="camera">Select Camera: </label>
      {videoDevices.length > 0 ? (
        <select id="camera" onChange={handleChange} value={selectedCamera}>
          {videoDevices.map((camera) => (
            <option key={camera.deviceId} value={camera.deviceId}>
              {camera.label || `Camera ${camera.deviceId}`}
            </option>
          ))}
        </select>
      ) : (
        <p>No cameras found.</p>
      )}
    </div>
  );
};

export default CameraSelector;
