// filepath: /home/sci/WSL_Space/real-time-object-detection/frontend/api.js
let socket;

const createSocket = (cameraId) => {
  socket = new WebSocket(`ws://localhost:8000/ws?camera_id=${cameraId}`);

  socket.onopen = () => {
    console.log("Connected to WebSocket server");
    socket.send(JSON.stringify({ message: "Hello from client" }));
  };

  socket.onerror = (error) => {
    console.error("WebSocket error:", error);
  };

  socket.onclose = (event) => {
    console.log("WebSocket closed:", event);
  };

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log("Received detections:", data);
  };
  return socket;
};

export default createSocket;
