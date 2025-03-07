// filepath: /home/sci/WSL_Space/real-time-object-detection/frontend/api.js
const socket = new WebSocket("ws://localhost:8000/ws");

socket.onopen = () => {
  console.log("Connected to WebSocket server");
};

socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("Received detections:", data);
};

export default socket;