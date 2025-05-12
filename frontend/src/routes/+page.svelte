<script>
  import { onMount, onDestroy } from 'svelte';
  import VideoUpload from '../components/VideoUpload.svelte';

  // State variables
  let activeTab = 'camera'; // 'camera' or 'upload'
  let videoElement;
  let canvasElement;
  let processingElement;
  let processingContext;
  let wsConnection = null;
  let isConnected = false;
  let isDetecting = false;
  let fps = 0;
  let frameCount = 0;
  let lastTime = Date.now();
  let processingTime = 0;
  let detections = [];
  let errorMessage = null;
  let videoDevices = [];
  let currentDeviceId = '';
  let loadingCamera = false;

  // Configuration
  const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8081/ws';
  const FPS_INTERVAL = 1000; // Update FPS every second
  
  // Switch between camera and upload tabs
  function switchTab(tab) {
    activeTab = tab;
    
    // If switching away from camera, stop detection and disconnect
    if (tab !== 'camera' && isDetecting) {
      toggleDetection();
    }
  }

  // Setup the WebSocket connection
  function setupWebSocket() {
    if (wsConnection) {
      wsConnection.close();
    }

    try {
      errorMessage = null;
      
      // Log connection attempt
      console.log(`Attempting WebSocket connection to ${WS_URL}`);
      wsConnection = new WebSocket(WS_URL);
      
      // Add a timeout for connection
      const connectionTimeout = setTimeout(() => {
        if (wsConnection && wsConnection.readyState !== WebSocket.OPEN) {
          errorMessage = `WebSocket connection timeout. The server at ${WS_URL} is not responding.`;
          wsConnection.close();
        }
      }, 5000);
      
      wsConnection.onopen = () => {
        clearTimeout(connectionTimeout);
        isConnected = true;
        errorMessage = null;
        console.log('WebSocket connection established');
      };
      
      wsConnection.onclose = (event) => {
        clearTimeout(connectionTimeout);
        isConnected = false;
        isDetecting = false;
        console.log('WebSocket connection closed:', event.code, event.reason);
        
        // Try to reconnect after a delay if the detection is active
        if (isDetecting) {
          errorMessage = "Connection lost. Attempting to reconnect...";
          setTimeout(setupWebSocket, 3000);
        }
      };
      
      wsConnection.onerror = (error) => {
        clearTimeout(connectionTimeout);
        errorMessage = `WebSocket connection error. Check if the backend server is running at ${WS_URL.split('/')[2]}.`;
        console.error('WebSocket error:', error);
      };
      
      wsConnection.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          // Handle error message from the server
          if (data.error) {
            errorMessage = `Server error: ${data.error}`;
            return;
          }
          
          // Update processing time
          processingTime = data.processing_time ? Math.round(data.processing_time * 1000) : 0;
          
          // Update detections data
          detections = data.detections || [];
          
          // Display the processed frame
          if (data.processed_frame) {
            const img = new Image();
            img.onload = () => {
              processingContext.clearRect(0, 0, processingElement.width, processingElement.height);
              processingContext.drawImage(img, 0, 0, processingElement.width, processingElement.height);
            };
            img.src = data.processed_frame;
          }
          
          // If we're still detecting, send the next frame
          if (isDetecting) {
            captureAndSendFrame();
          }
          
        } catch (err) {
          console.error('Error processing WebSocket message:', err);
        }
      };
    } catch (err) {
      errorMessage = `Failed to connect to WebSocket server: ${err.message}`;
      console.error('WebSocket setup error:', err);
    }
  }

  // Enumerate available video devices
  async function enumerateVideoDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      videoDevices = devices.filter(device => device.kind === 'videoinput');
      console.log('Available video devices:', videoDevices);
      
      // Set default device if not already set
      if (!currentDeviceId && videoDevices.length > 0) {
        currentDeviceId = videoDevices[0].deviceId;
      }
      
      return videoDevices;
    } catch (err) {
      console.error('Error enumerating video devices:', err);
      return [];
    }
  }
  
  // Initialize video stream from the webcam
  async function initializeCamera(deviceId = null) {
    try {
      loadingCamera = true;
      
      // If a current stream exists, stop it
      if (videoElement && videoElement.srcObject) {
        const tracks = videoElement.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }
      
      // Get available devices if not already done
      if (videoDevices.length === 0) {
        await enumerateVideoDevices();
      }
      
      const constraints = { 
        video: {
          width: 640,
          height: 480
        }
      };
      
      // If specific device ID is provided, use it
      if (deviceId) {
        constraints.video.deviceId = { exact: deviceId };
        currentDeviceId = deviceId;
      } else if (currentDeviceId) {
        constraints.video.deviceId = { exact: currentDeviceId };
      }
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      videoElement.srcObject = stream;
      videoElement.play();
      
      // Initialize the processing canvas
      processingContext = processingElement.getContext('2d');
      
      errorMessage = null;
      loadingCamera = false;
    } catch (err) {
      errorMessage = `Camera access error: ${err.message}`;
      console.error('Camera initialization error:', err);
      loadingCamera = false;
    }
  }
  
  // Switch to next available camera
  async function switchCamera() {
    if (loadingCamera) return;
    
    try {
      // If no device list is available yet, enumerate devices
      if (videoDevices.length === 0) {
        await enumerateVideoDevices();
      }
      
      if (videoDevices.length <= 1) {
        errorMessage = "No additional cameras found on this device.";
        return;
      }
      
      // Find the index of the current device
      const currentIndex = videoDevices.findIndex(device => device.deviceId === currentDeviceId);
      
      // Calculate the next device index
      const nextIndex = (currentIndex + 1) % videoDevices.length;
      const nextDeviceId = videoDevices[nextIndex].deviceId;
      
      // Initialize with the next camera
      await initializeCamera(nextDeviceId);
      
      console.log(`Switched to camera: ${videoDevices[nextIndex].label || 'Camera ' + (nextIndex + 1)}`);
    } catch (err) {
      errorMessage = `Failed to switch camera: ${err.message}`;
      console.error('Camera switching error:', err);
    }
  }

  // Capture a frame from video and send to the WebSocket
  function captureAndSendFrame() {
    if (!isConnected || !videoElement || !canvasElement || !wsConnection) return;
    
    try {
      const context = canvasElement.getContext('2d');
      if (!context) return;
      
      // Draw the current video frame on the canvas
      context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
      
      // Convert the canvas to a data URL (JPEG format with 0.8 quality)
      const imageData = canvasElement.toDataURL('image/jpeg', 0.8);
      
      // Send the image data to the server
      wsConnection.send(imageData);
      
      // Update frame count for FPS calculation
      frameCount++;
      const now = Date.now();
      if (now - lastTime >= FPS_INTERVAL) {
        fps = Math.round((frameCount * 1000) / (now - lastTime));
        frameCount = 0;
        lastTime = now;
      }
    } catch (err) {
      console.error('Error capturing/sending frame:', err);
    }
  }

  // Toggle detection on/off
  function toggleDetection() {
    isDetecting = !isDetecting;
    
    if (isDetecting) {
      if (!isConnected) {
        setupWebSocket();
      }
      captureAndSendFrame();
    }
  }

  // Component lifecycle
  onMount(async () => {
    if (activeTab === 'camera') {
      // First enumerate available devices
      await enumerateVideoDevices();
      // Then initialize camera with the first device
      await initializeCamera();
      setupWebSocket();
    }
  });

  onDestroy(() => {
    isDetecting = false;
    if (wsConnection) {
      wsConnection.close();
    }
    // Stop video stream if exists
    if (videoElement && videoElement.srcObject) {
      const tracks = videoElement.srcObject.getTracks();
      tracks.forEach(track => track.stop());
    }
  });
</script>

<svelte:head>
  <title>Real-Time Object Detection</title>
  <meta name="description" content="Real-time object detection with camera and video upload capabilities" />
</svelte:head>

<main>
  <h1>Real-Time Object Detection</h1>
  
  <div class="tabs">
    <button 
      class:active={activeTab === 'camera'} 
      on:click={() => switchTab('camera')}
    >
      Camera Detection
    </button>
    <button 
      class:active={activeTab === 'upload'} 
      on:click={() => switchTab('upload')}
    >
      Video Upload
    </button>
  </div>
  
  {#if activeTab === 'camera'}
    <div class="status-info">
      <div class="status-item">
        <span class="label">Connection:</span>
        <span class="value" class:connected={isConnected} class:disconnected={!isConnected}>
          {isConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>
      
      <div class="status-item">
        <span class="label">Detection:</span>
        <span class="value" class:running={isDetecting} class:stopped={!isDetecting}>
          {isDetecting ? 'Running' : 'Stopped'}
        </span>
      </div>
      
      <div class="status-item">
        <span class="label">FPS:</span>
        <span class="value">{fps}</span>
      </div>
      
      <div class="status-item">
        <span class="label">Processing:</span>
        <span class="value">{processingTime} ms</span>
      </div>
    </div>

    {#if errorMessage}
      <div class="error-message">
        {errorMessage}
      </div>
    {/if}

    <div class="video-container">
      <div class="video-wrapper">
        <h3>Camera Input</h3>
        <video bind:this={videoElement} width="640" height="480" autoplay muted></video>
      </div>
      
      <div class="video-wrapper">
        <h3>Object Detection</h3>
        <canvas bind:this={processingElement} width="640" height="480"></canvas>
      </div>
    </div>

    <!-- Hidden canvas for capturing frames -->
    <canvas bind:this={canvasElement} width="640" height="480" style="display: none;"></canvas>

    <div class="controls">
      <button on:click={toggleDetection} class:active={isDetecting}>
        {isDetecting ? 'Stop Detection' : 'Start Detection'}
      </button>
      <button on:click={setupWebSocket} disabled={isConnected}>
        Reconnect
      </button>
      <button on:click={switchCamera} disabled={loadingCamera || videoDevices.length <= 1}>
        {loadingCamera ? 'Switching...' : 'Switch Camera'}
      </button>
    </div>

    <!-- Display loading indicator or current camera info -->
    {#if loadingCamera}
      <div class="info-message">Switching cameras, please wait...</div>
    {:else if videoDevices.length > 0}
      <div class="camera-info">
        Current camera: {videoDevices.find(d => d.deviceId === currentDeviceId)?.label || 'Default camera'} 
        ({videoDevices.length} {videoDevices.length === 1 ? 'camera' : 'cameras'} available)
      </div>
    {/if}

  {:else if activeTab === 'upload'}
    <VideoUpload />
  {/if}
</main>

<style>
  main {
    max-width: 1300px;
    margin: 0 auto;
    padding: 20px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
  }

  h1 {
    text-align: center;
    margin-bottom: 20px;
    color: #333;
  }
  
  .tabs {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
    border-bottom: 2px solid #e0e0e0;
  }
  
  .tabs button {
    padding: 10px 20px;
    background-color: transparent;
    border: none;
    border-bottom: 3px solid transparent;
    color: #666;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s;
    margin: 0 10px;
    min-height: 44px; /* Minimum tap target size for mobile */
  }
  
  .tabs button:hover {
    color: #2196F3;
  }
  
  .tabs button.active {
    color: #2196F3;
    border-bottom: 3px solid #2196F3;
    background-color: transparent;
  }
  
  @media (max-width: 480px) {
    .tabs {
      width: 100%;
    }
    
    .tabs button {
      flex: 1;
      padding: 12px 8px;
      margin: 0;
      font-size: 14px;
    }
  }

  .status-info {
    display: flex;
    justify-content: space-around;
    margin-bottom: 20px;
    background-color: #f5f5f5;
    padding: 10px;
    border-radius: 5px;
    flex-wrap: wrap;
  }

  .status-item {
    display: flex;
    align-items: center;
    gap: 5px;
  }

  .status-item .label {
    font-weight: bold;
  }

  .status-item .value {
    min-width: 60px;
  }

  .connected { color: green; }
  .disconnected { color: red; }
  .running { color: green; }
  .stopped { color: orange; }

  .error-message {
    background-color: #ffdddd;
    color: #990000;
    padding: 10px;
    margin-bottom: 20px;
    border-radius: 5px;
    border-left: 5px solid #990000;
  }
  
  .info-message {
    background-color: #e3f2fd;
    color: #0d47a1;
    padding: 10px;
    margin-bottom: 20px;
    border-radius: 5px;
    border-left: 5px solid #0d47a1;
    text-align: center;
  }
  
  .camera-info {
    text-align: center;
    margin-top: 10px;
    font-size: 14px;
    color: #666;
  }

  .video-container {
    display: flex;
    justify-content: space-between;
    gap: 20px;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }

  .video-wrapper {
    flex: 1;
    min-width: 320px;
    border: 1px solid #ccc;
    padding: 10px;
    border-radius: 5px;
    background-color: #f9f9f9;
  }

  .video-wrapper h3 {
    margin-top: 0;
    margin-bottom: 10px;
    text-align: center;
  }

  video, canvas {
    width: 100%;
    height: auto;
    background-color: #000;
    border-radius: 5px;
  }
  
  @media (max-width: 480px) {
    .video-wrapper {
      min-width: unset;
      padding: 8px;
      margin-bottom: 15px;
    }
    
    .video-wrapper h3 {
      font-size: 16px;
      margin-bottom: 8px;
    }
    
    .video-container {
      gap: 10px;
    }
  }

  .controls {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-bottom: 20px;
  }

  button {
    padding: 10px 20px;
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 16px;
    transition: background-color 0.3s;
  }

  button:hover {
    background-color: #45a049;
  }

  button:disabled {
    background-color: #cccccc;
    cursor: not-allowed;
  }

  button.active {
    background-color: #f44336;
  }

  .label {
    font-weight: bold;
    text-transform: capitalize;
  }


  @media (max-width: 768px) {
    .video-container {
      flex-direction: column;
    }
    
    h1 {
      font-size: 24px;
      margin-bottom: 15px;
    }
    
    .controls {
      flex-wrap: wrap;
    }
    
    .controls button {
      flex: 1 0 40%;
      margin-bottom: 10px;
      font-size: 14px;
      padding: 8px 12px;
    }
    
    .status-info {
      padding: 8px 5px;
    }
    
    .status-item {
      font-size: 14px;
      margin: 5px 0;
      flex: 1 0 45%;
      justify-content: center;
    }
  }
</style>