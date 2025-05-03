<script>
  import { onMount } from 'svelte';

  // API URL (fallback to localhost if not provided)
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

  // State variables
  let uploading = false;
  let uploadProgress = 0;
  let videoId = null;
  let videoStatus = null;
  let processingProgress = 0;
  let errorMessage = null;
  let videoUrl = null;
  let thumbnailUrl = null;
  let selectedFile = null;
  let statusCheckInterval = null;

  // Handle file selection
  function handleFileSelect(event) {
    const input = event.target;
    if (input.files && input.files.length > 0) {
      selectedFile = input.files[0];
      
      // Reset state
      videoId = null;
      videoStatus = null;
      processingProgress = 0;
      videoUrl = null;
      thumbnailUrl = null;
      errorMessage = null;
    }
  }

  // Handle video upload
  async function uploadVideo() {
    if (!selectedFile) {
      errorMessage = "Please select a video file first";
      return;
    }

    // Check if it's a video file
    if (!selectedFile.type.startsWith('video/')) {
      errorMessage = "Selected file is not a video";
      return;
    }

    try {
      uploading = true;
      uploadProgress = 0;
      errorMessage = null;

      // Create FormData
      const formData = new FormData();
      formData.append('file', selectedFile);

      // Upload video using Fetch API with progress tracking
      const response = await fetch(`${API_URL}/video/upload/`, {
        method: 'POST',
        body: formData
      }).catch(err => {
        console.error("Fetch error:", err);
        throw new Error(`Network error: ${err.message}. Make sure the backend server is running at ${API_URL}`);
      });

      if (!response.ok) {
        const errorData = await response.json().catch(e => ({ detail: `HTTP error ${response.status}: ${response.statusText}` }));
        throw new Error(errorData.detail || `HTTP error ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      videoId = data.video_id;
      videoStatus = data.status;

      // Start checking the processing status
      startStatusCheck();
    } catch (error) {
      errorMessage = error.message || "Failed to upload video";
      console.error("Upload error:", error);
    } finally {
      uploading = false;
    }
  }

  // Check processing status periodically
  function startStatusCheck() {
    if (!videoId) return;

    // Clear any existing interval
    if (statusCheckInterval) {
      clearInterval(statusCheckInterval);
    }

    statusCheckInterval = setInterval(async () => {
      try {
        if (!videoId) {
          clearInterval(statusCheckInterval);
          return;
        }

        const response = await fetch(`${API_URL}/video/status/${videoId}`);
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Status check failed');
        }

        const data = await response.json();
        videoStatus = data.status;
        processingProgress = data.progress || 0;

        // If processing completed or failed, check result and stop polling
        if (videoStatus === 'completed' || videoStatus === 'failed') {
          if (videoStatus === 'completed') {
            await getVideoResult();
          } else if (videoStatus === 'failed' && data.error) {
            errorMessage = `Processing failed: ${data.error}`;
          }
          clearInterval(statusCheckInterval);
          statusCheckInterval = null;
        }
      } catch (error) {
        errorMessage = error.message || "Failed to check video status";
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
      }
    }, 3000); // Check every 3 seconds
  }

  // Get video result when processing is complete
  async function getVideoResult() {
    if (!videoId) return;

    try {
      const response = await fetch(`${API_URL}/video/result/${videoId}`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get video result');
      }

      const data = await response.json();
      if (data.status === 'completed') {
        videoUrl = `${API_URL}${data.video_url}`;
        thumbnailUrl = `${API_URL}${data.thumbnail_url}`;
      }
    } catch (error) {
      errorMessage = error.message || "Failed to get video result";
    }
  }

  // Clean up on component destruction
  onMount(() => {
    return () => {
      if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
      }
    };
  });
</script>

<div class="video-upload-container">
  <h2>Video Processing</h2>
  
  {#if !videoId || (videoId && videoStatus === 'failed')}
    <div class="upload-section">
      <label for="video-upload" class="file-input-label">
        {selectedFile ? selectedFile.name : 'Select Video File'}
      </label>
      <input 
        id="video-upload" 
        type="file" 
        accept="video/*"
        on:change={handleFileSelect} 
      />
      
      {#if selectedFile}
        <div class="file-info">
          <p>Selected file: {selectedFile.name}</p>
          <p>Size: {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</p>
        </div>
        
        <button 
          on:click={uploadVideo} 
          disabled={uploading}
          class="upload-button"
        >
          {uploading ? 'Uploading...' : 'Upload & Process Video'}
        </button>
      {/if}
    </div>
  {/if}
  
  {#if errorMessage}
    <div class="error-message">
      {errorMessage}
    </div>
  {/if}
  
  {#if videoId && videoStatus}
    <div class="processing-info">
      <h3>Processing Status</h3>
      <p class="status">Status: <span class={videoStatus}>{videoStatus}</span></p>
      
      {#if videoStatus === 'processing'}
        <div class="progress-bar">
          <div class="progress-fill" style="width: {processingProgress}%"></div>
        </div>
        <p class="progress-text">{processingProgress}% complete</p>
      {/if}
    </div>
  {/if}
  
  {#if videoUrl && thumbnailUrl}
    <div class="result-container">
      <h3>Processing Complete</h3>
      
      <div class="video-preview">
        <video controls src={videoUrl} poster={thumbnailUrl}>
          Your browser does not support the video tag.
        </video>
      </div>
      
      <div class="video-actions">
        <a href={videoUrl} download class="download-button">Download Processed Video</a>
      </div>
    </div>
  {/if}
</div>

<style>
  .video-upload-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f9f9f9;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
  }
  
  h2 {
    text-align: center;
    color: #333;
    margin-top: 0;
    margin-bottom: 20px;
  }
  
  .upload-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 15px;
    margin-bottom: 20px;
  }
  
  input[type="file"] {
    display: none;
  }
  
  .file-input-label {
    display: inline-block;
    padding: 12px 20px;
    background-color: #4CAF50;
    color: white;
    border-radius: 4px;
    cursor: pointer;
    font-weight: bold;
    text-align: center;
    transition: background-color 0.3s;
    width: 100%;
    max-width: 300px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  
  .file-input-label:hover {
    background-color: #45a049;
  }
  
  .file-info {
    background-color: #eee;
    padding: 10px 15px;
    border-radius: 4px;
    width: 100%;
    max-width: 300px;
  }
  
  .file-info p {
    margin: 5px 0;
    font-size: 14px;
  }
  
  .upload-button {
    padding: 12px 24px;
    background-color: #2196F3;
    color: white;
    border: none;
    border-radius: 4px;
    font-size: 16px;
    cursor: pointer;
    transition: background-color 0.3s;
  }
  
  .upload-button:hover:not(:disabled) {
    background-color: #0b7dda;
  }
  
  .upload-button:disabled {
    background-color: #ccc;
    cursor: not-allowed;
  }
  
  .processing-info {
    text-align: center;
    margin: 20px 0;
  }
  
  .status {
    font-weight: bold;
  }
  
  .status .uploaded,
  .status .processing {
    color: #ff9800;
  }
  
  .status .completed {
    color: #4CAF50;
  }
  
  .status .failed {
    color: #f44336;
  }
  
  .progress-bar {
    height: 20px;
    background-color: #e0e0e0;
    border-radius: 10px;
    margin: 15px 0;
    overflow: hidden;
  }
  
  .progress-fill {
    height: 100%;
    background-color: #4CAF50;
    transition: width 0.3s ease-in-out;
  }
  
  .progress-text {
    margin: 5px 0;
    font-size: 14px;
  }
  
  .result-container {
    margin-top: 20px;
  }
  
  .video-preview {
    margin: 15px 0;
  }
  
  .video-preview video {
    width: 100%;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  }
  
  .video-actions {
    display: flex;
    justify-content: center;
    margin-top: 15px;
  }
  
  .download-button {
    display: inline-block;
    padding: 10px 20px;
    background-color: #2196F3;
    color: white;
    text-decoration: none;
    border-radius: 4px;
    font-weight: bold;
    transition: background-color 0.3s;
  }
  
  .download-button:hover {
    background-color: #0b7dda;
  }
  
  .error-message {
    background-color: #ffdddd;
    color: #990000;
    padding: 10px 15px;
    border-radius: 4px;
    margin: 15px 0;
    border-left: 4px solid #990000;
  }
</style>