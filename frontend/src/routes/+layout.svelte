<script>
  // This layout wraps all pages in the application
</script>

<slot />

<style>
  :global(body) {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    background-color: #fafafa;
  }
  
  :global(*) {
    box-sizing: border-box;
  }
</style>