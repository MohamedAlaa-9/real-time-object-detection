<script>
  // This layout wraps all pages in the application
</script>

<svelte:head>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <meta name="theme-color" content="#4CAF50" />
  <meta name="apple-mobile-web-app-capable" content="yes" />
  <meta name="mobile-web-app-capable" content="yes" />
</svelte:head>

<slot />

<style>
  :global(body) {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    background-color: #fafafa;
    -webkit-tap-highlight-color: transparent;
    -webkit-touch-callout: none;
    overscroll-behavior: none;
  }
  
  :global(*) {
    box-sizing: border-box;
  }
  
  :global(button) {
    touch-action: manipulation;
  }
</style>