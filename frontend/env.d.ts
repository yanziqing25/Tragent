/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** The API's own origin when it is served apart from the app, e.g. https://api.example.com. */
  readonly VITE_API_BASE_URL?: string
}
