import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30000,
  retries: 0,
  use: {
    baseURL: "http://127.0.0.1:18000",
    headless: true,
    trace: "retain-on-failure",
  },
  webServer: {
    command:
      "DEMO_REPLAY=true LORASPEC_UI_DIST=../ui/dist LORASPEC_REPLAY_DIR=../assets/demo_replays ../.venv/bin/uvicorn loraspec.serving.gateway:app --host 127.0.0.1 --port 18000",
    url: "http://127.0.0.1:18000/health",
    reuseExistingServer: !process.env.CI,
    timeout: 30000,
  },
});

