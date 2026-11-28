import { expect, test } from "@playwright/test";

test("test_observatory_grassmann_cell_drilldown", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("pageerror", (err) => consoleErrors.push(err.message));

  await page.goto("/");
  await expect(page.getByTestId("health-badge")).toContainText("DEMO_REPLAY");
  await expect(page.getByTestId("observatory-grade")).toHaveText("C");

  const cell = page.getByTestId("grassmann-cell-rust-lora__python-lora");
  await expect(cell).toContainText("0.62");
  await cell.click();

  const drilldown = page.getByTestId("grassmann-drilldown");
  await expect(drilldown).toContainText("0.6200");
  await expect(drilldown).toContainText("44.0%");
  await expect(drilldown).toContainText("24.6°");
  await expect(drilldown).toContainText("64.5°");
  expect(consoleErrors).toEqual([]);
});

test("test_knots_lab_rank_slider_updates_energy", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("pageerror", (err) => consoleErrors.push(err.message));

  await page.goto("/");
  await page.getByTestId("nav-knots").click();

  const slider = page.getByTestId("knots-rank-slider");
  await expect(slider).toBeVisible();
  await slider.fill("4");
  await expect(page.getByTestId("knots-rank-label")).toContainText("r = 4");
  await expect(page.getByText("RULE-KNOTS-001")).toBeVisible();
  const lowEnergyText = await page.getByTestId("knots-energy-value").innerText();
  const lowEnergy = parseFloat(lowEnergyText.replace("%", ""));

  await slider.fill("64");
  await expect(page.getByTestId("knots-rank-label")).toContainText("r = 64");
  const highEnergyText = await page.getByTestId("knots-energy-value").innerText();
  const highEnergy = parseFloat(highEnergyText.replace("%", ""));

  expect(highEnergy).toBeGreaterThan(lowEnergy);
  expect(highEnergy).toBeGreaterThan(95);
  expect(consoleErrors).toEqual([]);
});

test("test_live_arena_token_colors_and_inspector", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("pageerror", (err) => consoleErrors.push(err.message));

  await page.goto("/");
  await page.getByTestId("nav-arena").click();
  await page.getByTestId("arena-replay-btn").click();

  const sqlToken = page.getByTestId("arena-token-9");
  await expect(sqlToken).toBeVisible();
  await expect(sqlToken).toHaveAttribute("data-top-adapter", "sql-lora");
  await expect(page.getByTestId("arena-token-0")).toHaveAttribute("data-top-adapter", "rust-lora");
  await expect(page.getByTestId("arena-token-15")).toHaveAttribute("data-top-adapter", "json-lora");

  await sqlToken.click();
  const inspector = page.getByTestId("token-span-inspector");
  await expect(inspector).toContainText("sql-lora");
  await expect(inspector).toContainText("token #9");
  await expect(inspector).toContainText("SELECT user_id");
  expect(consoleErrors).toEqual([]);
});

test("test_arena_xss_script_token_escaped", async ({ page }) => {
  const dialogs: string[] = [];
  page.on("dialog", (d) => {
    dialogs.push(d.message());
    void d.dismiss();
  });

  await page.goto("/");
  await page.getByTestId("nav-arena").click();
  await page.getByTestId("preset-xss_adversarial_probe").click();

  const stream = page.getByTestId("arena-token-stream");
  await expect(stream).toContainText("<script>");
  await expect(stream).toContainText("window.__xss=1;");
  await expect(stream).toContainText("</script>");

  const injectedScripts = await page.locator("#root script").count();
  const xssGlobal = await page.evaluate(
    () => (window as unknown as { __xss?: unknown }).__xss
  );
  expect(injectedScripts).toBe(0);
  expect(xssGlobal).toBeUndefined();
  expect(dialogs).toEqual([]);
});

test("test_roofline_l4_a100_four_arm_charts_render", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("pageerror", (err) => consoleErrors.push(err.message));

  await page.goto("/");
  await page.getByTestId("nav-roofline").click();

  const chart = page.getByTestId("roofline-latency-svg");
  await expect(chart).toBeVisible();
  for (const arm of ["peft_eager_arrow", "vllm_static", "knots_ties", "triton_arrow_sgmv"]) {
    await expect(chart.locator(`polyline[data-arm="${arm}"]`)).toHaveCount(1);
  }
  await expect(page.getByText("Speedup:")).toBeVisible();
  await expect(page.getByText("VRAM Saved:")).toBeVisible();
  expect(consoleErrors).toEqual([]);
});

