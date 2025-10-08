from playwright.sync_api import sync_playwright, TimeoutError

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    try:
        # Go to the app
        page.goto("http://localhost:8501", timeout=60000)

        # Wait for the sidebar to be ready
        page.wait_for_selector("section[data-testid='stSidebar']", timeout=60000)

        # --- Create a valid portfolio to prevent the app from halting ---
        # 1. Add a ticker
        ticker_input = page.get_by_label("Entrez un ticker (ex: 'AAPL', 'KER.PA')")
        ticker_input.fill("AAPL")

        # 2. Submit the form to add the ticker
        page.get_by_role("button", name="Ajouter et Valider").click()

        # 3. Wait for the weight input to appear, set it to 100, and press Enter
        weight_input = page.get_by_label("Poids pour AAPL")
        weight_input.wait_for(state="visible", timeout=30000)
        weight_input.fill("100")
        weight_input.press("Enter") # This is the crucial missing step

        # Wait for the app to process the new weight and remove the error message
        # We can wait for the error message to disappear
        error_message = page.get_by_text("Répartition invalide !")
        error_message.wait_for(state="hidden", timeout=10000)

        # --- Now, verify the theme switching ---

        # Take a screenshot of the initial (light) theme with a valid portfolio
        page.screenshot(path="jules-scratch/verification/light-theme.png")

        # Find and click the theme toggle button
        theme_button = page.get_by_role("button", name="🌜")
        theme_button.click()

        # Wait for the theme to change
        page.wait_for_timeout(2000)

        # Take a screenshot of the dark theme
        page.screenshot(path="jules-scratch/verification/dark-theme.png")

        # Find and click the theme toggle button again
        theme_button_dark = page.get_by_role("button", name="🌞")
        theme_button_dark.click()

        # Wait for the theme to change back
        page.wait_for_timeout(2000)

        # Take a final screenshot to ensure it switched back to light
        page.screenshot(path="jules-scratch/verification/light-theme-restored.png")

    except TimeoutError as e:
        # If it times out, save the HTML for debugging
        html_content = page.content()
        with open("jules-scratch/verification/page_source.html", "w") as f:
            f.write(html_content)
        print(f"Timeout occurred: {e}. Saved page source to jules-scratch/verification/page_source.html")
    finally:
        # ---------------------
        context.close()
        browser.close()

with sync_playwright() as playwright:
    run(playwright)