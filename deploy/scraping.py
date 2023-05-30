import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def scrape_comments(url):
    options = Options()
    options.add_argument('--headless')  # Enable headless mode

    # Path to the Chrome driver executable
    driver = webdriver.Chrome(executable_path='chromedriver', options=options)
    driver.get(url)
    
    element_review = driver.find_element(By.XPATH, "/html/body/app-root/div/div/app-product-detail/div/div[2]/div/div[1]/div/div[2]/div")
    driver.execute_script("arguments[0].click();", element_review)
    
    # Find the <a> element
    element = driver.find_element(By.XPATH, "//a[@class='btn btn-pink2']")

    # Get the href attribute value
    href = element.get_attribute('href')

    # Navigate the driver to the href link
    driver.get(href)

    # Scroll to the bottom of the page
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(5)  # Wait for the page to load the new comments

    # Extract comments
    comment_elements = driver.find_elements(
        By.XPATH, "//div[contains(@class, 'review-comment')]//div[contains(@class, 'comment')]")
    comments = []

    for comment_element in comment_elements:
        comment = comment_element.text
        comments.append(comment)

    # Scroll and load more comments until all comments are loaded
    while True:
        # Scroll to the bottom of the page
        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Wait for the page to load the new comments

        # Check if new comments are loaded
        new_comment_elements = driver.find_elements(
            By.XPATH, "//div[contains(@class, 'review-comment')]//div[contains(@class, 'comment')]")
        if len(new_comment_elements) == len(comment_elements):
            break  # No new comments loaded, exit the loop

        # Update comment_elements and extract new comments
        comment_elements = new_comment_elements
        for comment_element in comment_elements[len(comments):]:
            comment = comment_element.text
            comments.append(comment)

    driver.quit()

    return comments
