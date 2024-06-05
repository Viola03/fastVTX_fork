
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os

class Bookkeeping_Browser:
	
	def __init__(self, firefox_profile, headless=True):
		
		self.start_idx = -1

		print("\n\nOpening Bookkeeping Browser...")

		# Create an Options object
		options = Options()

		# Set the Firefox profile
		options.profile = firefox_profile
		options.set_preference("browser.download.folderList", 2)
		options.set_preference("browser.download.dir", os.getcwd())

		if headless:
			options.headless = True

		# Initialize the WebDriver with the options object
		self.driver = webdriver.Firefox(options=options)

		# Navigate to the URL
		self.driver.get('https://lbvobox315.cern.ch/DIRAC/?theme=Crisp&url_state=1|*LHCbDIRAC.BookkeepingBrowser.classes.BookkeepingBrowser:,')

		# Give the page some time to load
		time.sleep(3)

		print("Opened.\n")

		self.rows = None
	
	def get_driver(self):
		return self.driver

	def switch_to_eventType(self):

		print("Switching to EventType...")

		# Click the dropdown button to open the dropdown menu
		dropdown_button = self.driver.find_element(By.ID, 'button-1102-btnInnerEl')  # Replace with the actual ID of the dropdown button
		dropdown_button.click()

		# Wait for the dropdown menu to appear
		dropdown_menu = WebDriverWait(self.driver, 10).until(
			EC.visibility_of_element_located((By.XPATH, '//*[@id="menu-1103-targetEl"]'))  # Replace with the actual XPath of the dropdown menu
		)

		# Click the desired option in the dropdown menu
		desired_option = dropdown_menu.find_element(By.XPATH, '//*[contains(text(), "EventType")]')  # Replace with the text of the desired option
		desired_option.click()
		print("Switched.\n")

		time.sleep(2)

	def open_slash_directory(self):

		print("Opening /...")
		element = self.driver.find_element(By.ID, 'treeview-1100-record-77')
		actions = ActionChains(self.driver)
		actions.double_click(element).perform()
		time.sleep(1)
		print("Open.\n")

		return element, 1

	def open_MC_directory(self):

		print("Opening MC...")
		element = self.driver.find_element(By.ID, 'treeview-1100-record-86')
		actions = ActionChains(self.driver)
		actions.double_click(element).perform()
		time.sleep(1)
		print("Open.\n")

		return element, 2

	def get_branches(self, mother, mother_level, quiet=False):

		branches = {}

		# I think this is the slow command
		if self.rows == None:
			rows_i = self.driver.find_elements(By.XPATH, "//*[starts-with(@id, 'treeview-1100-record')]")
		else:
			rows_all = driver.find_elements(By.XPATH, '//*[@id]')
			# new rows
			rows_i = rows_all - self.rows
			self.rows += rows_i


		break_condition = False

		for row_i_idx, row_i in enumerate(rows_i):

			if self.start_idx != -1 and row_i_idx < self.start_idx:
				continue

			row_i_tr = row_i.find_elements(By.TAG_NAME, 'tr')

			for row_i_tr_i in row_i_tr:

				if int(row_i_tr_i.get_attribute('aria-level')) == mother_level+1:

					branches[row_i_tr_i.get_attribute('data-qtip')] = row_i.get_attribute('id')

				if int(row_i_tr_i.get_attribute('aria-level')) == mother_level and len(list(branches.keys()))>0:
					break_condition = True

			if break_condition:
				break

		if not quiet: print(list(branches.keys()))

		return branches


	def get_decay_target_branch(self, mother, mother_level, target=10000023):

		branches = {}

		rows_i = self.driver.find_elements(By.XPATH, "//*[starts-with(@id, 'treeview-1100-record')]")

		break_condition = False

		for row_i_idx, row_i in enumerate(rows_i):

			row_i_tr = row_i.find_elements(By.TAG_NAME, 'tr')

			for row_i_tr_i in row_i_tr:

				if int(row_i_tr_i.get_attribute('aria-level')) == mother_level+1:

					if int(row_i_tr_i.get_attribute('data-qtip')) == target:

						branches[row_i_tr_i.get_attribute('data-qtip')] = row_i.get_attribute('id')

						break_condition = True

			if break_condition:
				break
		
		self.start_idx = row_i_idx

		return branches

	def get_aria_level(self, element, element_id):

		rows = self.driver.find_elements(By.XPATH, "//*[starts-with(@id, 'treeview-1100-record')]")

		for row in rows:

			if row.get_attribute('id') == element_id:
				row_tr = row.find_elements(By.TAG_NAME, 'tr')

				for row_tr_i in row_tr:
					return int(row_tr_i.get_attribute('aria-level'))

	def expand_directory(self, element_id, title=''):

		print(f"Opening {title} (ID: {element_id})...")
		element = self.driver.find_element(By.ID, element_id)

		self.driver.execute_script("arguments[0].scrollIntoView(true);", element)

		actions = ActionChains(self.driver)
		actions.move_to_element(element).move_by_offset(15, 0).double_click().perform()
		time.sleep(3)
		aria_level = self.get_aria_level(element, element_id)
		print("Open.\n")

		return element, aria_level


	def get_BK_path(self):

		element = self.driver.find_element(By.ID, 'textfield-1149-inputEl')
		input_value = element.get_attribute('value')
		# print(input_value)
		cut_up_path = input_value.replace('evt+std:/','')
		# print(cut_up_path)
		cut_up_path_split = cut_up_path.split('/')
		# print(cut_up_path_split)
		cut_up_path_split.insert(-1, cut_up_path_split.pop(3))
		# print(cut_up_path_split)
		result_string = "/".join(cut_up_path_split)
		return result_string


	def save_file(self):

		element = self.driver.find_element(By.ID, 'toolbar-1190-innerCt')
		actions = ActionChains(self.driver)
		actions.click(element).perform()

		element = self.driver.find_element(By.ID, 'button-1218-btnInnerEl')
		actions = ActionChains(self.driver)
		actions.click(element).perform()




