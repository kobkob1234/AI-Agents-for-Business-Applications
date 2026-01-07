import requests
from bs4 import BeautifulSoup
import os
import time
import argparse
from urllib.parse import urljoin

# Constants
BASE_URL = "https://akama.arc.nasa.gov/ASRSDBOnline/"
FILTER_URL = urljoin(BASE_URL, "QueryWizard_Filter.aspx")
RESULTS_URL = urljoin(BASE_URL, "QueryWizard_Results.aspx")
EXPORT_URL = urljoin(BASE_URL, "QueryWizard_ExportExcel.aspx")
POPUP_URL = urljoin(BASE_URL, "QueryWizard_DatePopup.aspx")
DATA_DIR = "data"

class ASRSDownloader:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

    def get_viewstate_data(self, soup):
        data = {}
        for hidden in soup.find_all("input", type="hidden"):
            if hidden.get("name") in ["__VIEWSTATE", "__EVENTVALIDATION", "__VIEWSTATEGENERATOR", "__EVENTTARGET", "__EVENTARGUMENT"]:
                data[hidden.get("name")] = hidden.get("value", "")
        return data

    def initialize_session(self):
        print("Initializing session...")
        response = self.session.get(FILTER_URL)
        return BeautifulSoup(response.content, "html.parser")

    def add_date_filter(self, soup):
        print("Adding 'Date of Incident' filter...")
        # Get current ViewState
        form_data = self.get_viewstate_data(soup)
        
        # Determine the correct target for the "Add" button for "Date of Incident" (ID 2).
        # Based on manual inspection/previous knowledge, this is typically in a DataGrid or Repeater.
        # We need to find the input type="image" that corresponds to adding item 2.
        # It's usually `DataGrid1$ctl03$imgBtnAdd` or similar if it's the first item.
        # Let's search for the image button in the soup.
        
        # The button for "Date of Incident" has name="2" and id="2"
        add_btn = soup.find("input", {"name": "2", "type": "image"})
        
        if not add_btn:
             # Maybe it's already added? Check for 'Remove' button?
             # Note: The ID changes or checking context might be needed, but usually on this page 
             # the list of available filters persists unless added.
             # If added, the button might disappear or change to a remove button in the "Current Search Items" section.
             print("Add button not found, checking if filter is already active...")
             # Just return soup and hope it's there or handle in set_date_range?
             # If we can't find the add button, we assume we might proceed to set dates if it was already added.
             return soup

        form_data["__EVENTTARGET"] = "2" # Use the name as target
        form_data["2.x"] = "10"
        form_data["2.y"] = "10"

        # Postback to add filter
        response = self.session.post(FILTER_URL, data=form_data)
        return BeautifulSoup(response.content, "html.parser")


    def set_date_range(self, start_year, start_month, end_year, end_month):
        print(f"Setting date range: {start_month}/{start_year} - {end_month}/{end_year}")
        
        # 1. We need to "click" the "Click Here" link. 
        # Actually, in the browser, this opens a popup window. 
        # We can directly go to the Popup URL if we know the statementId.
        # "Date of Incident" ID is usually 2.
        popup_full_url = f"{POPUP_URL}?statementId=2&statementType=Filter"
        
        response = self.session.get(popup_full_url)
        soup = BeautifulSoup(response.content, "html.parser")
        
        form_data = self.get_viewstate_data(soup)
        
        # Set dropdowns
        # DropDownList1: Start Year
        # DropDownList2: Start Month
        # DropDownList3: End Year
        # DropDownList4: End Month
        
        # Note: We need to verify these IDs are correct. 
        # The browser subagent said:
        # DropDownList1: Start Year (e.g., "2024")
        # DropDownList2: Start Month (e.g., "January")
        
        # Let's double check the subagent output.
        # Output says: 
        # DropDownList1: Start Year
        # DropDownList2: Start Month
        # DropDownList3: End Year
        # DropDownList4: End Month
        # This seems standard. I will use this mapping.
        
        form_data["DropDownList1"] = str(start_year)
        form_data["DropDownList2"] = start_month
        form_data["DropDownList3"] = str(end_year)
        form_data["DropDownList4"] = end_month
        
        form_data["SaveButton.x"] = "10"
        form_data["SaveButton.y"] = "10"
        # Remove direct name key if present in ViewState extraction (unlikely for image, but good to ensure)
        if "SaveButton" in form_data: del form_data["SaveButton"]
        
        # Post to Popup URL
        self.session.post(popup_full_url, data=form_data)
        
        # After saving, the session on server is updated.
        # We should refresh the main filter page.
        response = self.session.get(FILTER_URL)
        return BeautifulSoup(response.content, "html.parser")

    def run_search(self, soup):
        print("Running search...")
        form_data = self.get_viewstate_data(soup)
        
        # Find the "Run Search" button
        # Browser subagent: input element with alt='Perform this search and go to the Results page.'
        # It's usually an image button.
        search_btn = soup.find("input", {"alt": "Perform this search and go to the Results page."})
        if not search_btn:
            # Check if we are already on results page?
            if "Results" in (soup.title.string if soup.title else ""):
                 return soup
            raise Exception("Could not find Run Search button")
            
        if "__EVENTTARGET" in form_data: del form_data["__EVENTTARGET"]
        if "__EVENTARGUMENT" in form_data: del form_data["__EVENTARGUMENT"]
        
        form_data[search_btn["name"] + ".x"] = "50"
        form_data[search_btn["name"] + ".y"] = "10"
        
        response = self.session.post(FILTER_URL, data=form_data)
        # Verify we are on results page? URL might change or content.
        if "QueryWizard_Results.aspx" not in response.url and "Your search returned" not in response.text:
             # Sometimes it redirects, sometimes it just renders.
             pass
        return BeautifulSoup(response.content, "html.parser")

    def download_csv(self, filename):
        print("Downloading CSV...")
        # Direct hit to CSV export URL
        export_link = f"{EXPORT_URL}?ExportType=CSV"
        response = self.session.get(export_link, stream=True)
        
        # Check if successful
        if response.status_code == 200:
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved to {filepath}")
        else:
            print(f"Failed to download. Status: {response.status_code}")

    def process_month(self, year, month_name):
        print(f"\n--- Processing {month_name} {year} ---")
        try:
            # 1. Init (Get Filter Page)
            soup = self.initialize_session()
            
            # 2. Add Filter (if not already there - naive check: just try to add, if it fails assume it's there or logic needs update)
            # Actually, for a fresh session, it's empty.
            # But "initialize_session" just gets the page. If we reuse the object, session cookies persist.
            # So subsequent calls *might* have the filter already.
            # Best to check if "Date of Incident" is in the "Current Search Items" list.
            
            # Resetting session for each chunk to ensure clean state is safer but slower. 
            # Let's try to reset the session for each month to avoid state pollution.
            self.session.cookies.clear() # Clear cookies
            
            soup = self.initialize_session()
            soup = self.add_date_filter(soup)
            
            # 3. Set Date Range
            soup = self.set_date_range(year, month_name, year, month_name)
            
            # 4. Run Search
            soup = self.run_search(soup)
            
            # 5. Check results count
            # "Your search returned X ACNs"
            # If 0, skip download?
            
            # 6. Download
            filename = f"asrs_{year}_{month_name}.csv"
            self.download_csv(filename)
            
        except Exception as e:
            print(f"Error processing {month_name} {year}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NASA ASRS Downloader")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--start-month", type=int, required=True, help="1-12")
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--end-month", type=int, required=True, help="1-12")
    
    args = parser.parse_args()
    
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    
    downloader = ASRSDownloader()
    
    # Iterate
    # Simple logic: iterate from start date to end date
    
    current_year = args.start_year
    current_month_idx = args.start_month - 1 # 0-indexed
    
    end_year = args.end_year
    end_month_idx = args.end_month - 1
    
    while (current_year < end_year) or (current_year == end_year and current_month_idx <= end_month_idx):
        month_name = months[current_month_idx]
        downloader.process_month(current_year, month_name)
        
        current_month_idx += 1
        if current_month_idx > 11:
            current_month_idx = 0
            current_year += 1
        
        time.sleep(2) # Politeness delay
