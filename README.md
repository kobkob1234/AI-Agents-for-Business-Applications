# NASA ASRS Downloader
This tool allows you to programmatically download report data from the [NASA Aviation Safety Reporting System (ASRS)](https://asrs.arc.nasa.gov/).

## Setup
1.  Ensure you have Python 3 installed.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Run the script by specifying the start and end dates (Month/Year). The script will iterate through each month in the range and download the data as CSV files into the `data/` directory.

### Example: Download data for a specific month
```bash
python3 asrs_downloader.py --start-year 2024 --start-month 1 --end-year 2024 --end-month 1
```

### Example: Download data for a full year
```bash
python3 asrs_downloader.py --start-year 2023 --start-month 1 --end-year 2023 --end-month 12
```

## How It Works
The script automates the interaction with the NASA ASRS web interface:
1.  **Session Management**: Maintains a session with cookies and ASP.NET ViewState.
2.  **Filter Application**: Adds the "Date of Incident" filter to the search query.
3.  **Date Selection**: Programmatically sets the start and end dates in the popup window logic.
4.  **Search Execution**: Submits the search query.
5.  **Download**: Triggers the CSV export and saves the file.

## Troubleshooting
If the script fails to find buttons or elements, the structure of the NASA website may have changed. The script uses `BeautifulSoup` to parse HTML and simulate form submissions (POST requests) with valid ViewState.
